"""Few-Shot Learning detector for Model X-Ray steganography evaluation.

This module implements the FSL detection framework described in the Model X-Ray
paper: a :class:`SRNetDetector` backbone is trained with triplet loss on small
sets of benign and steganographically-modified model checkpoints, then
classification is performed at test time using a Nearest-Centroid or 1-NN rule
in the learned embedding space.

Key classes
-----------
:class:`FSLDataset`
    Dataset of GF images with binary labels (0 = benign, 1 = malicious).

:class:`TripletDataset`
    Wraps :class:`FSLDataset` to yield ``(anchor, positive, negative)``
    triplets for triplet-loss training.

:class:`NearestCentroidClassifier`
    Fits class centroids from training embeddings; classifies by min distance.

:class:`NNClassifier`
    1-NN classifier in embedding space.

:class:`FSLDetector`
    High-level API: wraps SRNet backbone, triplet training, and classification
    into a single :meth:`~FSLDetector.fit` / :meth:`~FSLDetector.predict` API.
    Produces :class:`~evaluation.detector.DetectionResult` objects.

Triplet loss
    L(A, P, N) = max( d(A, P) - d(A, N) + margin, 0 )
    where d is Euclidean distance and the default margin is 0.5.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.srnet_detector import SRNetDetector, SRNetConfig
from utils.gf_image import gf_image_to_tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FSLConfig:
    """Configuration for :class:`FSLDetector`.

    Attributes:
        imsize: GF image side length fed to SRNet (default 256).
        embedding_dim: SRNet output embedding dimension.
        triplet_margin: Margin α in the triplet loss (default 0.5).
        learning_rate: Adam learning rate for SRNet fine-tuning.
        num_epochs: Triplet training epochs.
        batch_size: Triplet batch size.
        num_triplets_per_epoch: Triplets sampled per epoch.
        classifier: ``"centroid"`` (Nearest-Centroid) or ``"1nn"`` (1-NN).
        device: Torch device string (``"auto"`` / ``"cpu"`` / ``"cuda"`` /
            ``"mps"``).
        seed: Random seed for reproducible triplet sampling.
    """

    imsize: int = 256
    embedding_dim: int = 512
    triplet_margin: float = 0.5
    learning_rate: float = 6e-5
    num_epochs: int = 50
    batch_size: int = 32
    num_triplets_per_epoch: int = 200
    classifier: str = "centroid"   # "centroid" | "1nn"
    device: str = "auto"
    seed: int = 42


# ---------------------------------------------------------------------------
# FSL Dataset
# ---------------------------------------------------------------------------


class FSLDataset(Dataset):
    """Dataset of GF images with binary class labels.

    Each item is a ``(tensor, label)`` pair where:

    * ``tensor`` is a float32 tensor with shape ``(1, imsize, imsize)`` and
      values in ``[0, 255]``.
    * ``label`` is an integer — ``0`` for benign, ``1`` for malicious.

    GF images can be supplied either as pre-computed numpy arrays or via
    callable factories (e.g. load-from-disk functions).

    Args:
        benign_gf_images: Sequence of ``(H, W)`` uint8 numpy arrays for
            benign (clean) model checkpoints.
        malicious_gf_images: Sequence of ``(H, W)`` uint8 numpy arrays for
            steganographically-modified model checkpoints.
        imsize: Target spatial size; images are resized if needed.
    """

    def __init__(
        self,
        benign_gf_images: Sequence[np.ndarray],
        malicious_gf_images: Sequence[np.ndarray],
        imsize: int = 256,
    ) -> None:
        self.imsize = imsize
        self._images: List[np.ndarray] = []
        self._labels: List[int] = []

        for img in benign_gf_images:
            self._images.append(self._prepare(img))
            self._labels.append(0)

        for img in malicious_gf_images:
            self._images.append(self._prepare(img))
            self._labels.append(1)

        self._class_indices: Dict[int, List[int]] = {0: [], 1: []}
        for idx, label in enumerate(self._labels):
            self._class_indices[label].append(idx)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        img = self._images[index]
        tensor = gf_image_to_tensor(img)   # (1, H, W) float32 in [0, 255]
        return tensor, self._labels[index]

    @property
    def class_indices(self) -> Dict[int, List[int]]:
        """Dict mapping label → list of dataset indices with that label."""
        return self._class_indices

    # ------------------------------------------------------------------

    def _prepare(self, img: np.ndarray) -> np.ndarray:
        """Ensure the image is uint8 and sized ``(imsize, imsize)``."""
        import cv2  # local import to avoid top-level overhead

        arr = np.asarray(img, dtype=np.uint8)
        if arr.shape != (self.imsize, self.imsize):
            arr = cv2.resize(arr, (self.imsize, self.imsize),
                             interpolation=cv2.INTER_NEAREST)
        return arr


# ---------------------------------------------------------------------------
# Triplet Dataset
# ---------------------------------------------------------------------------


class TripletDataset(Dataset):
    """Generates ``(anchor, positive, negative)`` triplets from :class:`FSLDataset`.

    Triplets are sampled once per epoch and cached internally.  Call
    :meth:`resample` to generate a fresh set of triplets.

    Args:
        fsl_dataset: Labelled GF-image dataset.
        num_triplets: Number of triplets to generate.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        fsl_dataset: FSLDataset,
        num_triplets: int = 200,
        seed: int = 42,
    ) -> None:
        self.fsl_dataset = fsl_dataset
        self.num_triplets = num_triplets
        self.rng = random.Random(seed)
        self._triplets: List[Tuple[int, int, int]] = []
        self.resample()

    def resample(self) -> None:
        """Re-draw all triplets (call at the start of each epoch)."""
        ci = self.fsl_dataset.class_indices
        benign = ci.get(0, [])
        malicious = ci.get(1, [])

        if len(benign) < 1 or len(malicious) < 1:
            raise ValueError(
                "FSLDataset must contain at least one benign and one malicious "
                "sample to form triplets."
            )

        triplets: List[Tuple[int, int, int]] = []
        for _ in range(self.num_triplets):
            # 50-50 chance of a benign or malicious anchor.
            if self.rng.random() < 0.5:
                anchor_label, neg_label = 0, 1
            else:
                anchor_label, neg_label = 1, 0

            anchor_pool = ci[anchor_label]
            neg_pool = ci[neg_label]

            anchor = self.rng.choice(anchor_pool)
            # Positive: same class as anchor, different index if possible.
            if len(anchor_pool) > 1:
                positive = anchor
                while positive == anchor:
                    positive = self.rng.choice(anchor_pool)
            else:
                positive = anchor  # degenerate: only one sample of this class
            negative = self.rng.choice(neg_pool)
            triplets.append((anchor, positive, negative))

        self._triplets = triplets

    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        a_idx, p_idx, n_idx = self._triplets[index]
        a, _ = self.fsl_dataset[a_idx]
        p, _ = self.fsl_dataset[p_idx]
        n, _ = self.fsl_dataset[n_idx]
        return a, p, n


# ---------------------------------------------------------------------------
# Triplet loss
# ---------------------------------------------------------------------------


def triplet_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 0.5,
) -> Tensor:
    """Compute the triplet loss over a batch.

    L(A, P, N) = mean( max( ‖A-P‖₂ - ‖A-N‖₂ + margin, 0 ) )

    Args:
        anchor: Embeddings of anchor samples ``(B, D)``.
        positive: Embeddings of positive (same-class) samples ``(B, D)``.
        negative: Embeddings of negative (different-class) samples ``(B, D)``.
        margin: Minimum distance margin between positive and negative pairs.

    Returns:
        Scalar loss tensor.
    """
    d_ap = torch.norm(anchor - positive, p=2, dim=1)
    d_an = torch.norm(anchor - negative, p=2, dim=1)
    loss = F.relu(d_ap - d_an + margin)
    return loss.mean()


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


class NearestCentroidClassifier:
    """Nearest-Centroid classifier in SRNet embedding space.

    Fits per-class mean centroids from labelled training embeddings, then
    classifies a query by returning the label of the nearest centroid.

    Args:
        embedding_dim: Embedding dimension (for shape validation).
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        self.embedding_dim = embedding_dim
        self.centroids: Dict[int, Tensor] = {}

    def fit(self, embeddings: Tensor, labels: Sequence[int]) -> None:
        """Compute per-class centroids from training embeddings.

        Args:
            embeddings: Float tensor ``(N, D)``.
            labels: Sequence of integer class labels of length ``N``.
        """
        label_tensor = torch.tensor(labels, dtype=torch.long)
        for cls in torch.unique(label_tensor).tolist():
            cls = int(cls)
            mask = label_tensor == cls
            self.centroids[cls] = embeddings[mask].mean(dim=0).cpu()

    def predict(
        self, embedding: Tensor
    ) -> Tuple[int, float, float]:
        """Classify a single embedding vector.

        Args:
            embedding: Float tensor ``(D,)`` or ``(1, D)``.

        Returns:
            Tuple of ``(predicted_label, dist_to_benign, dist_to_malicious)``.
        """
        if embedding.ndim == 2:
            embedding = embedding.squeeze(0)
        emb = embedding.cpu()

        dists: Dict[int, float] = {}
        for cls, centroid in self.centroids.items():
            dists[cls] = float(torch.norm(emb - centroid, p=2).item())

        predicted = min(dists, key=lambda k: dists[k])
        d_benign = dists.get(0, float("inf"))
        d_malicious = dists.get(1, float("inf"))
        return predicted, d_benign, d_malicious

    def save(self, path: Union[str, Path]) -> None:
        """Persist centroids to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"centroids": self.centroids, "embedding_dim": self.embedding_dim}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NearestCentroidClassifier":
        """Load persisted centroids from disk."""
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        obj = cls(embedding_dim=ckpt["embedding_dim"])
        obj.centroids = ckpt["centroids"]
        return obj


class NNClassifier:
    """1-Nearest-Neighbour classifier in SRNet embedding space.

    Args:
        embedding_dim: Embedding dimension.
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        self.embedding_dim = embedding_dim
        self._train_embeddings: Optional[Tensor] = None
        self._train_labels: Optional[List[int]] = None

    def fit(self, embeddings: Tensor, labels: Sequence[int]) -> None:
        """Store training embeddings and labels.

        Args:
            embeddings: Float tensor ``(N, D)``.
            labels: Integer labels, length ``N``.
        """
        self._train_embeddings = embeddings.cpu()
        self._train_labels = list(labels)

    def predict(
        self, embedding: Tensor
    ) -> Tuple[int, float, float]:
        """Classify by nearest neighbour.

        Args:
            embedding: Float tensor ``(D,)`` or ``(1, D)``.

        Returns:
            ``(predicted_label, dist_to_nearest_benign, dist_to_nearest_malicious)``
        """
        if self._train_embeddings is None or self._train_labels is None:
            raise RuntimeError("NNClassifier must be fitted before predicting.")

        if embedding.ndim == 2:
            embedding = embedding.squeeze(0)
        emb = embedding.cpu()

        dists = torch.norm(self._train_embeddings - emb.unsqueeze(0), p=2, dim=1)
        nearest_idx = int(dists.argmin().item())
        predicted = self._train_labels[nearest_idx]

        label_tensor = torch.tensor(self._train_labels)
        mask_benign = label_tensor == 0
        mask_malicious = label_tensor == 1

        d_benign = float(dists[mask_benign].min().item()) if mask_benign.any() else float("inf")
        d_malicious = float(dists[mask_malicious].min().item()) if mask_malicious.any() else float("inf")
        return predicted, d_benign, d_malicious


# ---------------------------------------------------------------------------
# Paper metric
# ---------------------------------------------------------------------------


def weighted_metric(
    per_severity_accuracy: Dict[int, float],
    *,
    max_severity: int = 23,
) -> float:
    """Compute the Model X-Ray weighted detection metric.

    The paper weights lower mantissa attack severities more heavily:

    ``WM = 0.5 * (a0 + sum((s-i+1) * ai) / (s*(s+1)/2))``

    ``a0`` is the benign/zero-severity accuracy and ``ai`` is the accuracy
    measured at severity ``i``.  Missing severities are treated as zero,
    matching the reference implementation.

    Args:
        per_severity_accuracy: Mapping from severity ``0..max_severity`` to
            an accuracy in ``[0, 1]``.
        max_severity: Maximum severity included in the weighted sum.

    Returns:
        Weighted metric in the range ``[0, 1]``.

    Raises:
        ValueError: If ``max_severity`` is not positive or an accuracy is
            outside ``[0, 1]``.
    """
    if max_severity < 1:
        raise ValueError("max_severity must be at least 1.")
    if any(float(value) < 0.0 or float(value) > 1.0
           for value in per_severity_accuracy.values()):
        raise ValueError("Per-severity accuracies must be in the range [0, 1].")

    a0 = float(per_severity_accuracy.get(0, 0.0))
    denominator = max_severity * (max_severity + 1) / 2.0
    weighted_sum = sum(
        (max_severity - severity + 1)
        * float(per_severity_accuracy.get(severity, 0.0))
        for severity in range(1, max_severity + 1)
    )
    return float(0.5 * (a0 + weighted_sum / denominator))


# ---------------------------------------------------------------------------
# High-level FSLDetector
# ---------------------------------------------------------------------------


class FSLDetector:
    """High-level Few-Shot Learning detector for Model X-Ray evaluation.

    Wraps an :class:`~models.srnet_detector.SRNetDetector` backbone, a
    :class:`TripletDataset`-based training loop, and a configurable
    classifier (Nearest Centroid or 1-NN) into a single cohesive API.

    Usage::

        fsl = FSLDetector(FSLConfig(num_epochs=50))
        fsl.fit(benign_gf_images, malicious_gf_images)
        result = fsl.predict_gf(query_gf_image)

    Args:
        config: :class:`FSLConfig` hyperparameters.
        srnet: Optional pre-built :class:`~models.srnet_detector.SRNetDetector`.
            When ``None``, a fresh model is constructed from ``config``.
    """

    def __init__(
        self,
        config: Optional[FSLConfig] = None,
        srnet: Optional[SRNetDetector] = None,
    ) -> None:
        self.config = config or FSLConfig()
        self.device = _resolve_device(self.config.device)

        if srnet is not None:
            self.srnet = srnet.to(self.device)
        else:
            self.srnet = SRNetDetector(
                SRNetConfig(
                    in_channels=1,
                    embedding_dim=self.config.embedding_dim,
                    num_classes=None,
                )
            ).to(self.device)

        self._classifier: Optional[Union[NearestCentroidClassifier, NNClassifier]] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        benign_gf_images: Sequence[np.ndarray],
        malicious_gf_images: Sequence[np.ndarray],
    ) -> List[float]:
        """Train SRNet with triplet loss and fit the classifier.

        Args:
            benign_gf_images: GF images ``(H, W)`` of clean model checkpoints.
            malicious_gf_images: GF images ``(H, W)`` of modified checkpoints.

        Returns:
            List of per-epoch mean triplet losses.
        """
        cfg = self.config
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        fsl_ds = FSLDataset(benign_gf_images, malicious_gf_images, imsize=cfg.imsize)
        triplet_ds = TripletDataset(fsl_ds, num_triplets=cfg.num_triplets_per_epoch, seed=cfg.seed)

        optimizer = torch.optim.Adam(self.srnet.parameters(), lr=cfg.learning_rate)
        epoch_losses: List[float] = []

        logger.info(
            "FSL training: %d benign + %d malicious samples, %d epochs.",
            len(fsl_ds.class_indices.get(0, [])),
            len(fsl_ds.class_indices.get(1, [])),
            cfg.num_epochs,
        )

        self.srnet.train()
        for epoch in range(1, cfg.num_epochs + 1):
            triplet_ds.resample()
            loader = DataLoader(
                triplet_ds,
                batch_size=min(cfg.batch_size, len(triplet_ds)),
                shuffle=True,
                drop_last=False,
            )
            epoch_loss = 0.0
            num_batches = 0
            for anchors, positives, negatives in loader:
                anchors = anchors.to(self.device)
                positives = positives.to(self.device)
                negatives = negatives.to(self.device)

                emb_a = self.srnet.embed(anchors)
                emb_p = self.srnet.embed(positives)
                emb_n = self.srnet.embed(negatives)

                loss = triplet_loss(emb_a, emb_p, emb_n, margin=cfg.triplet_margin)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            mean_loss = epoch_loss / max(num_batches, 1)
            epoch_losses.append(mean_loss)

            if epoch % max(1, cfg.num_epochs // 10) == 0 or epoch == cfg.num_epochs:
                logger.info("  Epoch %d/%d  triplet_loss=%.4f", epoch, cfg.num_epochs, mean_loss)

        # ---- Fit classifier on training embeddings ----
        self.srnet.eval()
        self._fit_classifier(fsl_ds)
        self._is_fitted = True
        logger.info("FSLDetector fitted.")
        return epoch_losses

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_gf(self, gf_image: np.ndarray) -> "DetectionResult":
        """Classify a single GF image.

        Args:
            gf_image: ``uint8`` array with shape ``(H, W)``.

        Returns:
            :class:`~evaluation.detector.DetectionResult`.
        """
        from evaluation.detector import DetectionResult  # avoid circular at module level

        if not self._is_fitted:
            raise RuntimeError("FSLDetector must be fitted before predicting.")

        tensor = gf_image_to_tensor(gf_image).unsqueeze(0).to(self.device)  # (1, 1, H, W)
        self.srnet.eval()
        with torch.no_grad():
            emb = self.srnet.embed(tensor).squeeze(0)  # (D,)

        predicted, d_benign, d_malicious = self._classifier.predict(emb)

        # Convert distances to a continuous anomaly score in [0, 1].
        # A sample that is farther from benign (and therefore closer to the
        # malicious class) should receive the higher score.
        total = d_benign + d_malicious
        if total > 0:
            anomaly_score = float(d_benign / total)
        else:
            anomaly_score = 0.5

        confidence = max(d_benign, d_malicious) / (total + 1e-8)
        confidence = float(min(confidence, 1.0))

        return DetectionResult(
            is_malicious=bool(predicted == 1),
            confidence=confidence,
            anomaly_score=anomaly_score,
            dist_to_benign=float(d_benign),
            dist_to_malicious=float(d_malicious),
        )

    def predict_channels(self, channels: np.ndarray) -> "DetectionResult":
        """Classify from a ``(4, S, S)`` byte-channel array.

        Args:
            channels: uint8 array produced by
                :func:`~utils.representation.weights_to_channels`.

        Returns:
            :class:`~evaluation.detector.DetectionResult`.
        """
        from utils.gf_image import channels_to_gf_image, resize_gf_image

        gf = channels_to_gf_image(channels)
        gf = resize_gf_image(gf, self.config.imsize)
        return self.predict_gf(gf)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save SRNet weights and classifier centroids.

        Args:
            output_dir: Directory to save ``srnet.pt`` and ``classifier.pt``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.srnet.save(out / "srnet.pt")
        if hasattr(self._classifier, "save"):
            self._classifier.save(out / "classifier.pt")
        else:
            # NNClassifier: save embeddings + labels
            torch.save(
                {
                    "embeddings": self._classifier._train_embeddings,
                    "labels": self._classifier._train_labels,
                    "embedding_dim": self._classifier.embedding_dim,
                },
                out / "classifier.pt",
            )
        logger.info("FSLDetector saved to %s", out)

    @classmethod
    def load(
        cls,
        srnet_path: Union[str, Path],
        classifier_path: Union[str, Path],
        config: Optional[FSLConfig] = None,
        *,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "FSLDetector":
        """Load a saved :class:`FSLDetector` from disk.

        Args:
            srnet_path: Path to ``srnet.pt``.
            classifier_path: Path to ``classifier.pt``.
            config: Optional :class:`FSLConfig`.
            map_location: Device remapping for :func:`torch.load`.

        Returns:
            A fitted :class:`FSLDetector`.
        """
        srnet = SRNetDetector.load(srnet_path, map_location=map_location)
        cfg = config or FSLConfig()
        detector = cls(config=cfg, srnet=srnet)

        # Classifier checkpoints are generated locally by ``save``.  Explicitly
        # disable PyTorch's safe weights-only mode for version-independent
        # loading of the stored classifier metadata.
        try:
            ckpt = torch.load(
                classifier_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            ckpt = torch.load(classifier_path, map_location="cpu")
        if "centroids" in ckpt:
            classifier: Union[NearestCentroidClassifier, NNClassifier] = NearestCentroidClassifier.load(classifier_path)
        else:
            nn_cls = NNClassifier(ckpt["embedding_dim"])
            nn_cls._train_embeddings = ckpt["embeddings"]
            nn_cls._train_labels = ckpt["labels"]
            classifier = nn_cls

        detector._classifier = classifier
        detector._is_fitted = True
        return detector

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_classifier(self, fsl_ds: FSLDataset) -> None:
        """Embed all training samples and fit the chosen classifier."""
        all_embeddings: List[Tensor] = []
        all_labels: List[int] = []

        loader = DataLoader(fsl_ds, batch_size=16, shuffle=False)
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                embs = self.srnet.embed(imgs)
                all_embeddings.append(embs.cpu())
                all_labels.extend(labels.tolist())

        embeddings = torch.cat(all_embeddings, dim=0)

        if self.config.classifier == "1nn":
            clf: Union[NearestCentroidClassifier, NNClassifier] = NNClassifier(
                embedding_dim=self.config.embedding_dim
            )
        else:
            clf = NearestCentroidClassifier(embedding_dim=self.config.embedding_dim)

        clf.fit(embeddings, all_labels)
        self._classifier = clf

    @property
    def is_fitted(self) -> bool:
        """True if the detector has been fitted."""
        return self._is_fitted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)
