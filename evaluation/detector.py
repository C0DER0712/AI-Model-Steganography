from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from scipy.stats import entropy

from utils.device import get_device
from utils.representation import weights_to_channels

logger = logging.getLogger(__name__)

# Try importing ModelXRay
try:
    import model_xray
    MODEL_XRAY_AVAILABLE = True
except ImportError:
    logger.warning("model_xray module not found. Falling back to statistical detector.")
    MODEL_XRAY_AVAILABLE = False

# Optional SRNet / FSL detector — imported lazily to avoid circular imports
# and to keep the module usable even before those files are installed.
_FSL_AVAILABLE: bool = False
try:
    from models.srnet_detector import SRNetDetector  # noqa: F401 — availability check
    from evaluation.fsl_detector import FSLDetector  # noqa: F401
    _FSL_AVAILABLE = True
except ImportError:
    logger.debug("SRNet/FSL modules not available; _predict_srnet will use fallback.")


@dataclass(frozen=True)
class DetectionResult:
    """Result of a detection prediction.

    Attributes:
        is_malicious: Boolean flag indicating if model is detected as malicious.
        confidence: Softmax confidence in the predicted class (0.0 to 1.0).
        anomaly_score: Continuous score for ROC-AUC, higher = more suspicious.
        dist_to_benign: Distance to the benign centroid or class.
        dist_to_malicious: Distance to the malicious centroid or class.
    """
    is_malicious: bool
    confidence: float
    anomaly_score: float
    dist_to_benign: float
    dist_to_malicious: float


def _save_state_dict_to_temp(state_dict: Dict[str, torch.Tensor], temp_dir: Union[str, Path]) -> Path:
    """Saves a state_dict to a temporary file.

    Args:
        state_dict: The state dictionary to save.
        temp_dir: The temporary directory to save into.

    Returns:
        The path to the saved temporary file.
    """
    temp_dir = Path(temp_dir)
    temp_path = temp_dir / "temp_model.pt"
    torch.save(state_dict, temp_path)
    return temp_path


class ModelXRayDetector:
    """Detector wrapper for evaluating steganographically modified models.

    Provides a wrapper around the official Model X-Ray repository. If the required
    components are not available, it uses a fallback statistical detector based on
    GF image properties.
    """

    def __init__(
        self,
        srnet_weights_path: Optional[Union[str, Path]] = None,
        centroid_path: Optional[Union[str, Path]] = None,
        device: str = 'auto',
        image_size: int = 512,
    ) -> None:
        """Initializes the detector.

        Args:
            srnet_weights_path: Path to SRNet weights (``.pt`` file saved by
                :meth:`~models.srnet_detector.SRNetDetector.save`).
            centroid_path: Path to FSL classifier centroids / embeddings
                (``.pt`` file saved by
                :meth:`~evaluation.fsl_detector.FSLDetector.save`).
            device: Device to use for computation (``"auto"`` / ``"cpu"`` /
                ``"cuda"`` / ``"mps"``).
            image_size: Target side length of the GF image fed to SRNet.
                Defaults to 512; use 256 to match the FSL training config.
        """
        self.device = get_device(device)
        self.image_size = image_size

        self.use_srnet = False
        self._fsl_detector: Optional[Any] = None  # FSLDetector if available

        if srnet_weights_path is not None and centroid_path is not None:
            srnet_path = Path(srnet_weights_path)
            clf_path = Path(centroid_path)

            if srnet_path.exists() and clf_path.exists():
                if _FSL_AVAILABLE:
                    try:
                        from evaluation.fsl_detector import FSLDetector, FSLConfig
                        self._fsl_detector = FSLDetector.load(
                            srnet_path,
                            clf_path,
                            config=FSLConfig(imsize=image_size, device=device),
                            map_location=self.device,
                        )
                        self.use_srnet = True
                        logger.info(
                            "Loaded SRNet FSL detector from %s + %s.",
                            srnet_path,
                            clf_path,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to load SRNet FSL detector: %s. "
                            "Falling back to statistical detector.",
                            exc,
                        )
                else:
                    logger.warning(
                        "SRNet/FSL modules not importable. "
                        "Falling back to statistical detector."
                    )
            else:
                logger.warning(
                    "SRNet weights or centroids not found at provided paths. "
                    "Using statistical detector."
                )
        else:
            logger.info("Using fallback statistical detector.")

    def predict(self, model_path: Union[str, Path]) -> DetectionResult:
        """Predicts whether a saved model is malicious.

        Args:
            model_path: Path to the saved model file.

        Returns:
            A DetectionResult object.
            
        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        gf_image = self._extract_gf_image(model_path)
        
        if self.use_srnet:
            return self._predict_srnet(gf_image)
        else:
            return self._predict_statistical(gf_image)

    def predict_batch(self, model_paths: List[Union[str, Path]]) -> List[DetectionResult]:
        """Predicts whether a batch of saved models are malicious.

        Args:
            model_paths: List of paths to saved model files.

        Returns:
            A list of DetectionResult objects.
        """
        results = []
        for path in model_paths:
            try:
                results.append(self.predict(path))
            except Exception as e:
                logger.error(f"Error predicting for {path}: {e}")
                # Provide a fallback default result or re-raise
                results.append(DetectionResult(False, 0.0, 0.0, 1.0, 0.0))
        return results

    def predict_from_weights(self, state_dict: Dict[str, torch.Tensor]) -> DetectionResult:
        """Predicts from an in-memory state dictionary.

        Args:
            state_dict: The model's state dictionary.

        Returns:
            A DetectionResult object.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = _save_state_dict_to_temp(state_dict, temp_dir)
            return self.predict(temp_path)

    def predict_from_weight_tensor(self, weights: torch.Tensor) -> DetectionResult:
        """Predicts from a flat weight tensor.

        Args:
            weights: A 1D tensor of model weights.

        Returns:
            A DetectionResult object.
        """
        # Convert weight tensor to channels
        channels = weights_to_channels(weights)
        
        # Determine actual side length for image size handling
        c, h, w = channels.shape
        # Just use the raw channels as our GF image equivalent
        gf_image = channels
        
        if self.use_srnet:
            return self._predict_srnet(gf_image)
        else:
            return self._predict_statistical(gf_image)

    def _extract_gf_image(self, model_path: Union[str, Path]) -> np.ndarray:
        """Extracts the GF image from a model file.

        Args:
            model_path: Path to the model file.

        Returns:
            A numpy array representing the GF image of shape (4, H, W).
        """
        # Load model state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Extract and flatten floating-point weights directly
        tensors = []
        for tensor in state_dict.values():
            if tensor.is_floating_point():
                tensors.append(tensor.detach().cpu().to(dtype=torch.float32).reshape(-1))
        
        if not tensors:
            flat_weights = torch.empty(0, dtype=torch.float32)
        else:
            flat_weights = torch.cat(tensors)
        
        # Convert weights to 4-channel image
        channels = weights_to_channels(flat_weights)
        
        return channels

    def _predict_srnet(self, gf_image: np.ndarray) -> DetectionResult:
        """Prediction path using the trained SRNet FSL detector.

        Converts the 4-channel GF representation to a single-channel
        Grayscale-Fourpart image, passes it through the trained
        :class:`~evaluation.fsl_detector.FSLDetector`, and returns a
        fully-populated :class:`DetectionResult`.

        Args:
            gf_image: The GF channel array, shape ``(4, H, W)``.

        Returns:
            A :class:`DetectionResult` from the FSL detector.
        """
        if self._fsl_detector is None:
            logger.warning(
                "_predict_srnet called but FSL detector is not loaded; "
                "falling back to statistical detector."
            )
            return self._predict_statistical(gf_image)

        try:
            from utils.gf_image import channels_to_gf_image, resize_gf_image

            # gf_image may be a numpy array or a tensor coming from
            # predict_from_weight_tensor — normalise to uint8 numpy.
            if isinstance(gf_image, np.ndarray):
                channels = gf_image.astype(np.uint8)
            else:
                import torch as _torch
                channels = (
                    _torch.as_tensor(gf_image)
                    .detach()
                    .cpu()
                    .clamp(0, 255)
                    .byte()
                    .numpy()
                )

            gf_mono = channels_to_gf_image(channels)
            gf_mono = resize_gf_image(gf_mono, self.image_size)
            return self._fsl_detector.predict_gf(gf_mono)

        except Exception as exc:
            logger.warning(
                "SRNet prediction failed (%s); falling back to statistical detector.",
                exc,
            )
            return self._predict_statistical(gf_image)

    def _predict_statistical(self, gf_image: np.ndarray) -> DetectionResult:
        """Prediction path using the fallback statistical detector.

        Args:
            gf_image: The GF image array.

        Returns:
            A DetectionResult object.
        """
        # gf_image is expected to be shape (4, H, W)
        if gf_image.shape[0] != 4:
            raise ValueError(f"Expected 4 channels in GF image, got {gf_image.shape[0]}")

        p0, p1, p2, p3 = gf_image[0], gf_image[1], gf_image[2], gf_image[3]
        
        channels = [p0, p1, p2, p3]
        
        entropies = [self._compute_byte_entropy(c) for c in channels]
        lsb_biases = [self._compute_lsb_bias(c) for c in channels]
        
        kl_divs = []
        for i in range(4):
            for j in range(i + 1, 4):
                kl_divs.append(self._compute_histogram_kl(channels[i], channels[j]))
                
        # Statistical anomaly detection logic
        # High entropy in lower bytes, structured in upper
        # Payload tends to increase entropy and LSB bias in affected channels
        
        # Weight these features into a single anomaly score
        # These weights are heuristic for the fallback mechanism
        entropy_score = sum(entropies) / len(entropies)
        lsb_score = sum(lsb_biases) / len(lsb_biases)
        kl_score = sum(kl_divs) / max(len(kl_divs), 1)
        
        anomaly_score = (0.4 * entropy_score) + (0.4 * lsb_score) + (0.2 * np.clip(kl_score, 0, 1))
        
        # Normalize roughly to [0, 1] using sigmoid
        normalized_score = 1 / (1 + np.exp(-(anomaly_score - 0.5) * 10))
        
        threshold = 0.5
        is_malicious = bool(normalized_score > threshold)
        
        dist_to_benign = 1.0 - normalized_score
        dist_to_malicious = float(normalized_score)
        
        return DetectionResult(
            is_malicious=is_malicious,
            confidence=normalized_score if is_malicious else 1.0 - normalized_score,
            anomaly_score=float(normalized_score),
            dist_to_benign=dist_to_benign,
            dist_to_malicious=dist_to_malicious
        )

    def _compute_byte_entropy(self, channel: np.ndarray) -> float:
        """Computes the Shannon entropy of the byte distribution in a channel.

        Args:
            channel: The byte channel array.

        Returns:
            The normalized entropy (0.0 to 1.0).
        """
        # Ensure values are within 0-255
        vals = np.clip(channel, 0, 255).astype(np.uint8)
        counts = np.bincount(vals.flatten(), minlength=256)
        probs = counts / counts.sum()
        
        # Normalized entropy (max is log2(256) = 8)
        ent = entropy(probs, base=2)
        return float(ent / 8.0)

    def _compute_histogram_kl(self, channel1: np.ndarray, channel2: np.ndarray) -> float:
        """Computes the KL divergence between histograms of two channels.

        Args:
            channel1: First byte channel.
            channel2: Second byte channel.

        Returns:
            The KL divergence.
        """
        vals1 = np.clip(channel1, 0, 255).astype(np.uint8)
        vals2 = np.clip(channel2, 0, 255).astype(np.uint8)
        
        counts1 = np.bincount(vals1.flatten(), minlength=256) + 1e-10
        counts2 = np.bincount(vals2.flatten(), minlength=256) + 1e-10
        
        probs1 = counts1 / counts1.sum()
        probs2 = counts2 / counts2.sum()
        
        return float(entropy(probs1, probs2))

    def _compute_lsb_bias(self, channel: np.ndarray) -> float:
        """Computes the LSB plane bias (deviation from 0.5).

        Args:
            channel: The byte channel array.

        Returns:
            The LSB bias magnitude (0.0 to 0.5).
        """
        vals = np.clip(channel, 0, 255).astype(np.uint8)
        lsbs = vals & 1
        
        prop = lsbs.mean()
        # Deviation from expected 0.5 proportion
        bias = abs(prop - 0.5)
        return float(bias)
