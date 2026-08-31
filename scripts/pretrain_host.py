#!/usr/bin/env python3
"""Fine-tune an ImageNet host model on CIFAR-10 before payload embedding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct execution with ``python scripts/pretrain_host.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models.host_models import build_host_model
from utils.seed import set_seed


IMAGENET_NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an ImageNet-pretrained MobileNetV2 on CIFAR-10."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download CIFAR-10 when it is not already present.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/host_pretraining/mobilenet_v2_cifar10.pt"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _cifar_root(path: Path) -> Path:
    """Return torchvision's root when given its batches directory directly."""
    return path.parent if path.name == "cifar-10-batches-py" else path


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        logits = model(images.to(device))
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        total += labels.numel()
    return correct / total


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive.")
    set_seed(args.seed)
    device = _resolve_device(args.device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), IMAGENET_NORMALIZE,
    ])
    data_root = _cifar_root(args.data_root)
    train_ds = CIFAR10(str(data_root), train=True, download=args.download, transform=transform)
    test_ds = CIFAR10(str(data_root), train=False, download=args.download, transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    host = build_host_model("mobilenet_v2", num_classes=10, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(host.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        host.train()
        total_loss = batches = 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(host(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        scheduler.step()
        accuracy = _accuracy(host, test_loader, device)
        print(f"epoch {epoch} | train_loss={total_loss / batches:.4f} | test_accuracy={accuracy:.2%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": host.state_dict(), "architecture": "mobilenet_v2", "num_classes": 10},
        args.output,
    )
    print(f"saved host checkpoint: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
