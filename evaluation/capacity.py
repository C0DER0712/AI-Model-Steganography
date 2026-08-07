"""Embedding capacity calculation for steganographic host models.

Computes the bits-per-parameter (BPP) embedding capacity and related metrics
that quantify how much payload a given host model can carry relative to its
weight count and the bit error rate achieved during training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from utils.weights import extract_weights, flatten_weights


@dataclass(frozen=True)
class CapacityResult:
    """Embedding capacity metrics for one host model.

    Attributes:
        num_float_parameters: Total floating-point parameter count.
        num_float_bits: Total bits available in the weight representation
            (32 × num_float_parameters).
        payload_bits: Number of payload bits embedded.
        bits_per_parameter: Payload bits divided by parameter count.
        embedding_rate: Payload bits as a fraction of total available bits.
        effective_payload_bits: Payload bits recoverable at the observed BER.
            Computed as ``payload_bits × (1 - ber)`` (information-theoretic
            lower bound).
        bit_error_rate: Observed BER at evaluation time, or ``None`` if not
            provided.
    """

    num_float_parameters: int
    num_float_bits: int
    payload_bits: int
    bits_per_parameter: float
    embedding_rate: float
    effective_payload_bits: float
    bit_error_rate: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        """Return all fields as a flat dictionary."""
        return {
            "num_float_parameters": self.num_float_parameters,
            "num_float_bits": self.num_float_bits,
            "payload_bits": self.payload_bits,
            "bits_per_parameter": self.bits_per_parameter,
            "embedding_rate": self.embedding_rate,
            "effective_payload_bits": self.effective_payload_bits,
            "bit_error_rate": self.bit_error_rate,
        }

    def __str__(self) -> str:
        ber_str = f"{self.bit_error_rate:.6f}" if self.bit_error_rate is not None else "N/A"
        return (
            f"CapacityResult(\n"
            f"  float parameters : {self.num_float_parameters:,}\n"
            f"  total weight bits : {self.num_float_bits:,}\n"
            f"  payload bits      : {self.payload_bits:,}\n"
            f"  bits/parameter    : {self.bits_per_parameter:.6f}\n"
            f"  embedding rate    : {self.embedding_rate:.6f}\n"
            f"  effective bits    : {self.effective_payload_bits:.1f}\n"
            f"  bit error rate    : {ber_str}\n"
            f")"
        )


def compute_capacity(
    model: nn.Module,
    payload_bits: int,
    *,
    bit_error_rate: float | None = None,
) -> CapacityResult:
    """Compute embedding capacity metrics for a host model.

    Args:
        model: PyTorch model whose floating-point weights carry the payload.
        payload_bits: Number of bits embedded in the model.
        bit_error_rate: Optional observed BER; used to compute effective
            recoverable payload.

    Returns:
        :class:`CapacityResult` with all capacity metrics.

    Raises:
        ValueError: If ``payload_bits`` exceeds available weight bits.
    """
    if payload_bits < 0:
        raise ValueError("payload_bits must be non-negative.")

    weight_records = extract_weights(model)
    flat = flatten_weights(weight_records)
    num_params = flat.numel()
    total_bits = num_params * 32  # float32: 32 bits per parameter

    if payload_bits > total_bits:
        raise ValueError(
            f"payload_bits={payload_bits} exceeds the total available "
            f"weight bits={total_bits} for this model."
        )

    bpp = payload_bits / num_params if num_params > 0 else 0.0
    rate = payload_bits / total_bits if total_bits > 0 else 0.0
    ber = bit_error_rate if bit_error_rate is not None else 0.0
    effective = payload_bits * (1.0 - ber)

    return CapacityResult(
        num_float_parameters=num_params,
        num_float_bits=total_bits,
        payload_bits=payload_bits,
        bits_per_parameter=bpp,
        embedding_rate=rate,
        effective_payload_bits=effective,
        bit_error_rate=bit_error_rate,
    )


def max_payload_bits(model: nn.Module, target_ber: float = 0.01) -> int:
    """Estimate the maximum payload size achievable below a target BER.

    This is a theoretical upper bound: it assumes the full weight bit budget
    is available and that the observed BER is uniformly distributed.  In
    practice the achievable capacity is determined by training.

    Args:
        model: Host model.
        target_ber: Maximum acceptable bit error rate.

    Returns:
        Maximum payload bit count as an integer.
    """
    if not 0.0 <= target_ber < 1.0:
        raise ValueError("target_ber must be in [0, 1).")

    weight_records = extract_weights(model)
    flat = flatten_weights(weight_records)
    total_bits = flat.numel() * 32
    # Shannon capacity: H(BER) corrected channel capacity
    channel_capacity = 1.0 - _binary_entropy(target_ber)
    return math.floor(total_bits * channel_capacity)


def capacity_summary(
    model: nn.Module,
    payload_bits_list: list[int],
    *,
    bit_error_rates: list[float] | None = None,
) -> list[CapacityResult]:
    """Compute capacity results for multiple payload sizes.

    Args:
        model: Host model.
        payload_bits_list: List of payload sizes to evaluate.
        bit_error_rates: Optional parallel list of BER values.

    Returns:
        List of :class:`CapacityResult` in the same order.
    """
    bers = bit_error_rates or [None] * len(payload_bits_list)
    if len(bers) != len(payload_bits_list):
        raise ValueError("payload_bits_list and bit_error_rates must have the same length.")

    return [
        compute_capacity(model, bits, bit_error_rate=ber)
        for bits, ber in zip(payload_bits_list, bers)
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _binary_entropy(p: float) -> float:
    """Binary entropy function H(p) = -p log2(p) - (1-p) log2(1-p)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)
