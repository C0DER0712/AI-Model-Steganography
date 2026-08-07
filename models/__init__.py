"""Model architectures for the defensive research codebase."""

from models.decoder import ChunkedPayloadDecoder, DecoderConfig, build_decoder
from models.encoder import EncoderConfig, WeightPayloadEncoder, build_encoder
from models.host_models import HostModelAdapter, HostModelConfig, HostModelName, build_host_model
from models.srnet_detector import SRNetConfig, SRNetDetector, build_srnet_detector

# EmbeddingPipeline and PipelineConfig are importable from models.pipeline directly.
# They are not re-exported here to avoid a circular import through training.__init__.

__all__ = [
    "ChunkedPayloadDecoder",
    "DecoderConfig",
    "EncoderConfig",
    "HostModelAdapter",
    "HostModelConfig",
    "HostModelName",
    "SRNetConfig",
    "SRNetDetector",
    "WeightPayloadEncoder",
    "build_decoder",
    "build_encoder",
    "build_host_model",
    "build_srnet_detector",
]
