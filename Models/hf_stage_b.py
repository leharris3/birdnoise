"""
HuggingFace Hub integration for StageBModel.

Wraps StageBModel with PyTorchModelHubMixin for save_pretrained /
from_pretrained / push_to_hub support.

Only trainable parameters (classifier, temperature, epsilon, gate_network)
are serialized; the frozen NatureLM encoder is re-downloaded on load.
"""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .frozen_audio_encoder import FrozenAudioEncoder
from .stage_b_model import StageBModel


class HFStageBModel(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="finch",
    pipeline_tag="audio-classification",
):
    """
    HuggingFace Hub wrapper around StageBModel.

    All __init__ args are JSON-serializable and stored in config.json.
    The frozen NatureLM encoder is not included in saved checkpoints;
    it is re-downloaded when calling from_pretrained().

    Usage:
        # Create and train
        model = HFStageBModel(num_classes=1054)

        # Save / push
        model.save_pretrained("my-finch-model")
        model.push_to_hub("username/my-finch-model")

        # Reload
        model = HFStageBModel.from_pretrained("username/my-finch-model")

        # Convert from a legacy .pth checkpoint
        model = HFStageBModel.from_checkpoint("best_model.pth", num_classes=1054)
    """

    _ENCODER_PREFIX = "model.encoder."

    def __init__(
        self,
        num_classes: int,
        w_max: float = 2.0,
        init_w: float = 0.0308,
        init_temperature: float = 0.5101,
        init_epsilon: float = -0.049955,
        gate_hidden_dim: int = 64,
        pooling: str = "mean",
    ):
        super().__init__()
        encoder = FrozenAudioEncoder(pooling=pooling)
        self.model = StageBModel(
            encoder=encoder,
            num_classes=num_classes,
            w_max=w_max,
            init_w=init_w,
            init_temperature=init_temperature,
            init_epsilon=init_epsilon,
            gate_hidden_dim=gate_hidden_dim,
        )

    def forward(self, audio, prior_probs, metadata=None):
        return self.model(audio, prior_probs, metadata)

    def state_dict(self, *args, **kwargs):
        """Exclude frozen encoder weights from serialization."""
        full = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in full.items() if not k.startswith(self._ENCODER_PREFIX)
        }

    def load_state_dict(self, state_dict, strict=False, **kwargs):
        """Load with strict=False since encoder weights are excluded."""
        return super().load_state_dict(state_dict, strict=False, **kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, **config_kwargs):
        """Load from a legacy .pth training checkpoint into the HF format.

        Args:
            checkpoint_path: Path to a .pth checkpoint from training.
            **config_kwargs: Model config (num_classes, w_max, etc.).

        Returns:
            HFStageBModel with weights loaded from the checkpoint.
        """
        model = cls(**config_kwargs)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        # Remap legacy key names to current StageBModel names.
        _RENAMES = {"audio_classifier.": "classifier.", "audio_encoder.": "encoder."}
        mapped = {}
        for k, v in state.items():
            for old, new in _RENAMES.items():
                if k.startswith(old):
                    k = new + k[len(old):]
                    break
            # Fix scalar vs 1-d mismatch (e.g. temperature saved as [1]).
            if k in ("temperature", "epsilon") and v.dim() == 1:
                v = v.squeeze(0)
            mapped[f"model.{k}"] = v

        model.load_state_dict(mapped, strict=False)
        return model
