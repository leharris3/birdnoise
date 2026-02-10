import torch
import torch.nn as nn
import torch.nn.functional as F

from .frozen_audio_encoder import FrozenAudioEncoder
from .prior_gating_network import PriorGatingNetwork


class StageBModel(nn.Module):
    """
    Fusion model with gating network for Stage D training.

    final_logits = audio_logits / temperature + w(a,x,t) * log(prior + epsilon)
    """

    def __init__(
        self,
        encoder: FrozenAudioEncoder,
        num_classes: int,
        w_max: float = 2.0,
        init_w: float = 0.0308,
        init_temperature: float = 0.5101,
        init_epsilon: float = -0.049955,
        gate_hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes

        # Linear probe (classifier head)
        self.classifier = nn.Linear(encoder.encoder_dim, num_classes)

        # Learnable fusion parameters
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.epsilon = nn.Parameter(torch.tensor(init_epsilon))

        # Gating network for w(a,x,t)
        self.gate_network = PriorGatingNetwork(
            input_dim=12,
            hidden_dim=gate_hidden_dim,
            w_max=w_max,
            init_w=init_w,
        )

        print(f"StageBModel initialized:")
        print(f"  temperature: {init_temperature:.4f}")
        print(f"  epsilon: {init_epsilon:.6f}")
        print(f"  w_max: {w_max:.2f}")
        print(f"  init_w: {init_w:.4f}")

    def compute_audio_features(self, audio_logits: torch.Tensor) -> torch.Tensor:
        """Extract features from audio logits for gating."""
        probs = F.softmax(audio_logits, dim=1)
        max_prob = probs.max(dim=1)[0]
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

        # Margin: difference between top-1 and top-2
        top2_probs = torch.topk(probs, k=2, dim=1)[0]
        margin = top2_probs[:, 0] - top2_probs[:, 1]

        return torch.stack([max_prob, entropy, margin], dim=1)

    def compute_prior_features(self, prior_probs: torch.Tensor) -> torch.Tensor:
        """Extract features from prior probabilities for gating."""
        max_prob = prior_probs.max(dim=1)[0]
        entropy = -(prior_probs * torch.log(prior_probs + 1e-10)).sum(dim=1)

        # Margin: difference between top-1 and top-2
        top2_probs = torch.topk(prior_probs, k=2, dim=1)[0]
        margin = top2_probs[:, 0] - top2_probs[:, 1]

        return torch.stack([max_prob, entropy, margin], dim=1)

    def forward(
        self,
        audio: torch.Tensor,
        prior_probs: torch.Tensor,
        metadata: list = None,
    ):
        """
        Args:
            audio: (batch_size, audio_length)
            prior_probs: (batch_size, num_classes)
            metadata: list of metadata dicts for gating

        Returns:
            final_logits: (batch_size, num_classes)
            audio_logits: (batch_size, num_classes)
            w: (batch_size,) - gating weights for logging
        """
        # Encode audio and classify
        features = self.encoder(audio)
        audio_logits = self.classifier(features.float())

        # Temperature-scaled audio logits
        temp = self.temperature.abs().clamp(min=1e-8)
        audio_logits_scaled = audio_logits / temp

        # Compute gating weight w(a,x,t)
        if metadata is not None:
            audio_features = self.compute_audio_features(audio_logits_scaled)
            prior_features = self.compute_prior_features(prior_probs)
            w = self.gate_network(audio_features, prior_features, metadata)
        else:
            # Fallback to mean initial weight if no metadata
            w = torch.full(
                (audio_logits.size(0),),
                self.gate_network.init_w,
                device=audio_logits.device,
            )

        # Safe epsilon: ensure prior + eps > 0 to avoid log(0)
        # Clamp epsilon to ensure prior_probs + eps is always positive
        min_prior = prior_probs.min()
        safe_eps = self.epsilon.clamp(min=-min_prior.item() + 1e-8)
        log_prior = torch.log(prior_probs + safe_eps)

        # Fused logits
        final_logits = audio_logits_scaled + w.unsqueeze(1) * log_prior

        return final_logits, audio_logits, w
