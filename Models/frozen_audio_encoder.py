import torch
import torch.nn as nn


class FrozenAudioEncoder(nn.Module):
    """Frozen NatureLM audio encoder."""

    def __init__(self, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling

        print("Loading NatureLM-audio encoder...")
        from NatureLM.models import NatureLM
        self.naturelm = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")

        # Freeze all parameters
        for param in self.naturelm.parameters():
            param.requires_grad = False
        self.naturelm.eval()

        # Get encoder dimension
        self.encoder_dim = self.naturelm.llama_model.config.hidden_size
        print(f"Encoder dimension: {self.encoder_dim}")

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract pooled features from audio."""
        self.naturelm.eval()
        device = next(self.naturelm.parameters()).device
        audio = audio.to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            audio_embeds, audio_atts = self.naturelm.encode_audio(audio)

        # Pool over sequence dimension
        if self.pooling == "mean":
            mask = audio_atts.unsqueeze(-1).float()
            features = (audio_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling == "max":
            features = audio_embeds.max(dim=1)[0]
        else:  # cls
            features = audio_embeds[:, 0, :]

        return features
