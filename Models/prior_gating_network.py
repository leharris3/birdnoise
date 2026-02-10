import math

import torch
import torch.nn as nn


class PriorGatingNetwork(nn.Module):
    """
    Gating network to learn w(a,x,t) based on audio and prior features.

    Input: 12-dim vector
        - 3 audio features: max_prob, entropy, margin
        - 3 prior features: max_prob, entropy, margin
        - 6 metadata features: sin/cos(day), sin/cos(hour), lat, lon

    Output: scalar w in [0, w_max]
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        w_max: float = 2.0,
        init_w: float = 0.0308,
    ):
        super().__init__()
        self.w_max = w_max
        self.init_w = init_w

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize final layer bias so sigmoid outputs ~init_w/w_max
        # sigmoid(bias) * w_max = init_w => bias = logit(init_w / w_max)
        target_sigmoid = init_w / w_max
        target_sigmoid = max(1e-6, min(1 - 1e-6, target_sigmoid))  # Clamp to valid range
        init_bias = math.log(target_sigmoid / (1 - target_sigmoid))  # Inverse sigmoid

        # Get final layer and set bias
        final_layer = self.mlp[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, init_bias)
        print(f"Gating network initialized: target w={init_w:.4f}, bias={init_bias:.4f}")

    def forward(
        self,
        audio_features: torch.Tensor,
        prior_features: torch.Tensor,
        metadata: list,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, 3) - [max_prob, entropy, margin]
            prior_features: (batch_size, 3) - [max_prob, entropy, margin]
            metadata: list of dicts with 'day_of_year', 'hour', 'latitude', 'longitude'

        Returns:
            w: (batch_size,) - gate weights in [0, w_max]
        """
        device = audio_features.device

        # Extract metadata safely
        def safe_float_list(key, default=0.0):
            result = []
            for m in metadata:
                val = m.get(key, default)
                try:
                    result.append(float(val))
                except (ValueError, TypeError):
                    result.append(float(default))
            return result

        day_of_year = torch.tensor(
            safe_float_list("day_of_year", 182), dtype=torch.float32, device=device
        )
        hour = torch.tensor(
            safe_float_list("hour", 12), dtype=torch.float32, device=device
        )
        lat = torch.tensor(
            safe_float_list("latitude", 0.0), dtype=torch.float32, device=device
        )
        lon = torch.tensor(
            safe_float_list("longitude", 0.0), dtype=torch.float32, device=device
        )

        # Normalize lat/lon to [-1, 1]
        lat_norm = lat / 90.0
        lon_norm = lon / 180.0

        # Sin/cos encoding for cyclical features
        day_sin = torch.sin(2 * math.pi * day_of_year / 365.25)
        day_cos = torch.cos(2 * math.pi * day_of_year / 365.25)
        hour_sin = torch.sin(2 * math.pi * hour / 24.0)
        hour_cos = torch.cos(2 * math.pi * hour / 24.0)

        # Concatenate all features: [audio(3), prior(3), metadata(6)] = 12
        meta_features = torch.stack(
            [day_sin, day_cos, hour_sin, hour_cos, lat_norm, lon_norm], dim=1
        )
        combined = torch.cat([audio_features, prior_features, meta_features], dim=1)

        # MLP to gate logit
        gate_logit = self.mlp(combined).squeeze(-1)

        # Sigmoid and scale to [0, w_max]
        w = torch.sigmoid(gate_logit) * self.w_max

        return w
