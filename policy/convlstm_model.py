"""ConvLSTM-based policy network scaffold.

This module contains a placeholder ConvLSTM policy architecture for end-to-end
driving experiments. Replace with a researched architecture and tune
hyperparameters for your dataset.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class ConvLSTMPolicy(nn.Module):
    """A small ConvLSTM-like policy network.

    This is intentionally minimal and acts as a scaffold for experiments.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 32, num_outputs: int = 1) -> None:
        super().__init__()
        # example encoder
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, 3, padding=1), nn.ReLU())
        # placeholder recurrent layer; replace with a true ConvLSTM cell
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, num_outputs))

    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, C, H, W) or (B, C, H, W) if single frame.
        """
        if x.dim() == 4:
            # add time dimension
            x = x.unsqueeze(1)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.encoder(x).mean(dim=[2, 3])  # global pooling HxW -> feature
        feat = feat.view(b, t, -1)
        out, hidden = self.rnn(feat, hidden)
        out = out[:, -1, :]
        out = self.head(out)
        return out


def main() -> None:
    """Quick sanity check that model forwards a dummy batch."""
    model = ConvLSTMPolicy()
    dummy = torch.randn(2, 4, 3, 128, 128)
    out = model(dummy)
    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()
