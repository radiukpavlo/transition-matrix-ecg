"""Multilead triad classifier definition."""

from __future__ import annotations


def build_model(
    in_leads: int = 12,
    triad_length: int = 3,
    samples_per_beat: int = 256,
    latent_dim: int = 512,
    num_classes: int = 9,
):
    try:
        import torch  # type: ignore
        from torch import nn  # type: ignore
    except ImportError as exc:
        raise RuntimeError("build_model requires torch") from exc

    class TriadCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = in_leads * triad_length
            self.features = nn.Sequential(
                nn.Conv1d(channels, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.penultimate = nn.Linear(256, latent_dim)
            self.classifier = nn.Linear(latent_dim, num_classes)

        def forward(self, x):  # type: ignore[no-untyped-def]
            x = self.features(x).squeeze(-1)
            preactivation = self.penultimate(x)
            logits = self.classifier(torch.relu(preactivation))
            return logits, preactivation

    return TriadCNN()
