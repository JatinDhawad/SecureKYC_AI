"""
SecureKYC AI - Multimodal KYC Fusion Neural Network (MultimodalKYCFusionNet)
PyTorch deep learning model fusing spatial 3D lip landmark dynamics,
temporal sequence features, and cross-modal phoneme-viseme desync metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalKYCFusionNet(nn.Module):
    """
    Multimodal neural network architecture for real-time deepfake injection defense.

    Inputs:
        - lip_sequence: Tensor of shape (batch_size, seq_len) or (batch_size, 1, seq_len)
          representing 3D lip aperture distance trajectory D_lip(t).
        - desync_features: Tensor of shape (batch_size, num_desync_features)
          containing overall desync score, failed closure ratio, mean aperture, etc.

    Output:
        - Deepfake probability tensor of shape (batch_size, 1) in [0, 1].
          Label 0.0 = Authentic / Real, Label 1.0 = Synthetic / Deepfake Injection.
    """

    def __init__(
        self,
        seq_len: int = 100,
        desync_feature_dim: int = 3,
        conv_out_channels: int = 32,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 2,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.desync_feature_dim = desync_feature_dim

        # 1D Convolutional Spatial Feature Encoder
        self.conv1d_1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv1d_2 = nn.Conv1d(
            in_channels=16, out_channels=conv_out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm1d(conv_out_channels)

        # Bi-LSTM Temporal Sequence Encoder
        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0.0,
        )

        # Visual sequence embedding dimension (bidirectional = hidden_dim * 2)
        visual_embedding_dim = lstm_hidden_dim * 2

        # Multimodal Fusion Classification Head
        fusion_in_dim = visual_embedding_dim + desync_feature_dim

        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        lip_sequence: torch.Tensor,
        desync_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for MultimodalKYCFusionNet.

        Args:
            lip_sequence: Tensor of shape (B, seq_len) or (B, 1, seq_len)
                representing 3D lip aperture distance trajectory D_lip(t).
            desync_features: Tensor of shape (B, 3) containing cross-modal desynchronization metrics:
                - desync_features[:, 0]: Mean penalty (S_desync = mean(Penalty(t) for t in T_bilabial))
                - desync_features[:, 1]: Maximum penalty across T_bilabial (max_t Penalty(t))
                - desync_features[:, 2]: Fraction of bilabials with penalty > 0 (failed closure ratio)

        Returns:
            Tensor of shape (B, 1) deepfake confidence probability score in range [0.0, 1.0].
        """
        # Format lip_sequence to (B, 1, L)
        if lip_sequence.ndim == 2:
            x_vis = lip_sequence.unsqueeze(1)  # (B, 1, L)
        else:
            x_vis = lip_sequence

        # Apply 1D Spatial Convolutions
        x_vis = F.relu(self.bn1(self.conv1d_1(x_vis)))  # (B, 16, L)
        x_vis = self.pool1(x_vis)                        # (B, 16, L/2)
        x_vis = F.relu(self.bn2(self.conv1d_2(x_vis)))  # (B, conv_out, L/2)

        # Transpose for PyTorch LSTM: (B, L_reduced, channels)
        x_vis = x_vis.transpose(1, 2)                    # (B, L/2, conv_out)

        # Process through Bi-LSTM
        lstm_out, _ = self.lstm(x_vis)                  # (B, L/2, lstm_hidden * 2)

        # Global average pooling over time dimension
        visual_embedding = torch.mean(lstm_out, dim=1)  # (B, lstm_hidden * 2)

        # Ensure desync_features has proper dimension
        if desync_features.ndim == 1:
            desync_features = desync_features.unsqueeze(0)

        # Multimodal Feature Fusion
        fused_features = torch.cat([visual_embedding, desync_features], dim=1)

        # Classification Logits -> Sigmoid
        out = self.fusion_fc(fused_features)
        return out


class VisionOnlyNet(nn.Module):
    """Ablation Model 1: Vision-Only Sequence Classifier (No Desync Metrics)."""

    def __init__(self, seq_len: int = 100):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, lip_sequence: torch.Tensor, desync_features: torch.Tensor = None) -> torch.Tensor:
        if lip_sequence.ndim == 2:
            x = lip_sequence.unsqueeze(1)
        else:
            x = lip_sequence
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        emb = torch.mean(lstm_out, dim=1)
        return self.fc(emb)


class DesyncOnlyNet(nn.Module):
    """Ablation Model 2: Desync Metric Only Linear Classifier."""

    def __init__(self, desync_feature_dim: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(desync_feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, lip_sequence: torch.Tensor = None, desync_features: torch.Tensor = None) -> torch.Tensor:
        if desync_features.ndim == 1:
            desync_features = desync_features.unsqueeze(0)
        return self.fc(desync_features)


class ConstrainedLateFusion(nn.Module):
    """
    Constrained Late-Fusion Meta-Learner combining VisionOnlyNet and DesyncOnlyNet.

    Decision formula:
        z = w_vis * P_vis + w_desync * P_desync + intercept
        P_fused = sigmoid(z)

    Non-negativity constraints (w_vis >= 0, w_desync >= 0) guarantee catastrophic-failure
    resistance by preventing gradient inversion on small validation cohorts, preserving
    the acoustic desync baseline under domain shift.
    """

    def __init__(
        self,
        vision_net: VisionOnlyNet,
        desync_net: DesyncOnlyNet,
        w_vis: float = 5.3688,
        w_desync: float = 4.9111,
        intercept: float = -6.0411,
    ):
        super().__init__()
        self.vision_net = vision_net
        self.desync_net = desync_net
        self.register_buffer("w_vis", torch.tensor(w_vis, dtype=torch.float32))
        self.register_buffer("w_desync", torch.tensor(w_desync, dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(intercept, dtype=torch.float32))

    def forward(
        self,
        lip_sequence: torch.Tensor,
        desync_features: torch.Tensor,
    ) -> torch.Tensor:
        p_vis = self.vision_net(lip_sequence)
        p_desync = self.desync_net(desync_features=desync_features)
        z = self.w_vis * p_vis + self.w_desync * p_desync + self.intercept
        return torch.sigmoid(z)

