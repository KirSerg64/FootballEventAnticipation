import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowFusion(nn.Module):
    def __init__(self, input_dim, flow_dim=2, hidden_dim=32):
        super().__init__()
        self.flow_dim = flow_dim
        # InstanceNorm2d normalises each [H, W] map independently per channel
        # per sample, so frame i never influences the normalisation of fame j.
        # This is correct because the B*S leading dimension mixes time steps,
        # and motion statistics vary significantly across frames.
        # affine=True adds learnable scale/shift so the network can adjust range.
        self.flow_norm = nn.InstanceNorm2d(flow_dim, affine=True)
        self.conv_1 = nn.Conv2d(input_dim + flow_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, rgb_features, flow_features):
        if flow_features.shape[-2:] != rgb_features.shape[-2:]:
            flow_features = F.interpolate(
                flow_features, size=rgb_features.shape[-2:],
                mode="bilinear", align_corners=False)
        flow_features = self.flow_norm(flow_features)
        fused = torch.cat((rgb_features, flow_features), dim=1)
        x = self.conv_1(fused)
        x = self.relu(x)
        x = self.conv_2(x)
        return x
