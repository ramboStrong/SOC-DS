from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


VARIANT_SPECS: Dict[str, Dict[str, object]] = {
    "A0_full_socds": {
        "family": "transport",
        "input_mode": "two_frame",
        "use_depth_guidance": True,
        "use_flow_conditioning": True,
        "use_multiscale_trunk": True,
        "description": (
            "Full two-frame SOC-DS with target-supported depth guidance, "
            "RAFT-conditioned multi-scale fusion, and structured KxQ transport reconstruction."
        ),
    },
    "A1_single_frame_stratified": {
        "family": "direct_stratified",
        "input_mode": "duplicate_current",
        "use_depth_guidance": True,
        "use_flow_conditioning": False,
        "use_multiscale_trunk": True,
        "description": (
            "Single-unique-frame direct three-stratum density baseline. The current frame is "
            "duplicated to preserve the same 1024-channel fusion interface, and branch fusion is uniform."
        ),
    },
    "A2_no_depth": {
        "family": "transport",
        "input_mode": "two_frame",
        "use_depth_guidance": False,
        "use_flow_conditioning": True,
        "use_multiscale_trunk": True,
        "description": "SOC-DS independently retrained without the target-supported depth branch.",
    },
    "A3_no_flow": {
        "family": "transport",
        "input_mode": "two_frame",
        "use_depth_guidance": True,
        "use_flow_conditioning": False,
        "use_multiscale_trunk": True,
        "description": (
            "SOC-DS independently retrained with uniform multi-scale branch fusion and no RAFT conditioning."
        ),
    },
    "A4_direct_stratified_head": {
        "family": "direct_stratified",
        "input_mode": "two_frame",
        "use_depth_guidance": True,
        "use_flow_conditioning": True,
        "use_multiscale_trunk": True,
        "description": (
            "Matched two-frame direct three-channel density head. It retains the same shared backbone, "
            "depth fusion, dilation branches, and RAFT-conditioned fusion as A0, but replaces the 30-channel "
            "transport tensor and reconstruction operator with a direct three-channel density output."
        ),
    },
    "A5_single_projected_density": {
        "family": "direct_total",
        "input_mode": "two_frame",
        "use_depth_guidance": True,
        "use_flow_conditioning": True,
        "use_multiscale_trunk": True,
        "description": (
            "Matched two-frame single projected-density baseline with the same backbone, depth fusion, "
            "multi-scale branches, and RAFT-conditioned fusion as A0."
        ),
    },
    "A6_direct_count_regression": {
        "family": "count",
        "input_mode": "two_frame",
        "use_depth_guidance": True,
        "use_flow_conditioning": False,
        "use_multiscale_trunk": False,
        "description": (
            "Task-specific two-frame direct total-count regression baseline. It retains the shared "
            "appearance encoder and target-supported depth fusion, but removes the dense multi-scale "
            "transport trunk and RAFT conditioning before global pooling and scalar regression."
        ),
    },
}


@dataclass
class ModelOutput:
    family: str
    prediction: torch.Tensor
    depth: Optional[torch.Tensor]
    intermediate: Optional[Dict[str, torch.Tensor]] = None


class ContextualModule(nn.Module):
    def __init__(self, features: int, out_features: int = 512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.weight_net = nn.Conv2d(features, features, kernel_size=1)

    @staticmethod
    def _make_scale(features: int, size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(features, features, kernel_size=1, bias=False),
        )

    def _make_weight(self, feature: torch.Tensor, scale_feature: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.weight_net(feature - scale_feature))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        h, w = feats.shape[-2:]
        multi_scales = [
            F.interpolate(stage(feats), size=(h, w), mode="bilinear", align_corners=False)
            for stage in self.scales
        ]
        weights = [self._make_weight(feats, scale_feature) for scale_feature in multi_scales]
        numerator = sum(sf * wt for sf, wt in zip(multi_scales, weights))
        denominator = sum(weights) + 1e-8
        fused_context = numerator / denominator
        return self.relu(self.bottleneck(torch.cat([fused_context, feats], dim=1)))


class WeightGenerationNetwork(nn.Module):
    def __init__(self, input_channels: int = 2, out_channels: int = 3, target_size=(45, 80)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(target_size)

    def forward(self, flow: torch.Tensor, return_score: bool = False):
        x = F.relu(self.conv1(flow), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.pool(x)
        score = F.relu(self.conv3(x), inplace=True)
        weight = F.softmax(score, dim=1)
        if return_score:
            return weight, score
        return weight


def make_layers(cfg, in_channels: int = 3, batch_norm: bool = False) -> nn.Sequential:
    layers = []
    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        conv = nn.Conv2d(in_channels, value, kernel_size=3, padding=1)
        if batch_norm:
            layers.extend([conv, nn.BatchNorm2d(value), nn.ReLU(inplace=True)])
        else:
            layers.extend([conv, nn.ReLU(inplace=True)])
        in_channels = value
    return nn.Sequential(*layers)


class SharedFeatureBackbone(nn.Module):

    def __init__(
        self,
        input_mode: str = "two_frame",
        use_depth_guidance: bool = True,
        use_pretrained_frontend: bool = True,
    ):
        super().__init__()
        if input_mode not in {"two_frame", "duplicate_current"}:
            raise ValueError(f"Unsupported input_mode: {input_mode}")

        self.input_mode = input_mode
        self.use_depth_guidance = use_depth_guidance
        self.frontend_cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.frontend = make_layers(self.frontend_cfg)
        self.context = ContextualModule(512, 512)

        encoded_channels = 1024
        self.rgb_layer = nn.Sequential(
            nn.Conv2d(encoded_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        if use_depth_guidance:
            self.depth_decoder = nn.Sequential(
                nn.Conv2d(encoded_channels, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.depth_head = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.ReLU(inplace=True))
            self.depth_layer = nn.Sequential(
                nn.Conv2d(64, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.depth_fusion = nn.Sequential(
                nn.Conv2d(1024, 64, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2, kernel_size=1),
            )

        self._initialize_weights()
        if use_pretrained_frontend:
            self._load_vgg16_frontend_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _load_vgg16_frontend_weights(self) -> None:
        try:
            source = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            source = models.vgg16(pretrained=True)
        source_dict = {
            key[9:]: value
            for key, value in source.state_dict().items()
            if key.startswith("features.") and key[9:] in self.frontend.state_dict()
        }
        self.frontend.load_state_dict(source_dict, strict=False)

    def encode(self, prev_img: torch.Tensor, curr_img: torch.Tensor):
        curr_raw = self.frontend(curr_img)
        curr_feat = self.context(curr_raw)

        if self.input_mode == "two_frame":
            prev_raw = self.frontend(prev_img)
            prev_feat = self.context(prev_raw)
        else:
            prev_raw = curr_raw
            prev_feat = curr_feat

        encoded = torch.cat([prev_feat, curr_feat], dim=1)
        rgb_feature = self.rgb_layer(encoded)

        depth = None
        depth_latent = None
        depth_feature = None
        fusion_weight = None
        if self.use_depth_guidance:
            depth_latent = self.depth_decoder(encoded)
            depth = self.depth_head(depth_latent)
            depth_feature = self.depth_layer(depth_latent)
            fusion_weight = F.softmax(
                self.depth_fusion(torch.cat([rgb_feature, depth_feature], dim=1)), dim=1
            )
            refined = rgb_feature * fusion_weight[:, 0:1] + depth_feature * fusion_weight[:, 1:2]
        else:
            refined = rgb_feature

        intermediate = {
            "prev_raw": prev_raw,
            "curr_raw": curr_raw,
            "prev_feat": prev_feat,
            "curr_feat": curr_feat,
            "encoded": encoded,
            "rgb_feature": rgb_feature,
            "depth_latent": depth_latent,
            "depth_feature": depth_feature,
            "fusion_weight": fusion_weight,
            "refined": refined,
        }
        return refined, depth, intermediate


class MultiScaleFusionTrunk(nn.Module):

    def __init__(self, use_flow_conditioning: bool = True):
        super().__init__()
        self.use_flow_conditioning = bool(use_flow_conditioning)

        def branch(dilation: int):
            return nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        self.branch1 = branch(1)
        self.branch2 = branch(2)
        self.branch3 = branch(3)
        self.weight_net = WeightGenerationNetwork(2, 3) if self.use_flow_conditioning else None
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, refined: torch.Tensor, flow: Optional[torch.Tensor]):
        b1 = self.branch1(refined)
        b2 = self.branch2(refined)
        b3 = self.branch3(refined)

        if self.use_flow_conditioning:
            if flow is None:
                raise ValueError("This model requires a RAFT flow tensor.")
            assert self.weight_net is not None
            weight, score = self.weight_net(flow, return_score=True)
            if weight.shape[-2:] != refined.shape[-2:]:
                weight = F.interpolate(weight, size=refined.shape[-2:], mode="bilinear", align_corners=False)
                weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
                score = F.interpolate(score, size=refined.shape[-2:], mode="bilinear", align_corners=False)
        else:
            score = None
            weight = torch.full(
                (refined.shape[0], 3, refined.shape[2], refined.shape[3]),
                1.0 / 3.0,
                device=refined.device,
                dtype=refined.dtype,
            )

        fused = b1 * weight[:, 0:1] + b2 * weight[:, 1:2] + b3 * weight[:, 2:3]
        return fused, {
            "branch1": b1,
            "branch2": b2,
            "branch3": b3,
            "branch_score": score,
            "branch_weight": weight,
            "branch_fused": fused,
        }


class StructuredTransportModel(nn.Module):
    family = "transport"

    def __init__(
        self,
        input_mode: str = "two_frame",
        use_depth_guidance: bool = True,
        use_flow_conditioning: bool = True,
        use_pretrained_frontend: bool = True,
        num_layers: int = 3,
        channels_per_layer: int = 10,
    ):
        super().__init__()
        self.backbone = SharedFeatureBackbone(
            input_mode=input_mode,
            use_depth_guidance=use_depth_guidance,
            use_pretrained_frontend=use_pretrained_frontend,
        )
        self.trunk = MultiScaleFusionTrunk(use_flow_conditioning=use_flow_conditioning)
        self.output_head = nn.Sequential(
            nn.Conv2d(64, num_layers * channels_per_layer, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self._initialize_head()

    def _initialize_head(self):
        for module in self.output_head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, prev_img, curr_img, flow=None, return_intermediate: bool = False) -> ModelOutput:
        refined, depth, intermediate = self.backbone.encode(prev_img, curr_img)
        fused, trunk_intermediate = self.trunk(refined, flow)
        propagation = self.output_head(fused)
        if return_intermediate:
            intermediate.update(trunk_intermediate)
        else:
            intermediate = None
        return ModelOutput(self.family, propagation, depth, intermediate)


class MatchedDirectDensityModel(nn.Module):
    def __init__(
        self,
        output_channels: int,
        input_mode: str = "two_frame",
        use_depth_guidance: bool = True,
        use_flow_conditioning: bool = True,
        use_pretrained_frontend: bool = True,
    ):
        super().__init__()
        if output_channels not in {1, 3}:
            raise ValueError("output_channels must be 1 or 3")
        self.family = "direct_stratified" if output_channels == 3 else "direct_total"
        self.backbone = SharedFeatureBackbone(
            input_mode=input_mode,
            use_depth_guidance=use_depth_guidance,
            use_pretrained_frontend=use_pretrained_frontend,
        )
        self.trunk = MultiScaleFusionTrunk(use_flow_conditioning=use_flow_conditioning)
        self.head = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        for module in self.head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, prev_img, curr_img, flow=None, return_intermediate: bool = False) -> ModelOutput:
        refined, depth, intermediate = self.backbone.encode(prev_img, curr_img)
        fused, trunk_intermediate = self.trunk(refined, flow)
        prediction = self.head(fused)
        if return_intermediate:
            intermediate.update(trunk_intermediate)
        else:
            intermediate = None
        return ModelOutput(self.family, prediction, depth, intermediate)


class MatchedCountRegressionModel(nn.Module):

    family = "count"

    def __init__(
        self,
        input_mode: str = "two_frame",
        use_depth_guidance: bool = True,
        use_flow_conditioning: bool = False,
        use_pretrained_frontend: bool = True,
        use_multiscale_trunk: bool = False,
        max_count: float = 500.0,
    ):
        super().__init__()
        self.max_count = float(max_count)
        self.use_multiscale_trunk = bool(use_multiscale_trunk)
        self.backbone = SharedFeatureBackbone(
            input_mode=input_mode,
            use_depth_guidance=use_depth_guidance,
            use_pretrained_frontend=use_pretrained_frontend,
        )

        if self.use_multiscale_trunk:
            self.trunk = MultiScaleFusionTrunk(use_flow_conditioning=use_flow_conditioning)
            head_channels = 64
        else:
            self.trunk = None
            head_channels = 512

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(head_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, prev_img, curr_img, flow=None, return_intermediate: bool = False) -> ModelOutput:
        refined, depth, intermediate = self.backbone.encode(prev_img, curr_img)

        if self.trunk is not None:
            fused, trunk_intermediate = self.trunk(refined, flow)
            if return_intermediate:
                intermediate.update(trunk_intermediate)
        else:
            fused = refined
            if return_intermediate:
                intermediate["branch_fused"] = fused

        prediction = self.head(fused).squeeze(1) * self.max_count
        if not return_intermediate:
            intermediate = None
        return ModelOutput(self.family, prediction, depth, intermediate)


def build_model(variant: str, use_pretrained_frontend: bool = True) -> nn.Module:
    if variant not in VARIANT_SPECS:
        raise KeyError(f"Unknown variant '{variant}'. Available: {list(VARIANT_SPECS)}")
    spec = VARIANT_SPECS[variant]
    family = str(spec["family"])
    common = dict(
        input_mode=str(spec["input_mode"]),
        use_depth_guidance=bool(spec["use_depth_guidance"]),
        use_flow_conditioning=bool(spec["use_flow_conditioning"]),
        use_pretrained_frontend=use_pretrained_frontend,
    )
    if family == "transport":
        return StructuredTransportModel(**common)
    if family == "direct_stratified":
        return MatchedDirectDensityModel(output_channels=3, **common)
    if family == "direct_total":
        return MatchedDirectDensityModel(output_channels=1, **common)
    if family == "count":
        return MatchedCountRegressionModel(
            **common,
            use_multiscale_trunk=bool(spec.get("use_multiscale_trunk", False)),
        )
    raise RuntimeError(f"Unhandled family: {family}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
