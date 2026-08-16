from __future__ import annotations

from typing import Dict

from model_variants import StructuredTransportModel, count_trainable_parameters


SENSITIVITY_SPEC: Dict[str, object] = {
    "variant": "A0_full_socds",
    "family": "transport",
    "input_mode": "two_frame",
    "use_depth_guidance": True,
    "use_flow_conditioning": True,
    "description": (
        "Full SOC-DS architecture with a configurable number of camera-centric "
        "distance strata; Q=10 local transport components are retained per stratum."
    ),
}


def build_sensitivity_model(
    num_strata: int,
    *,
    channels_per_stratum: int = 10,
    use_pretrained_frontend: bool = True,
):
    model = StructuredTransportModel(
        input_mode="two_frame",
        use_depth_guidance=True,
        use_flow_conditioning=True,
        use_pretrained_frontend=use_pretrained_frontend,
        num_layers=int(num_strata),
        channels_per_layer=int(channels_per_stratum),
    )
    model.num_strata = int(num_strata)
    model.channels_per_stratum = int(channels_per_stratum)
    return model


__all__ = ["SENSITIVITY_SPEC", "build_sensitivity_model", "count_trainable_parameters"]
