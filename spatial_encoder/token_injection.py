"""
Spatial token injection for Qwen2-VL.

Replaces <obj_0>, <obj_1>, ... placeholder embeddings in the input
with position encoder outputs at every forward pass.
"""

import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration
from typing import Optional

from .position_encoder import PositionEncoder, get_position_encoder
from .coord_transform import extract_spatial_features, normalize_features

# Regex to find <obj_N> tokens in a string
OBJ_TOKEN_RE = re.compile(r"<obj_(\d+)>")


def register_spatial_tokens(tokenizer: AutoTokenizer, max_objects: int = 256) -> list[str]:
    """Add <obj_0>..<obj_N> as special tokens; return the list."""
    tokens = [f"<obj_{i}>" for i in range(max_objects)]
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    return tokens


class SpatialQwen2VL(nn.Module):
    """
    Qwen2-VL with a position encoder for spatial token injection.

    Trainable parameters:
        - position_encoder (all weights)
        - LoRA layers on the base LLM (add via peft externally)

    Frozen:
        - Everything else in Qwen2-VL
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        lora_config=None,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        register_spatial_tokens(self.tokenizer)

        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        # Resize token table to include new <obj_N> tokens
        self.vlm.resize_token_embeddings(len(self.tokenizer))

        cfg = self.vlm.config
        hidden_size = getattr(cfg, "hidden_size",
                      getattr(getattr(cfg, "text_config", cfg), "hidden_size", 3584))
        self.position_encoder = get_position_encoder(hidden_size)

        # Freeze base model
        for p in self.vlm.parameters():
            p.requires_grad_(False)
        # LoRA is added externally via peft after __init__

    # ------------------------------------------------------------------
    # Core injection logic
    # ------------------------------------------------------------------

    def _inject_spatial_embeddings(
        self,
        input_ids: torch.Tensor,          # (B, L)
        inputs_embeds: torch.Tensor,       # (B, L, H)
        spatial_vectors: torch.Tensor,     # (B, N_obj, 6) normalized
    ) -> torch.Tensor:
        """
        For each <obj_i> token in input_ids, replace its embedding
        with position_encoder(spatial_vectors[:, i, :]).
        """
        spatial_embeds = self.position_encoder(
            spatial_vectors.to(inputs_embeds.dtype)
        )  # (B, N_obj, H)

        # Build a lookup: token_id -> object index
        obj_token_ids = {
            self.tokenizer.convert_tokens_to_ids(f"<obj_{i}>"): i
            for i in range(spatial_vectors.shape[1])
        }

        out = inputs_embeds.clone()
        for token_id, obj_idx in obj_token_ids.items():
            mask = (input_ids == token_id)  # (B, L)
            if mask.any():
                # spatial_embeds[:, obj_idx, :] -> (B, H)
                emb = spatial_embeds[:, obj_idx, :]  # (B, H)
                out[mask] = emb.unsqueeze(1).expand_as(
                    out[mask].unsqueeze(0)
                ).reshape(-1, out.shape[-1])

        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        spatial_vectors: Optional[torch.Tensor] = None,  # (B, N_obj, 6) normalized
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Get token embeddings from VLM's embedding table
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)

        if spatial_vectors is not None:
            inputs_embeds = self._inject_spatial_embeddings(
                input_ids, inputs_embeds, spatial_vectors
            )

        return self.vlm(
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        spatial_vectors: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ):
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)

        if spatial_vectors is not None:
            inputs_embeds = self._inject_spatial_embeddings(
                input_ids, inputs_embeds, spatial_vectors
            )

        return self.vlm.generate(
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            **generate_kwargs,
        )


# ------------------------------------------------------------------
# Helper: build one training sample from a scene JSON + QA pair
# ------------------------------------------------------------------

def build_sample(
    scene_json: dict,
    question: str,
    answer: str,
    object_names: list[str],
    camera_name: str = "Camera.001",
) -> dict:
    """
    Prepare a single training sample.

    Args:
        scene_json:    parsed Blender export JSON
        question:      e.g. "<obj_0> is the sofa. <obj_1> is the table. Which is closer?"
        answer:        e.g. "The sofa (<obj_0>) is closer."
        object_names:  ordered list matching <obj_0>, <obj_1>, ...
                       e.g. ["Coffee Table", "Lamp"]

    Returns dict with:
        "prompt":           full prompt string with <obj_N> placeholders
        "answer":           answer string
        "spatial_vectors":  (N_obj, 6) float32 numpy array, normalized
        "norm_params":      (mean, std) for reproducibility
    """
    features = extract_spatial_features(
        scene_json, object_names=object_names, camera_name=camera_name
    )

    # Keep only the requested objects, in order
    ordered_features = {name: features[name] for name in object_names if name in features}
    if len(ordered_features) < len(object_names):
        missing = set(object_names) - set(ordered_features)
        raise ValueError(f"Objects not found in scene: {missing}")

    normalized, mean, std = normalize_features(ordered_features)

    import numpy as np
    spatial_vectors = np.stack([normalized[n] for n in object_names])  # (N, 6)

    return {
        "prompt": question,
        "answer": answer,
        "object_names": object_names,
        "spatial_vectors": spatial_vectors,
        "norm_mean": mean,
        "norm_std": std,
    }
