"""
Automatic QA generation from Blender scene spatial features.
All answers are derived purely from coordinate math — zero human annotation.
"""

import json
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .coord_transform import extract_spatial_features, SpatialFeature


@dataclass
class QASample:
    question: str
    answer: str
    qa_type: str                  # "direction" | "distance_compare" | "nearest" | "count"
    object_names: list[str]       # objects referenced, in <obj_0>, <obj_1>... order
    raw_features: dict            # name -> SpatialFeature, for build_sample()


# ------------------------------------------------------------------
# Direction helpers
# ------------------------------------------------------------------

def _azimuth_to_direction(az_deg: float) -> str:
    if az_deg > 30:
        return "右侧"
    elif az_deg > 10:
        return "右前方"
    elif az_deg > -10:
        return "正前方"
    elif az_deg > -30:
        return "左前方"
    else:
        return "左侧"


def _elevation_to_vertical(el_deg: float) -> str:
    if el_deg > 20:
        return "上方"
    elif el_deg < -20:
        return "下方"
    return ""


# ------------------------------------------------------------------
# Filter: objects usable for QA
# ------------------------------------------------------------------

GENERIC_PREFIXES = (
    "Cube", "_3D", "Plane", "Point", "Area",
    "String", "Venetian", "Obj3d", "Light",
    "Camera", "VenetianFrame",
)


def _is_semantic(name: str) -> bool:
    return not any(name.startswith(p) for p in GENERIC_PREFIXES)


def _display_name(name: str, idx: int, use_placeholder: bool) -> str:
    """Return either '<obj_N> (name)' or just the name."""
    if use_placeholder:
        return f"<obj_{idx}>"
    return name


# ------------------------------------------------------------------
# Four QA types
# ------------------------------------------------------------------

def gen_direction(features: dict[str, SpatialFeature],
                  semantic_only: bool = True) -> Optional[QASample]:
    """Q: <obj_0> 在你的哪个方向？"""
    pool = {k: v for k, v in features.items()
            if (not semantic_only or _is_semantic(k)) and v.z > 0}
    if not pool:
        return None

    name, feat = random.choice(list(pool.items()))
    az_deg = np.degrees(feat.azimuth)
    el_deg = np.degrees(feat.elevation)
    direction = _azimuth_to_direction(az_deg)
    vertical = _elevation_to_vertical(el_deg)
    full_dir = f"{direction}{'、' + vertical if vertical else ''}"

    return QASample(
        question=f"<obj_0> 在你的哪个方向？",
        answer=f"<obj_0> 在你的{full_dir}（方位角 {az_deg:+.1f}°，距离 {feat.distance:.1f}m）。",
        qa_type="direction",
        object_names=[name],
        raw_features={name: feat},
    )


def gen_distance_compare(features: dict[str, SpatialFeature],
                         semantic_only: bool = True) -> Optional[QASample]:
    """Q: <obj_0> 和 <obj_1> 哪个离你更近？"""
    pool = list({k: v for k, v in features.items()
                 if not semantic_only or _is_semantic(k)}.items())
    if len(pool) < 2:
        return None

    (n0, f0), (n1, f1) = random.sample(pool, 2)

    if f0.distance < f1.distance:
        closer_ref, farther_ref = "<obj_0>", "<obj_1>"
        diff = f1.distance - f0.distance
    else:
        closer_ref, farther_ref = "<obj_1>", "<obj_0>"
        diff = f0.distance - f1.distance

    return QASample(
        question=f"<obj_0> 和 <obj_1> 哪个离你更近？",
        answer=f"{closer_ref} 更近（近 {diff:.1f}m）。",
        qa_type="distance_compare",
        object_names=[n0, n1],
        raw_features={n0: f0, n1: f1},
    )


def gen_nearest(features: dict[str, SpatialFeature], offset: int = 0) -> Optional[QASample]:
    """Q: 在给定的几个物体中，哪个距你最近？
    offset: shift the candidate window so repeated calls yield different questions.
    """
    if not features:
        return None

    sorted_feats = sorted(features.items(), key=lambda kv: kv[1].distance)
    n_candidates = 4
    start = offset * 2                          # slide window by 2 each call
    candidates = sorted_feats[start: start + n_candidates]
    if not candidates:
        candidates = sorted_feats[:n_candidates]

    # Shuffle so the nearest isn't always obj_0
    random.shuffle(candidates)
    obj_names = [n for n, _ in candidates]
    feats_map = {n: f for n, f in candidates}

    nearest_name = min(feats_map, key=lambda n: feats_map[n].distance)
    answer_idx = obj_names.index(nearest_name)
    nearest_feat = feats_map[nearest_name]

    choice_str = "、".join(f"<obj_{i}>" for i in range(len(obj_names)))

    return QASample(
        question=f"在 {choice_str} 中，哪个物体距你最近？",
        answer=f"<obj_{answer_idx}>（距离 {nearest_feat.distance:.1f}m）。",
        qa_type="nearest",
        object_names=obj_names,
        raw_features=feats_map,
    )


def gen_count_front(features: dict[str, SpatialFeature]) -> Optional[QASample]:
    """Q: 你正前方（z > 0）有几个物体？"""
    if not features:
        return None

    front = [n for n, f in features.items() if f.z > 0]
    count = len(front)

    return QASample(
        question="你的正前方共有几个物体？",
        answer=f"{count} 个。",
        qa_type="count",
        object_names=[],
        raw_features={},
    )


# ------------------------------------------------------------------
# Main: generate a batch of QA samples from one scene
# ------------------------------------------------------------------

QA_GENERATORS = [gen_direction, gen_distance_compare, gen_nearest, gen_count_front]


def generate_qa_samples(
    scene_json: dict,
    camera_name: str = "Camera.001",
    n_direction: int = 5,
    n_compare: int = 5,
    n_nearest: int = 2,
    n_count: int = 1,
    seed: int = 42,
) -> list[QASample]:
    """
    Generate QA samples for one scene.
    Returns a flat list of QASample objects ready for build_sample().
    """
    random.seed(seed)
    np.random.seed(seed)

    features = extract_spatial_features(scene_json, camera_name=camera_name)

    samples = []

    for _ in range(n_direction):
        s = gen_direction(features, semantic_only=True)
        if s:
            samples.append(s)

    for _ in range(n_compare):
        s = gen_distance_compare(features)
        if s:
            samples.append(s)

    for i in range(n_nearest):
        s = gen_nearest(features, offset=i)
        if s:
            samples.append(s)

    for _ in range(n_count):
        s = gen_count_front(features)
        if s:
            samples.append(s)

    return samples


# ------------------------------------------------------------------
# CLI: preview generated QA
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "sample/ffc5bced-d6c3-4475-a289-0da7ec342be7-Brick_house_Glass_wall_info.json"

    with open(path) as f:
        scene = json.load(f)

    samples = generate_qa_samples(scene)

    for i, s in enumerate(samples):
        obj_map = {f"<obj_{j}>": n for j, n in enumerate(s.object_names)}
        q = s.question
        a = s.answer
        for placeholder, name in obj_map.items():
            q = q.replace(placeholder, f"[{name}]")
            a = a.replace(placeholder, f"[{name}]")
        print(f"[{s.qa_type}] Q: {q}")
        print(f"          A: {a}")
        print()
