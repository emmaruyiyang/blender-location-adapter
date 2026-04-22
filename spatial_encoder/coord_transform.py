"""
Blender JSON -> 6D camera-space spatial features
"""

import json
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SpatialFeature:
    name: str
    # Camera-space cartesian
    x: float   # right
    y: float   # up
    z: float   # forward (depth)
    # Spherical
    distance: float
    azimuth: float    # radians, left/right angle
    elevation: float  # radians, up/down angle

    def to_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z,
                         self.distance, self.azimuth, self.elevation],
                        dtype=np.float32)


def euler_xyz_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Blender XYZ Euler (radians) -> 3x3 rotation matrix (world orientation of camera)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1,  0,   0],
                   [0,  cx, -sx],
                   [0,  sx,  cx]])
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]])
    return Rz @ Ry @ Rx


# Blender camera points -Z forward, Y up.
# OpenCV convention:  Z forward, -Y up.
# This matrix converts between them.
BLENDER_TO_OPENCV = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=np.float64)


def build_camera_transform(camera_obj: dict):
    """
    Returns (R_cw, t_cw) such that P_cam = R_cw @ P_world + t_cw.
    R_cw: 3x3, t_cw: (3,)
    """
    loc = camera_obj["location"]
    rot = camera_obj["rotation"]

    t_world = np.array([loc["x"], loc["y"], loc["z"]])
    R_world = euler_xyz_to_rotation_matrix(rot["x"], rot["y"], rot["z"])

    # R_world is camera-in-world orientation; invert to get world->camera
    R_cw = BLENDER_TO_OPENCV @ R_world.T
    t_cw = -R_cw @ t_world
    return R_cw, t_cw


def world_to_camera(P_world: np.ndarray, R_cw: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
    return R_cw @ P_world + t_cw


def cartesian_to_spherical(p: np.ndarray):
    """p: (x, y, z) in camera space (OpenCV: x right, y down, z forward)."""
    x, y, z = p
    distance = np.linalg.norm(p)
    # azimuth: angle in XZ plane, positive = right
    azimuth = np.arctan2(x, z)
    # elevation: positive = above horizon
    elevation = np.arctan2(-y, np.sqrt(x**2 + z**2))
    return distance, azimuth, elevation


def extract_spatial_features(scene_json: dict,
                              object_names: Optional[list[str]] = None,
                              camera_name: str = "Camera.001") -> dict[str, SpatialFeature]:
    """
    Args:
        scene_json:    parsed JSON dict from Blender export
        object_names:  which objects to process; None = all visible MESH objects
        camera_name:   name of the camera object in the JSON

    Returns:
        dict mapping object name -> SpatialFeature
    """
    objects_by_name = {o["name"]: o for o in scene_json["objects"]}

    camera_obj = objects_by_name.get(camera_name)
    if camera_obj is None:
        raise ValueError(f"Camera '{camera_name}' not found in scene JSON")

    R_cw, t_cw = build_camera_transform(camera_obj)

    results = {}
    for obj in scene_json["objects"]:
        if obj["type"] != "MESH":
            continue
        if not obj.get("visible", True) or obj.get("hide_render", False):
            continue
        if object_names is not None and obj["name"] not in object_names:
            continue

        loc = obj["location"]
        P_world = np.array([loc["x"], loc["y"], loc["z"]])
        P_cam = world_to_camera(P_world, R_cw, t_cw)

        distance, azimuth, elevation = cartesian_to_spherical(P_cam)

        results[obj["name"]] = SpatialFeature(
            name=obj["name"],
            x=float(P_cam[0]),
            y=float(P_cam[1]),
            z=float(P_cam[2]),
            distance=float(distance),
            azimuth=float(azimuth),
            elevation=float(elevation),
        )

    return results


def normalize_features(features: dict[str, SpatialFeature]) -> dict[str, np.ndarray]:
    """Z-score normalize across all objects in the scene."""
    vectors = np.stack([f.to_vector() for f in features.values()])  # (N, 6)
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0) + 1e-8
    return {
        name: (feat.to_vector() - mean) / std
        for name, feat in features.items()
    }, mean, std


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "sample/ffc5bced-d6c3-4475-a289-0da7ec342be7-Brick_house_Glass_wall_info.json"

    with open(path) as f:
        scene = json.load(f)

    features = extract_spatial_features(scene)
    print(f"Extracted {len(features)} objects\n")

    # Show a few named objects for sanity-check
    named = {k: v for k, v in features.items()
             if not k.startswith("_") and not k.startswith("Cube")}
    for name, feat in list(named.items())[:8]:
        print(f"{name:30s}  z={feat.z:6.2f}m  az={np.degrees(feat.azimuth):+6.1f}°  "
              f"el={np.degrees(feat.elevation):+5.1f}°  d={feat.distance:.2f}m")
