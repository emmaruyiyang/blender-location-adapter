"""
验证坐标转换是否正确，不需要 GPU / 模型权重。
对照截图人工核查：场景截图里哪个物体在左/右/近/远。
"""

import json
import numpy as np
from spatial_encoder.coord_transform import extract_spatial_features

SCENE_JSON = "sample/ffc5bced-d6c3-4475-a289-0da7ec342be7-Brick_house_Glass_wall_info.json"

with open(SCENE_JSON) as f:
    scene = json.load(f)

features = extract_spatial_features(scene)

# 只打印有语义名称的物体（排除 Cube.xxx 和 _3DGeom_xxx）
def is_semantic(name: str) -> bool:
    return not (name.startswith("Cube") or name.startswith("_3D") or
                name.startswith("Plane") or name.startswith("Point") or
                name.startswith("Area") or name.startswith("String") or
                name.startswith("Venetian") or name.startswith("Obj3d"))

named = {k: v for k, v in features.items() if is_semantic(k)}

print(f"{'Object':<35} {'z(fwd)':>8} {'azimuth':>10} {'elevation':>10} {'dist':>8}")
print("-" * 75)
for name, feat in sorted(named.items(), key=lambda kv: kv[1].distance):
    az_deg = np.degrees(feat.azimuth)
    el_deg = np.degrees(feat.elevation)
    direction = ("RIGHT" if az_deg > 5 else "LEFT" if az_deg < -5 else "CENTER")
    print(f"{name:<35} {feat.z:>8.2f}  {az_deg:>+8.1f}°  {el_deg:>+8.1f}°  {feat.distance:>6.2f}m  {direction}")

print()
print("=== QA 自动生成示例 ===")
sorted_by_dist = sorted(named.items(), key=lambda kv: kv[1].distance)
if len(sorted_by_dist) >= 2:
    nearest_name, nearest = sorted_by_dist[0]
    farthest_name, farthest = sorted_by_dist[-1]
    second_name, second = sorted_by_dist[1]

    print(f"\nQ: 场景中距你最近的物体是哪个？")
    print(f"A: {nearest_name}（距离 {nearest.distance:.2f}m）")

    print(f"\nQ: {nearest_name} 和 {second_name} 哪个离你更近？")
    print(f"A: {nearest_name} 更近（{nearest.distance:.2f}m vs {second.distance:.2f}m）")

    az = np.degrees(nearest.azimuth)
    if az > 15:
        direction = "右前方"
    elif az < -15:
        direction = "左前方"
    elif nearest.z > 0:
        direction = "正前方"
    else:
        direction = "正后方"
    print(f"\nQ: {nearest_name} 在你的哪个方向？")
    print(f"A: {direction}（方位角 {az:+.1f}°）")

    # 计数：正前方物体
    front_objs = [n for n, f in named.items() if f.z > 0]
    print(f"\nQ: 你正前方有几个物体？")
    print(f"A: {len(front_objs)} 个")
