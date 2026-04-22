# Blender Adapter — Step-by-Step Guide

## 项目结构

```
blender-adapter/
├── spatial_encoder/
│   ├── coord_transform.py   # Step 1: JSON → 6D 相机坐标特征
│   ├── qa_generator.py      # Step 2: 自动生成 QA 对
│   ├── position_encoder.py  # Step 3: MLP 位置编码器（需要 GPU）
│   └── token_injection.py   # Step 4: SpatialQwen2VL 模型（需要 GPU）
├── verify_coord.py          # 验证脚本（不需要 GPU）
└── sample/
    ├── ffc5bced-...-Brick_house_Glass_wall_info.json  # 示例场景 JSON
    └── imagefolder/
        └── ffc5bced-...-Brick_house_Glass_wall.jpg    # 对应渲染图
```

---

## 前置要求

```bash
# 无 GPU 的步骤（Step 1-2）只需要：
pip install numpy

# 有 GPU 的步骤（Step 3-4）还需要：
pip install torch transformers peft accelerate
```

---

## Step 1：验证坐标转换

**目的**：确认 Blender JSON → 相机坐标系转换正确，人工对照截图核查方位角。

**不需要 GPU，不需要模型权重。**

```bash
python3 verify_coord.py
```

**预期输出**：
```
Object                              z(fwd)    azimuth  elevation     dist
-----------------------------------------------------------------------
Cylinder.001                          5.97      +9.9°      -7.6°    6.11m  RIGHT
Coffee Table                         15.50     -18.2°      -4.1°   16.36m  LEFT
...

=== QA 自动生成示例 ===
Q: 场景中距你最近的物体是哪个？
A: Cylinder.001（距离 6.11m）
```

**如何核查**：对照 `sample/imagefolder/ffc5bced-...-Brick_house_Glass_wall.jpg`，
确认"LEFT"的物体在截图左侧，"RIGHT"的在右侧，`z` 值越小越近。

---

## Step 2：生成 QA 样本

**目的**：从一个场景 JSON 自动批量生成四类 QA 对。

```bash
python3 -m spatial_encoder.qa_generator
```

**预期输出**（每条格式为 `[类型] Q: ... A: ...`）：
```
[direction]        Q: [Coffee Table] 在你的哪个方向？
                   A: [Coffee Table] 在你的左前方（方位角 -18.2°，距离 16.4m）。

[distance_compare] Q: [IKEA_TRYSIL_BED.013] 和 [Coffee Table] 哪个离你更近？
                   A: [IKEA_TRYSIL_BED.013] 更近（近 6.3m）。

[nearest]          Q: 在 [A]、[B]、[C]、[D] 中，哪个物体距你最近？
                   A: [Cylinder.001]（距离 6.1m）。

[count]            Q: 你的正前方共有几个物体？
                   A: 89 个。
```

**在代码里调用**：
```python
import json
from spatial_encoder.qa_generator import generate_qa_samples

with open("sample/ffc5bced-...-Brick_house_Glass_wall_info.json") as f:
    scene = json.load(f)

samples = generate_qa_samples(
    scene,
    n_direction=5,
    n_compare=5,
    n_nearest=2,
    n_count=1,
)

for s in samples:
    print(s.qa_type, "|", s.question)
    print("           →", s.answer)
    print("  objects  :", s.object_names)
    print()
```

---

## Step 3：构建训练样本（build_sample）

**目的**：把一条 QASample 打包成模型可用的训练格式（归一化坐标向量）。

**不需要 GPU。**

```python
import json
from spatial_encoder.qa_generator import generate_qa_samples
from spatial_encoder.token_injection import build_sample

with open("sample/ffc5bced-...-Brick_house_Glass_wall_info.json") as f:
    scene = json.load(f)

samples = generate_qa_samples(scene)
qa = samples[0]  # 取第一条

# 只对有具名物体的 QA 调用 build_sample
if qa.object_names:
    sample = build_sample(
        scene_json=scene,
        question=qa.question,
        answer=qa.answer,
        object_names=qa.object_names,
    )
    print("prompt          :", sample["prompt"])
    print("answer          :", sample["answer"])
    print("spatial_vectors :", sample["spatial_vectors"].shape)  # (N_obj, 6)
    print("norm_mean       :", sample["norm_mean"])
```

**输出**：
```
prompt          : <obj_0> 和 <obj_1> 哪个离你更近？
answer          : <obj_0> 更近（近 2.1m）。
spatial_vectors : (2, 6)
```

---

## Step 4：加载模型并推理（需要 GPU）

**前置**：需要 HuggingFace 访问权限下载 Qwen2-VL-7B。

```bash
huggingface-cli login
```

```python
import torch
import json
import numpy as np
from PIL import Image
from spatial_encoder.token_injection import SpatialQwen2VL, build_sample
from spatial_encoder.qa_generator import generate_qa_samples

# 加载模型（首次运行会下载权重，约 15GB）
model = SpatialQwen2VL("Qwen/Qwen2-VL-7B-Instruct")
tokenizer = model.tokenizer

# 准备数据
with open("sample/ffc5bced-...-Brick_house_Glass_wall_info.json") as f:
    scene = json.load(f)

samples = generate_qa_samples(scene)
qa = next(s for s in samples if s.object_names)  # 找一条有具名物体的

sample = build_sample(scene, qa.question, qa.answer, qa.object_names)

# Tokenize
inputs = tokenizer(sample["prompt"], return_tensors="pt").to("cuda")
spatial = torch.tensor(sample["spatial_vectors"]).unsqueeze(0).to("cuda")  # (1, N, 6)

# 推理
output_ids = model.generate(
    input_ids=inputs["input_ids"],
    spatial_vectors=spatial,
    max_new_tokens=64,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## Step 5：SFT 训练（需要 GPU + peft）

```python
from peft import get_peft_model, LoraConfig, TaskType

# 在 Step 4 的模型基础上加 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
)
model.vlm = get_peft_model(model.vlm, lora_config)
model.vlm.print_trainable_parameters()
# 输出示例: trainable params: 20M || all params: 7.2B || trainable%: 0.28%

# 可训练参数：LoRA 层 + 位置编码器
optimizer = torch.optim.AdamW([
    {"params": model.vlm.parameters(), "lr": 2e-4},
    {"params": model.position_encoder.parameters(), "lr": 1e-3},
])

# 训练循环（伪代码，实际用 HuggingFace Trainer 或自定义 DataLoader）
for batch in dataloader:
    outputs = model(
        input_ids=batch["input_ids"],
        pixel_values=batch["pixel_values"],
        spatial_vectors=batch["spatial_vectors"],
        labels=batch["labels"],           # answer 部分，question 处为 -100
    )
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 当前可运行的范围

| Step | 内容 | 是否可运行 |
|------|------|-----------|
| Step 1 | 坐标转换验证 | ✅ 现在就能跑 |
| Step 2 | QA 自动生成 | ✅ 现在就能跑 |
| Step 3 | build_sample | ✅ 现在就能跑（只需 numpy）|
| Step 4 | 模型推理 | 需要 GPU + 下载权重 |
| Step 5 | SFT 训练 | 需要 GPU + peft |

**下一步建议**：在有 GPU 的机器（A100 或 H100）上先跑 Step 4 的单条推理，
确认 token 注入和 forward 正常，再接 Step 5 的训练循环。
