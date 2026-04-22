# Blender Adapter

Spatial reasoning benchmark and VLM training code based on Blender 3D scene data.

## Setup

```bash
pip install numpy                                    # local (no GPU)
pip install torch transformers peft accelerate       # training (GPU required)
```

## Local Usage

```bash
# Verify coordinate transform (no GPU needed)
python3 verify_coord.py

# Preview auto-generated QA pairs
python3 -m spatial_encoder.qa_generator
```

## Training on Colab

Open `train.ipynb` in Google Colab (GPU runtime).

1. Set your GitHub repo URL in the first cell
2. Upload `sample/` to Google Drive and update the Drive path in the notebook
3. Run all cells in order

## Project Structure

```
spatial_encoder/
├── coord_transform.py   # Blender JSON → 6D camera-space features
├── qa_generator.py      # Auto-generate spatial QA pairs
├── position_encoder.py  # MLP: 6D → VLM embedding space
└── token_injection.py   # SpatialQwen2VL model + build_sample()
train.ipynb              # Colab training notebook (Step 3–8)
verify_coord.py          # Sanity-check coordinate transform locally
```
