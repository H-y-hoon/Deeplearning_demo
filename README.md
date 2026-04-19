# Deeplearning Demo

A collection of demo scripts for object detection, visual question answering, and instance segmentation using Vision-Language Models (VLMs).  
All three models share a consistent interface, making it easy to run and compare experiments.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Scripts & Usage](#4-scripts--usage)
   - [gdino\_detect.py — Grounding DINO Object Detection](#41-gdino_detectpy--grounding-dino-object-detection)
   - [qwen\_vlchat.py — Qwen VL Visual Grounding](#42-qwen_vlchatpy--qwen-vl-visual-grounding)
   - [sam\_segment.py — SAM3 Instance Segmentation](#43-sam_segmentpy--sam3-instance-segmentation)
5. [Output Format](#5-output-format)

---

## 1. Overview

| Script | Model | Task | Output |
|---|---|---|---|
| `gdino_detect.py` | `IDEA-Research/grounding-dino-base` | Text-query-based object detection | BBox + `overlay.png` |
| `qwen_vlchat.py` | `Qwen/Qwen3.5-9B` | Natural language visual reasoning + grounding | BBox · Point + `annotated.png` |
| `sam_segment.py` | `facebook/sam3` | Text-prompt-based instance segmentation | Pixel mask + `overlay.png` |

---

## 2. Repository Structure

```
Deeplearning_demo/
├── gdino_detect.py              # Grounding DINO object detection
├── qwen_vlchat.py               # Qwen VL visual question answering
├── sam_segment.py               # SAM3 instance segmentation
├── requirements.txt
├── img/
│   └── test1.jpg                # Sample input image (1920×1080)
├── gdino_outputs/               # Output directory for gdino_detect.py
├── grounding_dino_outputs/      # Alternative output directory for gdino_detect.py
├── qwen_outputs/                # Output directory for qwen_vlchat.py
└── sam3_outputs/                # Output directory for sam_segment.py
```

> **Note:** `*_outputs/` directories contain only placeholder `.gitkeep` files.  
> Actual result files are excluded via `.gitignore`.

---

## 3. Environment Setup

### Prerequisites

- Python 3.10+
- CUDA 12.x or later (recommended)

### Install Dependencies

```bash
# 1. Create a conda virtual environment (recommended)
conda create -n deeplearning_demo python=3.10 -y
conda activate deeplearning_demo

# 2. Install PyTorch with CUDA 12.x support
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies
pip install -r requirements.txt
```

---

## 4. Scripts & Usage

### 4.1 `gdino_detect.py` — Grounding DINO Object Detection

Detects objects matching a text query and draws bounding boxes on the image.

```bash
# Basic usage
python gdino_detect.py --text-queries laptop

# Detect multiple objects at once
python gdino_detect.py --text-queries laptop keyboard mouse

# Full options
python gdino_detect.py \
    --image img/test1.jpg \
    --text-queries laptop keyboard \
    --output-dir grounding_dino_outputs \
    --model-id IDEA-Research/grounding-dino-base \
    --box-threshold 0.25 \
    --text-threshold 0.25 \
    --max-detections 20 \
    --device cuda
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--image` | `img/test1.jpg` | Path to input image |
| `--text-queries` | (required) | Objects to detect (space-separated) |
| `--output-dir` | `grounding_dino_outputs` | Output directory |
| `--box-threshold` | `0.25` | Bounding box confidence threshold |
| `--text-threshold` | `0.25` | Text matching threshold |
| `--max-detections` | `20` | Maximum number of detections to save |
| `--device` | `cuda` | Inference device (`cuda` / `cpu` / `auto`) |

---

### 4.2 `qwen_vlchat.py` — Qwen VL Visual Grounding

Answers a natural language question about an image and returns the BBox and Point coordinates of the relevant subject.  
The image is resized to 1000×1000 for inference and coordinates are scaled back to the original resolution.

```bash
# Basic usage (uses DEFAULT_QUESTION defined in the script)
python qwen_vlchat.py

# Custom question
python qwen_vlchat.py \
    --image img/test1.jpg \
    --question "Find the person closest to the door." \
    --output-dir qwen_outputs \
    --max-new-tokens 512 \
    --device cuda
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--image` | `img/xai506_example_image.jpg` | Path to input image |
| `--question` | (built-in default) | Natural language question about the image |
| `--output-dir` | `qwen_outputs` | Output directory |
| `--model-id` | `Qwen/Qwen3.5-9B` | HuggingFace model ID |
| `--max-new-tokens` | `512` | Maximum number of tokens to generate |
| `--device` | `cuda` | Inference device |

---

### 4.3 `sam_segment.py` — SAM3 Instance Segmentation

Extracts pixel-level masks for objects specified by a text prompt.  
Each instance is saved as an individual mask image (`mask_XX.png`) and overlay (`overlay_XX.png`).

```bash
# Basic usage
python sam_segment.py --prompt "person"

# Full options
python sam_segment.py \
    --image img/test1.jpg \
    --prompt "laptop" \
    --output-dir sam3_outputs \
    --model-id facebook/sam3 \
    --threshold 0.3 \
    --mask-threshold 0.5 \
    --max-masks 5 \
    --device auto
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--image` | `img/test1.jpg` | Path to input image |
| `--prompt` | (required) | Text prompt for the target object |
| `--output-dir` | `sam3_outputs` | Output directory |
| `--threshold` | `0.3` | Instance confidence threshold |
| `--mask-threshold` | `0.5` | Mask binarization threshold |
| `--max-masks` | `5` | Maximum number of masks to save |
| `--device` | `auto` | Inference device |

---

## 5. Output Format

All scripts save a `result.json` and a visualization image to the output directory.

### `gdino_detect.py` / `sam_segment.py` — `result.json`

```json
{
  "model_id": "IDEA-Research/grounding-dino-base",
  "image_path": "img/test1.jpg",
  "text_queries": ["laptop"],
  "image_size": [1920, 1080],
  "num_detections": 1,
  "detections": [
    {
      "index": 0,
      "label": "laptop",
      "score": 0.512,
      "box": [100.0, 200.0, 400.0, 350.0]
    }
  ],
  "overlay_path": "grounding_dino_outputs/overlay.png"
}
```

### `qwen_vlchat.py` — `result.json`

```json
{
  "model_id": "Qwen/Qwen3.5-9B",
  "question": "Find the person closest to the door.",
  "original_image_size": [4032, 3024],
  "inference_image_size": [1000, 1000],
  "parsed": {
    "answer": "The man in the black hoodie on the far left.",
    "found": true,
    "bbox": [0, 452, 114, 567],
    "point": [56, 508]
  },
  "bbox_on_original_image": [0, 1367, 460, 1715],
  "point_on_original_image": [226, 1536],
  "annotated_image": "qwen_outputs/annotated.png"
}
```
