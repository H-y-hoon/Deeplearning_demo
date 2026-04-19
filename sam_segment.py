import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm
from transformers import Sam3Model, Sam3Processor


DEFAULT_IMAGE_PATH = "/home/yanghoon/workspace/project/deeplearning/yh/img/test1.jpg"
DEFAULT_OUTPUT_DIR = "/home/yanghoon/workspace/project/deeplearning/yh/sam3_outputs"
DEFAULT_MODEL_ID = "facebook/sam3"


def parse_args():
    parser = argparse.ArgumentParser(description="Open-vocabulary segmentation with Meta SAM3.")
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH, help="Input image path.")
    parser.add_argument("--prompt", required=True, help="Text prompt to segment, e.g. 'Jimin'.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for overlay, masks, and JSON.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Instance score threshold.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask binarization threshold.")
    parser.add_argument("--max-masks", type=int, default=5, help="Maximum number of masks to save.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device_arg


def load_model(model_id: str, device: str):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device}", flush=True)
    print(f"Dtype: {dtype}", flush=True)
    print(f"Loading model: {model_id}", flush=True)

    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    return processor, model, dtype


def run_segmentation(model, processor, image, prompt, threshold, mask_threshold):
    device = next(model.parameters()).device

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    return processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=[image.size[::-1]],
    )[0]


def tensor_to_list(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    return value


def get_color(index: int):
    colors = [
        (230, 0, 30, 120),
        (0, 170, 80, 120),
        (40, 120, 255, 120),
        (245, 180, 0, 120),
        (170, 70, 230, 120),
    ]
    return colors[index % len(colors)]


def make_dimmed_base(image, opacity=0.45):
    base = image.convert("RGBA")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.blend(base, white, opacity)


def load_annotation_font(image_size):
    font_size = max(18, min(image_size) // 45)
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_annotation(draw, box, label, color, image_size):
    x1, y1, x2, y2 = [float(v) for v in box]
    outline = color[:3] + (255,)
    text_color = (20, 20, 20, 255)
    label_bg = (255, 255, 255, 230)
    font = load_annotation_font(image_size)
    padding = max(4, min(image_size) // 180)

    draw.rectangle([x1, y1, x2, y2], outline=outline, width=3)

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    label_x = x1
    label_y = max(0, y1 - text_h - padding * 2)
    draw.rectangle(
        [label_x, label_y, label_x + text_w + padding * 2, label_y + text_h + padding * 2],
        fill=label_bg,
        outline=outline,
        width=2,
    )
    draw.text((label_x + padding, label_y + padding), label, fill=text_color, font=font)


def apply_mask(image, mask_array, color):
    color_layer = Image.new("RGBA", image.size, color)
    alpha_mask = Image.fromarray((mask_array.astype(np.uint8) * color[3]), mode="L")
    return Image.composite(color_layer, image, alpha_mask)


def save_outputs(image, result, output_dir, prompt):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = result.get("scores", torch.empty(0)).detach().cpu()
    boxes = result.get("boxes", torch.empty((0, 4))).detach().cpu()
    masks = result.get("masks", torch.empty((0, image.height, image.width))).detach().cpu()

    max_masks = int(result.get("max_masks", len(scores)))
    if len(scores) > 0:
        order = torch.argsort(scores, descending=True)[:max_masks]
        scores = scores[order]
        boxes = boxes[order]
        masks = masks[order]

    overlay = make_dimmed_base(image)
    instances = []

    for idx, (score, box, mask) in enumerate(zip(scores, boxes, masks)):
        mask_array = mask.numpy().astype(bool)
        mask_path = output_dir / f"mask_{idx:02d}.png"
        Image.fromarray((mask_array.astype(np.uint8) * 255), mode="L").save(mask_path)

        color = get_color(idx)
        label = f"{prompt} {float(score):.2f}"
        overlay = apply_mask(overlay, mask_array, color)

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        draw = ImageDraw.Draw(overlay)
        draw_annotation(draw, [x1, y1, x2, y2], label, color, image.size)

        instance_overlay = make_dimmed_base(image)
        instance_overlay = apply_mask(instance_overlay, mask_array, color)
        instance_draw = ImageDraw.Draw(instance_overlay)
        draw_annotation(instance_draw, [x1, y1, x2, y2], label, color, image.size)
        instance_overlay_path = output_dir / f"overlay_{idx:02d}.png"
        instance_overlay.convert("RGB").save(instance_overlay_path)

        instances.append(
            {
                "index": idx,
                "score": float(score),
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "label": label,
                "mask_path": str(mask_path),
                "overlay_path": str(instance_overlay_path),
            }
        )

    overlay_path = output_dir / "overlay.png"
    overlay.convert("RGB").save(overlay_path)

    payload = {
        "model_id": result.get("model_id"),
        "image_path": result.get("image_path"),
        "prompt": prompt,
        "image_size": [image.width, image.height],
        "num_masks": len(instances),
        "threshold": result.get("threshold"),
        "mask_threshold": result.get("mask_threshold"),
        "instances": instances,
        "overlay_path": str(overlay_path),
    }

    result_path = output_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if not instances:
        print("No masks found", flush=True)

    return overlay_path, result_path


def main():
    args = parse_args()
    device = resolve_device(args.device)

    with tqdm(total=6, desc="Load image", unit="step") as progress:
        image = Image.open(args.image).convert("RGB")
        print(f"Image: {args.image} ({image.width}x{image.height})", flush=True)
        progress.update()

        progress.set_description("Load model")
        processor, model, dtype = load_model(args.model_id, device)
        progress.update()

        progress.set_description("Preprocess")
        progress.update()

        progress.set_description("Inference")
        result = run_segmentation(model, processor, image, args.prompt, args.threshold, args.mask_threshold)
        progress.update()

        progress.set_description("Postprocess")
        result["model_id"] = args.model_id
        result["image_path"] = args.image
        result["threshold"] = args.threshold
        result["mask_threshold"] = args.mask_threshold
        result["max_masks"] = args.max_masks
        print(f"Prompt: {args.prompt}", flush=True)
        print(f"Masks found: {len(result.get('scores', []))}", flush=True)
        print(f"Runtime dtype: {dtype}", flush=True)
        progress.update()

        progress.set_description("Save")
        overlay_path, result_path = save_outputs(image, result, args.output_dir, args.prompt)
        progress.update()

    print("=== RESULT FILES ===", flush=True)
    print(f"Overlay image: {overlay_path}", flush=True)
    print(f"Result json: {result_path}", flush=True)


if __name__ == "__main__":
    main()
