import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm
from transformers import (
    BertTokenizerFast,
    GroundingDinoConfig,
    GroundingDinoForObjectDetection,
    GroundingDinoImageProcessor,
    GroundingDinoProcessor,
)


DEFAULT_IMAGE_PATH = "/home/yanghoon/workspace/project/deeplearning/yh/img/test1.jpg"
DEFAULT_OUTPUT_DIR = "/home/yanghoon/workspace/project/deeplearning/yh/grounding_dino_outputs"
DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"


def parse_args():
    parser = argparse.ArgumentParser(description="Open-vocabulary detection with Grounding DINO.")
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH, help="Input image path.")
    parser.add_argument("--text-queries", nargs="+", required=True, help="Objects to detect, e.g. pen laptop.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for overlay and JSON.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id.")
    parser.add_argument("--box-threshold", type=float, default=0.25, help="Box score threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text matching threshold.")
    parser.add_argument("--max-detections", type=int, default=20, help="Maximum number of boxes to save.")
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"], help="Inference device.")
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device_arg


def load_config(model_id: str):
    config_path = hf_hub_download(model_id, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    if isinstance(config_dict.get("two_stage"), int) and not isinstance(config_dict.get("two_stage"), bool):
        config_dict["two_stage"] = bool(config_dict["two_stage"])

    return GroundingDinoConfig(**config_dict)


def load_model(model_id: str, device: str):
    dtype = torch.float32
    print(f"Device: {device}", flush=True)
    print(f"Dtype: {dtype}", flush=True)
    print(f"Loading model: {model_id}", flush=True)

    config = load_config(model_id)
    image_processor = GroundingDinoImageProcessor.from_pretrained(model_id)
    tokenizer = BertTokenizerFast.from_pretrained(model_id)
    processor = GroundingDinoProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = GroundingDinoForObjectDetection.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return processor, model, dtype


def run_detection(model, processor, image, text_queries, box_threshold, text_threshold):
    device = next(model.parameters()).device
    inputs = processor(images=image, text=[text_queries], return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    return processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.get("input_ids"),
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
        text_labels=[text_queries],
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


def draw_detection(draw, box, label, score, image_size):
    x1, y1, x2, y2 = [float(v) for v in box]
    outline = (230, 0, 30)
    text_color = (20, 20, 20)
    label_bg = (255, 255, 255)
    font = load_annotation_font(image_size)
    padding = max(4, min(image_size) // 180)
    text = f"{label} {float(score):.2f}"

    draw.rectangle([x1, y1, x2, y2], outline=outline, width=3)

    text_bbox = draw.textbbox((0, 0), text, font=font)
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
    draw.text((label_x + padding, label_y + padding), text, fill=text_color, font=font)


def save_outputs(image, result, output_dir, text_queries):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = result.get("scores", torch.empty(0)).detach().cpu()
    boxes = result.get("boxes", torch.empty((0, 4))).detach().cpu()
    labels = result.get("text_labels") or result.get("labels") or []

    max_detections = int(result.get("max_detections", len(scores)))
    if len(scores) > 0:
        order = torch.argsort(scores, descending=True)[:max_detections]
        scores = scores[order]
        boxes = boxes[order]
        labels = [labels[int(i)] for i in order.tolist()]

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    detections = []

    for idx, (score, box, label) in enumerate(zip(scores, boxes, labels)):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        draw_detection(draw, [x1, y1, x2, y2], label, score, image.size)
        detections.append(
            {
                "index": idx,
                "label": str(label),
                "score": float(score),
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )

    overlay_path = output_dir / "overlay.png"
    overlay.save(overlay_path)

    payload = {
        "model_id": result.get("model_id"),
        "image_path": result.get("image_path"),
        "text_queries": text_queries,
        "image_size": [image.width, image.height],
        "box_threshold": result.get("box_threshold"),
        "text_threshold": result.get("text_threshold"),
        "num_detections": len(detections),
        "detections": detections,
        "overlay_path": str(overlay_path),
    }

    result_path = output_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if not detections:
        print("No detections found", flush=True)

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
        result = run_detection(model, processor, image, args.text_queries, args.box_threshold, args.text_threshold)
        progress.update()

        progress.set_description("Postprocess")
        result["model_id"] = args.model_id
        result["image_path"] = args.image
        result["box_threshold"] = args.box_threshold
        result["text_threshold"] = args.text_threshold
        result["max_detections"] = args.max_detections
        print(f"Text queries: {args.text_queries}", flush=True)
        print(f"Detections found: {len(result.get('scores', []))}", flush=True)
        print(f"Runtime dtype: {dtype}", flush=True)
        progress.update()

        progress.set_description("Save")
        overlay_path, result_path = save_outputs(image, result, args.output_dir, args.text_queries)
        progress.update()

    print("=== RESULT FILES ===", flush=True)
    print(f"Overlay image: {overlay_path}", flush=True)
    print(f"Result json: {result_path}", flush=True)


if __name__ == "__main__":
    main()
