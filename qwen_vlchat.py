import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_IMAGE_PATH = "/home/yanghoon/workspace/project/deeplearning/yh/img/xai506_example_image.jpg"
# DEFAULT_IMAGE_PATH = "/home/yanghoon/workspace/project/deeplearning/yh/img/test1.jpg"
DEFAULT_OUTPUT_DIR = "/home/yanghoon/workspace/project/deeplearning/yh/qwen_outputs"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
# DEFAULT_QUESTION = "Who is the BTS member jimin in this image?"
DEFAULT_QUESTION = "Analyze the objects and the environmental setting in this room. Draw BBoxes to identify the 'object capable of producing the loudest sound' and the 'person with the highest proximity or potential to trigger that sound.' Explain the physical and situational rationale behind why you linked this specific person to that object."
INFERENCE_IMAGE_SIZE = (1000, 1000)


def parse_args():
    parser = argparse.ArgumentParser(description="Visual grounding with Qwen VL.")
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH, help="Input image path.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question to ask about the image.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for annotated image and JSON.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id.")
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of generated tokens.")
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

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    print("Model loaded with bf16 (CUDA) / fp32 (CPU).", flush=True)
    return processor, model, dtype


def build_prompt(question: str, inference_size: tuple[int, int]):
    inference_width, inference_height = inference_size
    return (
        "You are a visual grounding assistant. "
        f"The image you see has been resized to exactly {inference_width}x{inference_height} pixels. "
        f"Question: {question}\n"
        "Return ONLY one valid JSON object with keys: answer, found, bbox, point. "
        "Do not write analysis, reasoning, markdown, code fences, or extra text. "
        "- answer: short text answer\n"
        "- found: true/false\n"
        f"- bbox: [x1,y1,x2,y2] in absolute pixel coordinates of the {inference_width}x{inference_height} resized image or null\n"
        f"- point: [x,y] in absolute pixel coordinates of the {inference_width}x{inference_height} resized image or null\n"
        "If the target person cannot be identified with confidence, set found=false and bbox/point=null."
    )


def run_inference(model, processor, image, question, max_new_tokens, device):
    prompt = build_prompt(question, image.size)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    )

    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    prompt_len = inputs["input_ids"].shape[-1]
    completion = outputs[0][prompt_len:]
    return processor.decode(completion, skip_special_tokens=True).strip()


def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_bbox(bbox, width: int, height: int):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return None

    x1 = clamp(x1, 0, width - 1)
    x2 = clamp(x2, 0, width - 1)
    y1 = clamp(y1, 0, height - 1)
    y2 = clamp(y2, 0, height - 1)

    if x1 == x2 or y1 == y2:
        return None

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    return [left, top, right, bottom]


def normalize_point(point, width: int, height: int):
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    try:
        x, y = [int(round(float(v))) for v in point]
    except (TypeError, ValueError):
        return None

    x = clamp(x, 0, width - 1)
    y = clamp(y, 0, height - 1)
    return [x, y]


def scale_bbox(bbox, source_width: int, source_height: int, target_width: int, target_height: int):
    normalized = normalize_bbox(bbox, source_width, source_height)
    if normalized is None:
        return None

    x_scale = target_width / source_width
    y_scale = target_height / source_height
    x1, y1, x2, y2 = normalized
    return normalize_bbox(
        [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale],
        target_width,
        target_height,
    )


def scale_point(point, source_width: int, source_height: int, target_width: int, target_height: int):
    normalized = normalize_point(point, source_width, source_height)
    if normalized is None:
        return None

    x_scale = target_width / source_width
    y_scale = target_height / source_height
    x, y = normalized
    return normalize_point([x * x_scale, y * y_scale], target_width, target_height)


def parse_model_output(text: str):
    parsed = extract_json(text)
    if parsed is not None:
        return parsed
    return {
        "answer": "",
        "found": False,
        "bbox": None,
        "point": None,
        "parse_error": True,
    }


def annotate_image(image: Image.Image, result: dict, coordinate_size: tuple[int, int]):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    coord_width, coord_height = coordinate_size

    bbox = scale_bbox(result.get("bbox"), coord_width, coord_height, width, height)
    point = scale_point(result.get("point"), coord_width, coord_height, width, height)
    found = bool(result.get("found", False))
    answer = str(result.get("answer", "")).strip()

    line_width = max(2, min(width, height) // 300)

    if bbox is not None:
        draw.rectangle(bbox, outline="lime", width=line_width)

    if point is not None:
        radius = max(4, min(width, height) // 100)
        x, y = point
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline="red", width=line_width)

    status = "FOUND" if found else "NOT_FOUND"
    summary = f"{status} | answer: {answer if answer else 'N/A'}"

    label_x, label_y = 10, 10
    text_w = max(240, len(summary) * 7)
    text_h = 26
    draw.rectangle([label_x - 4, label_y - 4, label_x + text_w, label_y + text_h], fill="black")
    draw.text((label_x, label_y), summary, fill="white")

    return annotated, bbox, point


def save_outputs(image, result, output_dir, question, raw_output, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated, bbox, point = annotate_image(image, result, INFERENCE_IMAGE_SIZE)
    annotated_path = output_dir / "annotated.png"
    result_path = output_dir / "result.json"
    annotated.save(annotated_path)

    inference_width, inference_height = metadata["inference_image_size"]
    payload = {
        "model_id": metadata["model_id"],
        "image_path": metadata["image_path"],
        "question": question,
        "original_image_size": metadata["original_image_size"],
        "inference_image_size": metadata["inference_image_size"],
        "raw_output": raw_output,
        "parsed": result,
        "model_bbox_on_resized_image": normalize_bbox(result.get("bbox"), inference_width, inference_height),
        "model_point_on_resized_image": normalize_point(result.get("point"), inference_width, inference_height),
        "bbox_on_original_image": bbox,
        "point_on_original_image": point,
        "annotated_image": str(annotated_path),
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return annotated_path, result_path


def main():
    args = parse_args()
    device = resolve_device(args.device)

    with tqdm(total=7, desc="Load image", unit="step") as progress:
        image = Image.open(args.image).convert("RGB")
        print(f"Image: {args.image} ({image.width}x{image.height})", flush=True)
        progress.update()

        progress.set_description("Load model")
        processor, model, dtype = load_model(args.model_id, device)
        progress.update()

        progress.set_description("Resize image")
        inference_image = image.resize(INFERENCE_IMAGE_SIZE, Image.Resampling.LANCZOS)
        print(f"Inference image: {inference_image.width}x{inference_image.height}", flush=True)
        progress.update()

        progress.set_description("Inference")
        raw_output = run_inference(model, processor, inference_image, args.question, args.max_new_tokens, device)
        progress.update()

        progress.set_description("Postprocess")
        print("=== RAW MODEL OUTPUT ===", flush=True)
        print(raw_output, flush=True)
        parsed = parse_model_output(raw_output)
        metadata = {
            "model_id": args.model_id,
            "image_path": args.image,
            "original_image_size": [image.width, image.height],
            "inference_image_size": [inference_image.width, inference_image.height],
            "dtype": str(dtype),
        }
        print(f"Question: {args.question}", flush=True)
        print(f"Found: {bool(parsed.get('found', False))}", flush=True)
        print(f"Runtime dtype: {dtype}", flush=True)
        progress.update()

        progress.set_description("Save")
        annotated_path, result_path = save_outputs(image, parsed, args.output_dir, args.question, raw_output, metadata)
        progress.update()

        progress.set_description("Done")
        progress.update()

    print("=== RESULT FILES ===", flush=True)
    print(f"Annotated image: {annotated_path}", flush=True)
    print(f"Result json: {result_path}", flush=True)


if __name__ == "__main__":
    main()
