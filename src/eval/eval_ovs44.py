"""Two-GPU OVS44Reason evaluation for Think2Seg-RS-3B and SAM2.1."""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


SYSTEM_PROMPT = (
    "You are a remote sensing analysis assistant. Your task is to generate spatial "
    "prompts for the Segment Anything Model (SAM) based on a user's request."
)
USER_PROMPT = """Please find "{question}", identify the target. For each target instance, provide:
1. `bbox_2d`: A tight bounding box.
2. `positive_points`: Exactly two points, placed inside the target.
Output your thinking process in <think> </think> tags.
Output the final answer in <answer> </answer> tags with the specified JSON format. If no targets are found, output an empty list.
i.e. <think> thinking process here </think>
<answer>```json[{{"bbox_2d": [310,360,567,586], "positive_points": [[434,474], [450,460]]}}]```</answer>"""


def select_unique_masks(records, maximum):
    if not maximum or maximum >= len(records):
        return records
    selected, seen = [], set()
    for record in records:
        if record["mask_path"] in seen:
            continue
        selected.append(record)
        seen.add(record["mask_path"])
        if len(selected) == maximum:
            return selected
    return records[:maximum]


def parse_instances(text, image_size):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not match:
        return None, "missing <answer> block"
    body = match.group(1).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", body, re.DOTALL)
    payload = fenced.group(1) if fenced else body
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as original_error:
        # The released checkpoint sometimes appends a few unmatched brackets to
        # an otherwise valid JSON list. Accept the longest valid JSON prefix.
        data = None
        for end in range(len(payload) - 1, 0, -1):
            try:
                candidate = json.loads(payload[:end].rstrip())
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, list):
                data = candidate
                break
        if data is None:
            try:
                from json_repair import repair_json
                data = json.loads(repair_json(payload))
            except Exception:
                return None, f"invalid JSON: {original_error}"
    if not isinstance(data, list):
        return None, "answer JSON is not a list"
    cleaned = []
    for instance in data:
        if not isinstance(instance, dict):
            continue
        bbox = instance.get("bbox_2d")
        points = instance.get("positive_points")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not isinstance(points, list) or len(points) != 2):
            continue
        try:
            bbox = [float(value) for value in bbox]
            points = [[float(value) for value in point] for point in points]
        except (TypeError, ValueError):
            continue
        if any(len(point) != 2 for point in points):
            continue
        bbox[0] = min(max(bbox[0], 0), image_size - 1)
        bbox[1] = min(max(bbox[1], 0), image_size - 1)
        bbox[2] = min(max(bbox[2], 0), image_size)
        bbox[3] = min(max(bbox[3], 0), image_size)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        points = [[min(max(x, 0), image_size - 1),
                   min(max(y, 0), image_size - 1)] for x, y in points]
        cleaned.append({"bbox_2d": bbox, "positive_points": points})
    if data and not cleaned:
        return None, "JSON contains no valid SAM2 prompts"
    return cleaned, None


def predict_mask(predictor, image, instances):
    combined = np.zeros((image.height, image.width), dtype=bool)
    predictor.set_image(np.array(image, copy=True))
    for instance in instances:
        masks, _, _ = predictor.predict(
            point_coords=np.asarray(instance["positive_points"], dtype=np.float32),
            point_labels=np.ones(2, dtype=np.int32),
            box=np.asarray(instance["bbox_2d"], dtype=np.float32),
            multimask_output=False,
        )
        if len(masks):
            combined |= masks[0].astype(bool)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sam-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--resize-size", type=int, default=840)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    device = torch.device(f"cuda:{local_rank}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = select_unique_masks(records, args.max_samples)
    if not records:
        raise ValueError(f"Manifest is empty: {args.manifest}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam = build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(args.sam_root / "checkpoints" / "sam2.1_hiera_small.pt"),
        device=str(device), eval_mode=True,
    )
    predictor = SAM2ImagePredictor(sam)

    local_results = []
    for index in range(rank, len(records), world_size):
        record = records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        image = image.resize((args.resize_size, args.resize_size), Image.Resampling.LANCZOS)
        gt = Image.open(record["mask_path"]).convert("L")
        gt = np.asarray(gt.resize((args.resize_size, args.resize_size),
                                  Image.Resampling.NEAREST)) > 0
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT.format(question=record["query"])},
            ]},
        ]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, use_cache=True,
            )
        completion = generated[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(
            completion, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        instances, format_error = parse_instances(output_text, args.resize_size)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = (np.zeros_like(gt) if instances is None
                    else predict_mask(predictor, image, instances))
        intersection = int(np.logical_and(pred, gt).sum())
        union = int(np.logical_or(pred, gt).sum())
        iou = intersection / max(1, union)
        mask_name = record["id"].replace("/", "__") + ".png"
        mask_path = args.output_dir / "masks" / mask_name
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred.astype(np.uint8) * 255).save(mask_path)
        local_results.append({
            "index": index,
            "id": record["id"],
            "category": record["category"],
            "image_path": record["image_path"],
            "mask_path": record["mask_path"],
            "prediction_path": str(mask_path),
            "query": record["query"],
            "output_text": output_text,
            "instances": instances,
            "format_error": format_error,
            "intersection": intersection,
            "union": union,
            "iou": iou,
        })
        print(f"rank={rank} id={record['id']} valid={format_error is None} iou={iou:.4f}")

    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    for result in local_results:
        result["rank"] = rank
        result["peak_memory_mib"] = peak_memory_mib

    gathered = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, local_results)
    else:
        gathered = [local_results]
    if rank == 0:
        results = sorted(
            (item for rank_items in gathered for item in rank_items),
            key=lambda item: item["index"],
        )
        total_intersection = sum(item["intersection"] for item in results)
        total_union = sum(item["union"] for item in results)
        summary = {
            "model": args.model,
            "sam_root": str(args.sam_root),
            "world_size": world_size,
            "samples": len(results),
            "gpu_peak_memory_mib": {
                str(item["rank"]): item["peak_memory_mib"] for item in results
            },
            "valid_formats": sum(item["format_error"] is None for item in results),
            "mIoU": sum(item["iou"] for item in results) / len(results),
            "cIoU": total_intersection / max(1, total_union),
            "results": results,
        }
        result_path = args.output_dir / "metrics.json"
        result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key != "results"},
                         ensure_ascii=False, indent=2))
        print(f"Saved: {result_path}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
