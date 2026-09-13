"""Build query-level OVS44Reason manifests for Think2Seg-RS."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/OVS44Reason"))
    parser.add_argument("--output", type=Path,
                        default=Path("/home/Think2Seg-RS/data/ovs44"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {}
    for split in ("train", "test"):
        base = args.root / split
        images = {path.stem: path for path in (base / "rgb").glob("*") if path.is_file()}
        masks = {path.stem: path for path in (base / "D2mask").glob("*.png")}
        texts = sorted((base / "text").glob("*.json"))
        if set(masks) != {path.stem for path in texts}:
            raise ValueError(f"D2mask/text mismatch in {base}")
        records, empty_masks = [], []
        for text_path in texts:
            matches = [stem for stem in images if text_path.stem.startswith(stem + "_")]
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous RGB for {text_path}")
            image_stem = matches[0]
            image_path, mask_path = images[image_stem], masks[text_path.stem]
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    raise ValueError(f"Image/mask size mismatch: {text_path.stem}")
                array = np.asarray(mask)
                if array.ndim == 3:
                    if not np.all(array == array[..., :1]):
                        raise ValueError(f"Mask channels differ: {mask_path}")
                    array = array[..., 0]
                if not set(np.unique(array)).issubset({0, 1, 255}):
                    raise ValueError(f"Non-binary mask: {mask_path}")
                if not (array > 0).any():
                    empty_masks.append(text_path.stem)
                width, height = image.size
            qa = json.loads(text_path.read_text(encoding="utf-8"))
            questions = qa.get("questions")
            answers = qa.get("answers", qa.get("answer"))
            if (not isinstance(questions, list) or not isinstance(answers, list)
                    or not questions or len(questions) != len(answers)):
                raise ValueError(f"Invalid paired QA: {text_path}")
            category = text_path.stem[len(image_stem) + 1:]
            for question_index, (question, answer) in enumerate(zip(questions, answers)):
                records.append({
                    "id": f"{split}/{text_path.stem}/q{question_index}",
                    "split": split,
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "query": question.strip(),
                    "answer": answer.strip(),
                    "category": category,
                    "width": width,
                    "height": height,
                })
        path = args.output / f"{split}.json"
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        audit[split] = {"images": len(images), "masks": len(masks),
                        "queries": len(records), "empty_masks": empty_masks,
                        "manifest": str(path.resolve())}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
