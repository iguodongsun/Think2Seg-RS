"""OVS44Reason adapter for the Think2Seg-RS GRPO trainer."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

SYSTEM_PROMPT = (
    "You are a remote sensing analysis assistant. Your task is to generate spatial "
    "prompts for the Segment Anything Model (SAM) based on a user's request."
)
USER_PROMPT = """Please find "{Question}", identify the target. For each target instance, provide:
1. A tight bbox_2d bounding box.
2. Exactly two positive_points placed inside the target.
Output your thinking process in <think> </think> tags and the JSON list in
<answer> </answer> tags. If no targets are found, output an empty list."""


class OVS44ReasonDataset(Dataset):
    def __init__(self, manifest, resize_size=840):
        self.records = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if not self.records:
            raise ValueError(f"Empty OVS44Reason manifest: {manifest}")
        self.resize_size = resize_size

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _build_conversation(question):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT.format(Question=question)},
                ],
            },
        ]

    def __getitem__(self, index):
        record = self.records[index]
        size = (self.resize_size, self.resize_size)
        image = Image.open(record["image_path"]).convert("RGB").resize(
            size, Image.Resampling.LANCZOS
        )
        mask = Image.open(record["mask_path"]).convert("L").resize(
            size, Image.Resampling.NEAREST
        )
        return {
            "data_idx": record["id"],
            "image_path": record["image_path"],
            "image": image,
            "GT_mask_path": record["mask_path"],
            "GT_mask": np.asarray(mask) > 0,
            "problem": record["query"],
            "prompt": self._build_conversation(record["query"]),
        }
