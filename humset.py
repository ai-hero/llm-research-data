from collections import OrderedDict
import shutil
from pathlib import Path
import os
from datasets import Dataset, DatasetDict, DatasetInfo
from datasets import load_dataset
from fire import Fire
from numpy import dot, short
from utils import DatasetMover
import dotenv
from tqdm import tqdm
import json

dotenv.load_dotenv()


def split(
    dataset_name="nlp-thedeep/humset",
    output_dir="dataset",
    short_name="humset",
    bucket_name="fine-tuning-research",
):
    print(f"Loading {dataset_name}")
    splits = {}
    all_sectors = set()
    all_pillars = set()
    for split in ["test"]:  # "train", "validation",
        print(f"Building for {split}")
        new_rows = []
        dataset_split = load_dataset(dataset_name, split=split)
        for row in tqdm(dataset_split):
            text = row["excerpt"].strip()
            all_sectors.update(row["sectors"])
            pillars = row["pillars_1d"]
            pillars.extend(row["pillars_2d"])
            all_pillars.update(pillars)
            new_rows.append(
                {"text": text, "sectors": row["sectors"], "pillars": pillars}
            )
        if split == "validation":
            split = "val"

        splits[split] = new_rows

    print("Building llm dataset")
    final_splits = {}
    for split in splits.keys():
        rows = splits[split]
        final_rows = []
        for row in tqdm(rows):
            sectors = OrderedDict({k: k in row["sectors"] for k in all_sectors})
            pillars = OrderedDict({k: k in row["pillars"] for k in all_pillars})
            prompt = f"""Classify the sectors and pillars for this excerpt:
{row['text']}

Classes:"""
            completion = f"""
{json.dumps({"sectors": sectors, "pillars": pillars}, indent=2)}"""

            final_rows.append({"prompt": prompt, "completion": completion})

        final_splits[split] = Dataset.from_list(final_rows)

    print(f"Converting {dataset_name} and splitting it into train/val/test")
    for split in final_splits:
        print(f"Example in split {split}:")
        for row in final_splits[split]:
            print(row)
            break

    combined = DatasetDict(final_splits)
    dataset_info = DatasetInfo(
        description=f"Contains data from {dataset_name} into completions format",
        version="0.0.1",
    )
    for split, dataset in combined.items():
        dataset.dataset_info = dataset_info

    current_directory = Path(".")
    shutil.rmtree(current_directory / output_dir, ignore_errors=True)
    os.mkdir(current_directory / output_dir)
    dataset_path = (current_directory / output_dir / short_name).as_posix()
    combined.save_to_disk(dataset_path)

    # Compress the folder
    print(f"Compressing the folder {current_directory / output_dir/ short_name}")
    folder_to_compress = (current_directory / output_dir / short_name).as_posix()
    output_tar_file = f"{short_name}.tar.gz"
    print(f"Uploading {output_tar_file} to {bucket_name}")
    dataset_mover = DatasetMover()
    dataset_mover.upload(folder_to_compress, output_tar_file, bucket_name)


if __name__ == "__main__":
    Fire(split)
