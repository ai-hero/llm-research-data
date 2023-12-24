import csv
import shutil
import json
from pathlib import Path
import glob
import os
import traceback
import dotenv
from datasets import Dataset, DatasetDict, DatasetInfo
from jinja2 import Environment, FileSystemLoader
from utils import DatasetMover


class MMLUDataTransformer:
    def transform(self, row):
        prompt = f"{row[0]}\n"
        for option, choice in zip(["A", "B", "C", "D"], row[1:5]):
            prompt += f"{option}. {choice}\n"
        completion = row[5]
        return {
            "prompt": prompt,
            "completion": completion,
        }


def gen(split="dev"):
    mmlu_data = Path.home() / "data" / "mmlu"
    assert mmlu_data.exists()

    mmlu_data_transformer = MMLUDataTransformer()

    # Use glob to find all CSV files in the 'dev' folder
    count = 0
    for filepath in glob.glob((mmlu_data / split / "*.csv").as_posix()):
        print(f"Processing {filepath}")
        with open(filepath, "r") as fr:
            reader = csv.reader(fr)
            for row in reader:
                row = mmlu_data_transformer.transform(row)
                yield row

    print("split", split)
    print("count", count)


if __name__ == "__main__":
    dataset_name = "mmlu"
    dotenv.load_dotenv()
    mmlu_data = Path.home() / "data" / "mmlu"
    assert mmlu_data.exists()

    splits = {}
    for split in ["auxiliary_train", "dev", "val", "test"]:
        ds = Dataset.from_generator(gen, gen_kwargs={"split": split})
        if split == "auxiliary_train":
            splits["train"] = ds
        else:
            splits[split] = ds
    combined = DatasetDict(splits)
    dataset_info = DatasetInfo(
        description="Contains MMLU dataset string appended from cais/mmlu",
        version="0.1.0",
    )
    for split, dataset in combined.items():
        dataset.dataset_info = dataset_info

    current_directory = Path(".")
    shutil.rmtree(current_directory / "dataset", ignore_errors=True)
    os.mkdir(current_directory / "dataset")
    dataset_path = (current_directory / "dataset" / dataset_name).as_posix()
    combined.save_to_disk(dataset_path)

    # Compress the folder
    print(f"Compressing the folder {current_directory / 'dataset' / dataset_name}")
    folder_to_compress = (current_directory / "dataset" / dataset_name).as_posix()
    output_tar_file = f"{dataset_name}.tar.gz"
    bucket_name = "fine-tuning-research"
    print(f"Uploading {output_tar_file} to {bucket_name}")
    dataset_mover = DatasetMover()
    dataset_mover.upload(folder_to_compress, output_tar_file, bucket_name)
