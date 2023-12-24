import shutil
from pathlib import Path
import os
from datasets import DatasetDict, DatasetInfo
from datasets import load_dataset
from fire import Fire
from numpy import dot
from utils import DatasetMover
import dotenv

dotenv.load_dotenv()


def split(
    dataset_name="VMware/open-instruct",
    train_size=0.9,
    test_size=0.05,
    val_size=0.05,
    output_dir="dataset",
):
    print(f"Loading {dataset_name}")
    dataset = load_dataset(dataset_name)
    for row in dataset["train"]:
        row["prompt"] = row["alpaca_prompt"]
        row["completion"] = row["response"]
        for to_delete in [
            "alpaca_prompt",
            "response",
            "instruction",
            "task_name",
            "template_type",
        ]:
            row.pop(to_delete)
    short_name = dataset_name.split("/")[-1]

    # Split the dataset
    print("Splitting the dataset")
    if "test" not in dataset:
        train_testvalid = dataset["train"].train_test_split(train_size=train_size)
        test_valid = train_testvalid["test"].train_test_split(
            test_size=test_size / (test_size + val_size)
        )
        train_split = train_testvalid["train"]
        val_split = test_valid["train"]
        test_split = test_valid["test"]
    else:
        test_valid = dataset["test"].train_test_split(
            test_size=test_size / (test_size + val_size)
        )
        train_split = dataset["train"]
        val_split = test_valid["train"]
        test_split = test_valid["test"]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Converting {dataset_name} ad splitting it into train/val/test")
    # Save the splits
    splits = {}
    splits["train"] = train_split
    splits["val"] = val_split
    splits["test"] = test_split

    combined = DatasetDict(splits)
    dataset_info = DatasetInfo(
        description="Contains data from VMware/open-instruct but not formatted into llmos format",
        version="0.0.5",
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
    bucket_name = "fine-tuning-research"
    print(f"Uploading {output_tar_file} to {bucket_name}")
    dataset_mover = DatasetMover()
    dataset_mover.upload(folder_to_compress, output_tar_file, bucket_name)


if __name__ == "__main__":
    Fire(split)
