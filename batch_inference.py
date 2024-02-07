import shutil
from pathlib import Path
import os
from datasets import Dataset, DatasetDict, DatasetInfo
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
    short_name="open-instruct-inference",
    output_dir="dataset",
):
    print(f"Loading {dataset_name}")
    dataset = load_dataset(dataset_name)
    new_dataset = []
    for row in dataset["train"]:
        row["prompt"] = row["alpaca_prompt"].strip()
        row["completion"] = row["response"].strip()
        for to_delete in [
            "alpaca_prompt",
            "response",
            "instruction",
            "task_name",
            "template_type",
        ]:
            row.pop(to_delete)
        new_dataset.append(row)
        if len(new_dataset) > 100:
            break

    # Create a Dataset from your list of dictionaries
    new_dataset = Dataset.from_list(new_dataset)

    # If you're creating a new dataset from scratch:
    dataset_dict = DatasetDict(
        {
            "batch_inference": new_dataset  # Assign the new dataset as the train split
        }
    )

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Converting {dataset_name} ad splitting it into train/val/test")

    dataset_info = DatasetInfo(
        description="Contains data from VMware/open-instruct but not formatted into llmos format",
        version="0.0.5",
    )
    for split, dataset in dataset_dict.items():
        dataset.dataset_info = dataset_info

    current_directory = Path(".")
    shutil.rmtree(current_directory / output_dir, ignore_errors=True)
    os.mkdir(current_directory / output_dir)
    dataset_path = (current_directory / output_dir / short_name).as_posix()
    dataset_dict.save_to_disk(dataset_path)

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
