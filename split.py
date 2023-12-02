import shutil
from pathlib import Path
import os
from datasets import DatasetDict
from datasets import load_dataset
from fire import Fire
from numpy import dot
from utils import DatasetMover
import dotenv

dotenv.load_dotenv()


def split(
    dataset_name="tatsu-lab/alpaca",
    train_size=0.75,
    test_size=0.15,
    val_size=0.1,
    output_dir="data",
):
    print(f"Loading {dataset_name}")
    dataset = load_dataset(dataset_name)
    short_name = dataset_name.split("/")[-1]

    # Split the dataset
    print("Splitting the dataset")
    train_testvalid = dataset["train"].train_test_split(train_size=train_size)
    test_valid = train_testvalid["test"].train_test_split(
        test_size=test_size / (test_size + val_size)
    )

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Converting {dataset_name} ad splitting it into train/val/test")
    # Save the splits
    splits = {}
    splits["train"] = train_testvalid["train"]
    splits["val"] = test_valid["train"]
    splits["test"] = test_valid["test"]

    combined = DatasetDict(splits)

    current_directory = Path(".")
    shutil.rmtree(current_directory / "dataset", ignore_errors=True)
    os.mkdir(current_directory / "dataset")
    dataset_path = (current_directory / "dataset" / short_name).as_posix()
    combined.save_to_disk(dataset_path)

    # Compress the folder
    print(f"Compressing the folder {current_directory / 'dataset' / short_name}")
    folder_to_compress = (current_directory / "dataset" / short_name).as_posix()
    output_tar_file = f"{short_name}.tar.gz"
    bucket_name = "fine-tuning-research"
    print(f"Uploading {output_tar_file} to {bucket_name}")
    dataset_mover = DatasetMover()
    dataset_mover.upload(folder_to_compress, output_tar_file, bucket_name)


if __name__ == "__main__":
    Fire(split)
