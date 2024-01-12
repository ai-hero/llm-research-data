import shutil
from pathlib import Path
import os
from datasets import Dataset, DatasetDict, DatasetInfo
from datasets import load_dataset
from fire import Fire
from numpy import dot, short
from utils import DatasetMover
import dotenv

dotenv.load_dotenv()


def split(
    dataset_name="Clinton/Text-to-sql-v1",
    train_size=0.9,
    test_size=0.05,
    val_size=0.05,
    output_dir="dataset",
    short_name="text-to-sql",
    bucket_name="fine-tuning-research",
):
    print(f"Loading {dataset_name}")
    dataset = load_dataset(dataset_name)
    new_dataset = []
    for row in dataset["train"]:
        text = row["text"].strip()
        text = text.replace("###", "\n###").strip()
        SPLITTER = "### Response:"
        row["prompt"] = text[: text.index(SPLITTER) + len(SPLITTER)]
        row["completion"] = text[text.index(SPLITTER) + len(SPLITTER) :]
        for to_delete in [
            "instruction",
            "input",
            "response",
            "text",
        ]:
            row.pop(to_delete)
        new_dataset.append(row)

    if not short_name:
        short_name = dataset_name.split("/")[-1]

    # Create a Dataset from your list of dictionaries
    new_dataset = Dataset.from_list(new_dataset)

    # If you're creating a new dataset from scratch:
    dataset = DatasetDict(
        {
            "train": new_dataset  # Assign the new dataset as the train split
        }
    )

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

    for split in splits:
        print(f"Example in split {split}:")
        for row in splits[split]:
            print(row)
            break

    combined = DatasetDict(splits)
    dataset_info = DatasetInfo(
        description="Contains data from Clinton/Text-to-sql-v1 but not formatted into completions format",
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
