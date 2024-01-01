# Data Preparation for LLM Research by AI Hero

This project contains the dataset creation scripts for fine tuning research. The goal is to separate the dataset building process as it may involve downloading raw sources, preparing them, weaving and splitting the sources into train/val/test spits.

## Target Data Formats
We'll use HuggingFace Datasets as a target format. For each dataset we will assume that we create a DatasetDict, having three splits - `train`, `val`, and `test`. Each object in the dataset dict being a HuggingFace Dataset.

- `completions` format: For fine tuning, each dataset split should have two columns: `prompt` and `completion` containing text that the LLM will be trained on. During training, the `text` field will be constructed from them using `row['prompt'] + row['completion']` 

- `text` format:  For traditional datasets, we might just have the `text` field. This will work, but will not be the best use as the tuning library won't be able to chart out actual completion vs predicted. 

## Current Datasets:
- `mmlu.py` - Creates a multiple-choice Q&A dataset using the MMLU dataset (downloaded locally) 
- `open_instruct.py` - Convert the `VMWare/open-instruct` dataset into the fine-tuning format above.
- `yacheq.py` - for now, the autonomous agent research that AI Hero is undertaking is stored in this repo.

## Setup

### Requirements
```sh
pip install -r requirements.txt
```

### Environment
You'll also need a `.env` file in the top folder from where you'll run the jobs.

```
# For loading/saving data to Huggingface
HF_TOKEN=

# For loading/saving data to S3
S3_ENDPOINT=s3.amazonaws.com
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=
S3_REGION=us-east-2
S3_SECURE=true
```