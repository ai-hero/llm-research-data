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
    def __init__(self):
        self.prompt_template_str = """<identity>You are an agent that helps answer multiple choice questions.</identity>
<objective>Answer the question</objective>
<data>
this.data = {
    "question": "{{question | dumps}}",
    "choices": {{choices | dumps}},
    "answer": ""
}
</data>
<tests>
if(!choices.includes(answer)){
    console.error("Answer must be a valid choice")
}else{
    console.log("Answer is a valid choice")
}
</tests>
<actions> 
  <user>
    <ask>Select the best answer.</ask>
  </user>"""
        self.completion_template_str = """  <action>
    <do>set `answer` using GPT knowledge</do>
    <step>
    this.data.answer = {{answer | dumps}}
    </step>
  </action>
</actions>"""

        # Setup Jinja2 environment
        self.env = Environment(loader=FileSystemLoader("."))
        self.env.filters["dumps"] = json.dumps
        self.prompt_template = self.env.from_string(self.prompt_template_str)
        self.completion_template = self.env.from_string(self.completion_template_str)

    def _extract(self, remaining):
        assert remaining.startswith("<"), "Expected <, found " + remaining[0:10]
        tag = remaining[1:].split(">")[0]
        end = remaining.find("</" + tag + ">")
        assert end != -1
        return (
            tag,
            remaining[len(tag) + 2 : end].strip(),
            remaining[end + len(tag) + 3 :].strip(),
        )

    def parse(self, text):
        remainder = text.strip()
        parsed = {}
        while remainder:
            tag, content, remainder = self._extract(remainder)
            if tag == "actions":
                remainder_actions = content
                parsed_actions = []
                while remainder_actions:
                    # Parse the action tag
                    action_tag, action_details, remainder_actions = self._extract(
                        remainder_actions
                    )
                    if action_tag == "action":
                        remainder_step_actions = action_details
                        parsed_action_tags = {}
                        while remainder_step_actions:
                            # parse internal tags
                            intag, step, remainder_step_actions = self._extract(
                                remainder_step_actions
                            )
                            parsed_action_tags[intag] = step
                        parsed_actions.append(parsed_action_tags)
                    elif action_tag == "user":
                        remainder_step_actions = action_details
                        parsed_action_tags = {}
                        while remainder_step_actions:
                            # parse internal tags
                            intag, step, remainder_step_actions = self._extract(
                                remainder_step_actions
                            )
                            parsed_action_tags[intag] = step
                        parsed_actions.append(parsed_action_tags)
                parsed[tag] = parsed_actions
            else:
                parsed[tag] = content
        return parsed

    def test(self, parsed):
        code = ""
        test = ""
        for tag, content in parsed.items():
            if tag == "actions":
                for action in content:
                    if "do" in action:
                        code += "# " + action["do"] + "\n"
                    if "step" in action:
                        code += action["step"] + "\n"
            elif tag == "test":
                test = content
            elif tag == "data":
                code += content + "\n"
            # elif tag == "template":
            #     code += '"""' + content + '"""' + "\n"
            # elif tag == "schema":
            #     code += content + "\n"
            elif tag == "objective":
                code += "\n# " + content + "\n"
            elif tag == "intent":
                code += "\n# " + content + "\n"
            else:
                pass
        code += test + "\n"
        try:
            # print("Testing code:")
            # print(code)
            # print("----------")
            # exec(code)
            return True
        except Exception:
            traceback.print_exc()
            return False

    def transform(self, row: dict):
        return {
            "prompt": self.prompt_template.render(**row),
            "completion": self.completion_template.render(**row),
        }


def gen(split="dev"):
    mmlu_data = Path.home() / "data" / "mmlu"
    assert mmlu_data.exists()

    mmlu_data_transformer = MMLUDataTransformer()

    # Use glob to find all CSV files in the 'dev' folder
    count = 0
    error = 0
    for filepath in glob.glob((mmlu_data / split / "*.csv").as_posix()):
        print(f"Processing {filepath}")
        with open(filepath, "r") as fr:
            reader = csv.reader(fr)
            for row in reader:
                lookup = {
                    "A": 0,
                    "B": 1,
                    "C": 2,
                    "D": 3,
                }
                choices = row[1:5]
                row = mmlu_data_transformer.transform(
                    {
                        "question": row[0],
                        "choices": choices,
                        "answer": choices[lookup[row[5]]],
                    }
                )
                text = row["prompt"] + "\n" + row["completion"]
                parsed = mmlu_data_transformer.parse(text)
                if mmlu_data_transformer.test(parsed):
                    count += 1
                    yield row
                else:
                    error += 1
    print("split", split)
    print("count", count)
    print("error", error)


if __name__ == "__main__":
    dataset_name = "yacheq"
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
        description="Contains MMLU dataset where the answer is entirely in text. Schema is part of data",
        version="0.3.0",
    )
    for split, dataset in combined.items():
        dataset.dataset_info = dataset_info

    for split in splits:
        print(f"Example in split {split}:")
        for row in splits[split]:
            print(row)
            break

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
