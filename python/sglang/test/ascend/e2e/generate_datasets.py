import json
import os
import random
import string

import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer


def generate_dataset(
    model_path, source_dataset_path, batch_size, input_len, output_file
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = []
    with open(source_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data["question"])

    dataset_new = []
    for sentence in dataset:
        words = tokenizer.tokenize(sentence)
        len_num = len(words) // input_len
        if len_num == 0:
            multiplier = (input_len // len(words)) + 1
            repeated_len = words * multiplier
            words = repeated_len[:input_len]
            decoded_text = tokenizer.convert_tokens_to_string(words)
            if len(words) != input_len:
                print(
                    f"Generate DataSet Error: the length of new input is {len(words)}, not {input_len}"
                )
            dataset_new.append(decoded_text)

    batch_num = len(dataset_new) // batch_size
    if batch_num == 0:
        multiplier = (batch_size // len(dataset_new)) + 1
        repeated_batch = dataset_new * multiplier
        dataset_new = repeated_batch[:batch_size]
    else:
        dataset_new = dataset_new[:batch_size]

    random.shuffle(dataset_new)

    if len(dataset_new) != batch_size:
        print(
            f"Generate DataSet Error: the size of new dataset is {len(dataset_new)}, not {batch_size}"
        )

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(len(dataset_new)):
            f.write(
                json.dumps(
                    {"question": f"{dataset_new[i]}", "answer": "none"},
                    ensure_ascii=False,
                )
            )
            f.write("\n")


def generate_json(text_json, image_json, image_dir):
    with open(text_json, "r") as input_file, open(image_json, "w") as output_file:
        for line in input_file:
            json_obj = json.loads(line.strip())
            random_string = "".join(
                random.choices(string.ascii_letters + string.digits, k=10)
            )
            json_obj["type"] = "image"
            json_obj["path"] = [f"{image_dir}/{random_string}.jpg"]
            json.dump(json_obj, output_file)
            output_file.write("\n")


def generate_random_images(jsonl_path, size):
    if not os.path.exists(jsonl_path):
        print("jsonl file does not exist")
        return

    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    print(f"begin to generate, total {total_lines}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines):
            try:
                data = json.loads(line.strip())
                image_paths = data.get("path")

                for image_path in image_paths:
                    if not image_path:
                        continue

                    dir_name = os.path.dirname(image_path)
                    if dir_name and not os.path.exists(dir_name):
                        os.makedirs(dir_name, exist_ok=True)

                    random_array = np.random.randint(
                        0, 256, (size[1], size[0], 3), dtype=np.uint8
                    )

                    img = Image.fromarray(random_array)
                    img.save(image_path, quality=95)

            except Exception as e:
                print(e)

    print("Finished")


if __name__ == "__main__":

    batch_size = 1024
    input_len = 30

    generate_dataset(
        model_path="/models/xxx/",
        source_dataset_path="/root/.cache/modelscope/hub/datasets/grade_school_math/test.jsonl",
        batch_size=batch_size,
        input_len=input_len,
        output_file=f"GSM8K-in{input_len}-bs{batch_size}.jsonl",
    )
    generate_json(
        text_json=f"GSM8K-in{input_len}-bs{batch_size}.jsonl",
        image_json=f"1024x1024_${input_len}.jsonl",
        image_dir="/root/.cache/modelscope/hub/datasets/sglang_test/1024x1024",
    )
    generate_random_images(
        jsonl_path=f"1024x1024_${input_len}.jsonl",
        size=(1024, 1024),
    )
