import json
import os
import random
import string

import numpy as np
from PIL import Image
from transformers import AutoTokenizer


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    file_dir = os.path.dirname(file_path)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_qa(item):
    question = item["question"]
    answer = item["answer"]
    return f"Question: {question}\nLet's think step by step\nAnswer:\n{answer}\n\n"


def pad_to_target_tokens(
    question,
    few_shot_pool_token_ids,
    tokenizer,
    target_tokens,
    test_template="Question: {question}\nLet's think step by step\nAnswer:\n",
):
    test_prompt = test_template.format(question=question)
    test_token_ids = tokenizer.encode(test_prompt, add_special_tokens=False)

    remaining_tokens = target_tokens - len(test_token_ids)
    if remaining_tokens <= 0:
        return tokenizer.decode(
            test_token_ids[:target_tokens], skip_special_tokens=True
        )

    shuffled_ids = list(range(len(few_shot_pool_token_ids)))
    random.shuffle(shuffled_ids)

    prefix_ids = []
    for idx in shuffled_ids:
        fs_ids = few_shot_pool_token_ids[idx]
        if len(prefix_ids) + len(fs_ids) <= remaining_tokens:
            prefix_ids.extend(fs_ids)
        else:
            partial_gap = remaining_tokens - len(prefix_ids)
            if partial_gap > 0:
                prefix_ids.extend(fs_ids[:partial_gap])
            break

    if len(prefix_ids) < remaining_tokens and few_shot_pool_token_ids:
        padding_source_ids = few_shot_pool_token_ids[shuffled_ids[0]]
        repeat_count = (remaining_tokens // len(padding_source_ids)) + 1
        padding_ids = (padding_source_ids * repeat_count)[
            : remaining_tokens - len(prefix_ids)
        ]
        prefix_ids.extend(padding_ids)

    full_ids = prefix_ids + test_token_ids
    return tokenizer.decode(full_ids[:target_tokens], skip_special_tokens=True)


def generate_fixed_len_dataset(
    train_path,
    test_path,
    tokenizer_path,
    target_tokens,
    num_prompts,
    trust_remote_code=False,
    test_template="Question: {question}\nLet's think step by step\nAnswer:\n",
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=trust_remote_code
    )

    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)
    if num_prompts > 0 and num_prompts > len(test_data):
        multiplier = (num_prompts // len(test_data)) + 1
        test_data = (test_data * multiplier)[:num_prompts]
    elif num_prompts > 0:
        test_data = test_data[:num_prompts]

    few_shot_pool = [format_qa(item) for item in train_data]
    few_shot_pool_token_ids = [
        tokenizer.encode(fs, add_special_tokens=False) for fs in few_shot_pool
    ]

    output_data = []
    for i, test_item in enumerate(test_data):
        padded_question = pad_to_target_tokens(
            question=test_item["question"],
            few_shot_pool_token_ids=few_shot_pool_token_ids,
            tokenizer=tokenizer,
            target_tokens=target_tokens,
            test_template=test_template,
        )
        output_data.append(
            {
                "question": padded_question,
                "answer": test_item["answer"],
            }
        )
        if (i + 1) % 100 == 0:
            actual_tokens = len(
                tokenizer.encode(padded_question, add_special_tokens=False)
            )
            print(
                f"Processed {i + 1}/{len(test_data)}, last item tokens: {actual_tokens}"
            )

    token_counts = [
        len(tokenizer.encode(item["question"], add_special_tokens=False))
        for item in output_data
    ]
    print(
        f"Token count stats: min={min(token_counts)}, max={max(token_counts)}, avg={sum(token_counts)/len(token_counts):.1f}"
    )

    return output_data


def generate_random_images(mm_dataset_data, size):
    total_image_num = len(mm_dataset_data)
    print(f"begin to generate images, total {total_image_num}")

    file_count = 0
    for item in mm_dataset_data:
        image_paths = item.get("path")

        for image_path in image_paths:
            if not image_path:
                print("Error: The image path is none.")
                continue

            dir_name = os.path.dirname(image_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            random_array = np.random.randint(
                0, 256, (size[1], size[0], 3), dtype=np.uint8
            )

            img = Image.fromarray(random_array)
            img.save(image_path, quality=95)
            if os.path.isfile(image_path):
                file_count += 1

    print(f"Finish images generation. Image num: {file_count}")


def generate_mm_dataset(
    train_path,
    test_path,
    tokenizer_path,
    target_tokens=3500,
    num_prompts=1024,
    trust_remote_code=False,
    test_template="Question: {question}\nLet's think step by step\nAnswer:\n",
    image_dir="/tmp/datasets/image",
    size=None,
):
    output_data = []
    text_data = generate_fixed_len_dataset(
        train_path,
        test_path,
        tokenizer_path,
        target_tokens,
        num_prompts,
        trust_remote_code,
        test_template,
    )

    for item in text_data:
        random_string = "".join(
            random.choices(string.ascii_letters + string.digits, k=10)
        )
        item["type"] = "image"
        item["path"] = [f"{image_dir}/{random_string}.jpg"]
        output_data.append(item)

    size = tuple(map(int, size.split("x")))
    generate_random_images(output_data, size)
    return output_data


def generate_dataset_from_gsm8k(
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

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(len(dataset_new)):
            f.write(
                json.dumps(
                    {"question": f"{dataset_new[i]}", "answer": "none"},
                    ensure_ascii=False,
                )
            )
            f.write("\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate GSM8K dataset with exact input token length"
    )
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to GSM8K train.jsonl"
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to GSM8K test.jsonl"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output jsonl path"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to model tokenizer"
    )
    parser.add_argument(
        "--target_tokens", type=int, default=3500, help="Target input token length"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for tokenizer",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=0,
        help="Number of prompts to generate, 0 means all",
    )
    args = parser.parse_args()

    output_data = generate_fixed_len_dataset(
        train_path=args.train_path,
        test_path=args.test_path,
        tokenizer_path=args.tokenizer_path,
        target_tokens=args.target_tokens,
        num_prompts=args.num_prompts,
        trust_remote_code=args.trust_remote_code,
    )
    save_jsonl(output_data, args.output_path)
    print(f"Done! Output {len(output_data)} items to {args.output_path}")


if __name__ == "__main__":
    main()
