import json
import os

from openai import OpenAI
import argparse


def call(prompt):
    try:
        client = OpenAI(
            api_key="sk-",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen3-14b",
            messages=[
                {'role': 'user', 'content': prompt}
                ],
            extra_body={"enable_thinking": False},
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"error information：{e}")


def load_existing_output(output_path):
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"error: {e}")
    return []


def process_json_array(input_path, output_path, save_every=5):
    output_data = load_existing_output(output_path)
    start_index = len(output_data)

    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    total = len(input_data)
    print(f" {total} itmes, process {start_index} item, from {start_index + 1} start...")

    for i in range(start_index, total):
        item = input_data[i]
        original_text = item.get("text", "")
        label = item.get("label", "")

        try:
            if original_text.strip():
                prompt = (
                        "/no_think Paraphrase this text while preserving the original meaning precisely, but feel free to rephrase clauses or use different expressions where appropriate. The following text: " + original_text
                )
                refined_text = call(prompt)
            else:
                refined_text = original_text

            new_item = {"text": refined_text, "label": label}
            output_data.append(new_item)


        except Exception as e:
            output_data.append({"text": "[ERROR]", "label": label})

        if (i + 1) % save_every == 0 or i == total - 1:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(output_data, f_out, ensure_ascii=False, indent=2)
            print(f"💾 save {output_path} (total {len(output_data)} item)")

    print("🎉 all done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save_every", type=int, default=20)
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    process_json_array(input_path=args.input, output_path=args.output, save_every=args.save_every)
