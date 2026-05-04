import json
import os

from openai import OpenAI
import argparse


# 连接到 Ollama 的 OpenAI 兼容 API（默认端口 11434）
def call(prompt):
    try:
        client = OpenAI(
            # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
            # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
            api_key="sk-a2a602fc8ec64257ac6e0bd68979d1a2",
            # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 调用时使用 exact model name from `ollama list`
        response = client.chat.completions.create(
            model="qwen3-14b",  # ← 关键：去掉 -q4
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={"enable_thinking": False},
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")


def load_existing_output(output_path):
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"警告：输出文件格式错误，将重新开始。错误: {e}")
    return []


def process_json_array(input_path, output_path, save_every=5):
    # 加载已处理的结果
    output_data = load_existing_output(output_path)
    start_index = len(output_data)

    # 读取输入
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    total = len(input_data)
    print(f"总共 {total} 条数据，已处理 {start_index} 条，从第 {start_index + 1} 条开始...")

    for i in range(start_index, total):
        item = input_data[i]
        original_text = item.get("text", "")
        label = item.get("label", "")

        try:
            if original_text.strip():
                prompt = (
                        "/no_think Rewrite this text in your own words, ensuring the core meaning remains identical, but using distinct phrasing and structure. The following text: " + original_text
                )
                refined_text = call(prompt)
            else:
                refined_text = original_text

            new_item = {"text": refined_text, "label": label}
            output_data.append(new_item)

            print(f"✅ 已处理第 {i + 1}/{total} 条")

        except Exception as e:
            print(f"❌ 第 {i + 1} 条出错: {e}")
            # 保留原文或标记错误
            output_data.append({"text": "[ERROR]", "label": label})

        # 每 save_every 条保存一次完整文件
        if (i + 1) % save_every == 0 or i == total - 1:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(output_data, f_out, ensure_ascii=False, indent=2)
            print(f"💾 已保存至 {output_path}（共 {len(output_data)} 条）")

    print("🎉 全部处理完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用 Ollama 模型对 JSON 数据进行 paraphrase")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--save_every", type=int, default=20, help="每处理 N 条保存一次")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:  # 防止 args.output 是当前目录下的文件（如 "test.json"）
        os.makedirs(output_dir, exist_ok=True)

    process_json_array(input_path=args.input, output_path=args.output, save_every=args.save_every)

    # # 使用
    # process_json_array('/data5/storage4/wudauser05/yyc/RobustStressBench/domain_specific_model_specific/xsum/test.json',
    #                    './specific_xsum_test.json', save_every=5)
