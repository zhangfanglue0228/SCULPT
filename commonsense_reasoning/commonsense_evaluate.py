# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# ... (保留原有的版权声明和 imports) ...
import sys,io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
import copy
import json
import os
import re
import sys
import argparse
import fire
import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# 定义所有常识推理数据集列表
ALL_COMMONSENSE_DATASETS = [
    "boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", 
    "ARC-Challenge", "ARC-Easy", "openbookqa"
]

def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()

    # --- 1. 模型加载 (只执行一次) ---
    print(f"Loading model: {args.base_model} with adapter: {args.lora_weights}")
    tokenizer, model = load_model(args)

    # --- 2. 权重合并逻辑 (只执行一次) ---
    # 注意：这里增加了 .lower() 处理，兼容传入 scuplt 小写的情况
    adapter_name_lower = args.adapter.lower()
    
    if adapter_name_lower in ["lora", "dora", "svdlora", "svdlora_v2", "svddora", "sculpt"]:
        print(f"Attempting to merge {args.adapter} weights into the original weights...")
        key_list = [(key,module) for key, module in model.model.named_modules()]
        for key,module in key_list:
            if isinstance(model.peft_config.target_modules, str):
                target_module_found = re.fullmatch(model.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in model.peft_config.target_modules)

            # DoRA 特殊处理逻辑
            if adapter_name_lower == "dora":
                if hasattr(model.peft_config, 'Wdecompose_target_modules') and model.peft_config.Wdecompose_target_modules != None:
                    if isinstance(model.peft_config.Wdecompose_target_modules, str):
                        wdecompose_target_module_found = re.fullmatch(model.peft_config.Wdecompose_target_modules, key)
                    else:
                        wdecompose_target_module_found = any(key.endswith(target_key) for target_key in model.peft_config.Wdecompose_target_modules)
                else: 
                    wdecompose_target_module_found = False
            else:
                wdecompose_target_module_found = False

            if target_module_found:
                # print(f"found {key}")
                module.merge_weights = True
                module.train(mode=False)
            elif wdecompose_target_module_found:
                # print(f"found {key}")
                module.merge_weights = True
                module.train(mode=False)
        print("Merge configuration complete.")

    # --- 3. 确定要跑的数据集列表 ---
    if args.dataset == 'all':
        target_datasets = ALL_COMMONSENSE_DATASETS
    else:
        target_datasets = [args.dataset]

    # --- 4. 循环评估 ---
    for ds_name in target_datasets:
        print(f"\n{'='*20}\nStart Evaluating: {ds_name}\n{'='*20}")
        
        # 临时修改 args.dataset 供 helper 函数使用 (extract_answer, load_data)
        args.dataset = ds_name 
        
        try:
            dataset = load_data(args)
        except FileNotFoundError:
            print(f"⚠️ Warning: Dataset file for {ds_name} not found, skipping...")
            continue
            
        batches = create_batch(dataset, args.batch_size)
        
        # 定义内部评估函数 (保持闭包环境)
        def evaluate_batch(instructions, input=None, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=32, **kwargs):
            prompts = [generate_prompt(instruction, input) for instruction in instructions]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(device)
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                **kwargs,
            )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
            s = generation_output.sequences
            outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
            outputs = [o.split("### Response:")[-1].strip() for o in outputs]
            return outputs

        total = len(batches)
        correct = 0
        current = 0
        
        pbar = tqdm(total=total, desc=f"Eval {ds_name}")
        for idx, batch in enumerate(batches):
            current += len(batch)
            instructions = [data.get('instruction') for data in batch]
            
            outputs = evaluate_batch(instructions)

            for data, output in zip(batch, outputs):
                label = data.get('answer')
                predict = extract_answer(args, output)
                if label == predict:
                    correct += 1
                print("-" * 20)
                print(f"Instruction: {data.get('instruction')}")
                print(f"Output Raw : {output}")
                print(f"Prediction : {predict}")
                print(f"Label      : {label}")
                print("-" * 20)
            
            pbar.set_postfix({'acc': f"{correct / current:.4f}"})
            pbar.update(1)
        pbar.close()

        accuracy = correct / current
        print(f"✅ Finished {ds_name}: Accuracy = {accuracy:.4f}")

        # 结果写入文件
        result_file_path = os.path.join(args.lora_weights, "ALL_results.txt")
        with open(result_file_path, "a") as f:
            f.write(f"{ds_name}: {accuracy}\n")
    
    print('\n🎉 All evaluations finished.')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    # 确保你的数据都在 dataset/数据集名称/test.json 下
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r', encoding='utf-8'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    # 修改: 移除了 choices 限制，增加了 'all' 选项，移除了 required=True (默认为 all)
    parser.add_argument('--dataset', default='all', help="Dataset name or 'all' for 8 commonsense datasets")
    parser.add_argument('--model', choices=['LLaMA-7B', "LLaMA-13B",'LLaMA2-7B','LLaMA3-8B'], required=True)
    # 修改: 移除了 adapter 的 choices 强校验，防止大小写导致报错，逻辑中转为大写判断
    parser.add_argument('--adapter', type=str, required=True) 
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit
    if "LLaMA" in args.model:
        if "Llama-3" in base_model:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ) 
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"":0}
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()