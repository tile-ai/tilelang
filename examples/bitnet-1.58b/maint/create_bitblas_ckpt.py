import argparse
import torch
import bitblas
from transformers.utils.hub import cached_file
import os
from transformers import GenerationConfig
import time
import json

import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../")
from modeling_bitnet import BitnetForCausalLM
from tokenization_bitnet import BitnetTokenizer

filepath = os.path.abspath(__file__)
dirpath = os.path.dirname(filepath)

torch.set_grad_enabled(False)
bitblas.set_log_level("INFO")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="1bitLLM/bitnet_b1_58-3B")
parser.add_argument("--saved_model_path", type=str, default=None)
args = parser.parse_args()

model_name_or_path = args.model_name_or_path
saved_model_path = (
    os.path.join(dirpath, "models", f"{model_name_or_path}_bitblas") if args.saved_model_path is None else args.saved_model_path
)


def generate_text(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.lm_head.weight.device)
    # Generate cos and sin values
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    generation_config = GenerationConfig(
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )

    start_time = time.time()
    output_ids = model.generate(input_ids, generation_config=generation_config)
    end_time = time.time()

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    generation_time = end_time - start_time
    num_tokens = len(output_ids[0])
    tokens_per_second = num_tokens / generation_time

    print(f"Generated {num_tokens} tokens in {generation_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    return generated_text


def main():
    model = (
        BitnetForCausalLM.from_pretrained(
            model_name_or_path,
            use_flash_attention_2=False,
            torch_dtype=torch.float16,
        )
        .cuda()
        .half()
    )
    tokenizer = BitnetTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    # print("original model generated text:")
    # print(generate_text(model, tokenizer, "Hi, ", max_length=100))
    input_ids = torch.ones((1, 1), dtype=torch.long).cuda()
    # naive model inference
    output = model(input_ids)
    print("original model output:", output)

    model.quantize(fuse_qkv=True, fuse_gateup=True)
    print("original model generated text:")
    print(generate_text(model, tokenizer, "Hi, ", max_length=100))

    model.save_pretrained(saved_model_path)

    # load quant config
    quant_config_path = cached_file(model_name_or_path, "quantize_config.json")
    with open(quant_config_path, "r") as f:
        quant_config = json.load(f)
    print("quant config:")
    print(quant_config)
    quant_config["checkpoint_format"] = "bitblas"
    quant_config["fuse_qkv"] = True
    quant_config["fuse_gateup"] = True

    # save quant config
    quant_config_path = os.path.join(saved_model_path, "quantize_config.json")
    with open(quant_config_path, "w") as f:
        json.dump(quant_config, f)
    print("quant config saved to:", quant_config_path)

    # copy benchmark filed into saved model path
    file_list = [
        "configuration_bitnet.py",
        "eval_utils.py",
        "modeling_bitnet.py",
        "tokenization_bitnet.py",
        "utils_quant.py",
        "README.md",
    ]
    for file in file_list:
        file_path = cached_file(model_name_or_path, file)
        os.system(f"cp {file_path} {saved_model_path}")
    # load quantized model
    qmodel = (
        BitnetForCausalLM.from_quantized(
            saved_model_path,
        )
        .cuda()
        .half()
    )
    print("quantized model generated text:")
    print(generate_text(qmodel, tokenizer, "Hi, ", max_length=100))


if __name__ == "__main__":
    main()
