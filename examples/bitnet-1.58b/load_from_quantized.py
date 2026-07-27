import torch
import bitblas
from modeling_bitnet import BitnetForCausalLM
from tokenization_bitnet import BitnetTokenizer
import os
from transformers import GenerationConfig
import time

filepath = os.path.abspath(__file__)
dirpath = os.path.dirname(filepath)

torch.set_grad_enabled(False)
bitblas.set_log_level("INFO")

model_name_or_path = "BitBLASModel/open_llama_3b_1.58bits"
saved_model_path = os.path.join(dirpath, "models", f"{model_name_or_path}_bitblas")


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
    # load quantized model
    qmodel = (
        BitnetForCausalLM.from_quantized(
            saved_model_path,
        )
        .cuda()
        .half()
    )
    tokenizer = BitnetTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    # print("original model generated text:")
    # print(generate_text(model, tokenizer, "Hi, ", max_length=100))
    input_ids = torch.ones((1, 1), dtype=torch.long).cuda()
    # naive model inference
    output = qmodel(input_ids)
    print("original model output:", output)
    print("quantized model generated text:")
    print(generate_text(qmodel, tokenizer, "Hi, ", max_length=100))


if __name__ == "__main__":
    main()
