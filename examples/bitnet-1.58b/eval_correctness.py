import torch
import bitblas
from modeling_bitnet import BitnetForCausalLM
from tokenization_bitnet import BitnetTokenizer
from transformers import GenerationConfig
import time
import transformers

print(f"transformers version is {transformers.__version__}")

# version must be lower than or equal to 4.40
assert transformers.__version__ <= "4.40.0"

torch.set_grad_enabled(False)
bitblas.set_log_level("INFO")


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


def profile(model, input_data):
    import numpy as np

    model = model.cuda()
    model.eval()

    def get_runtime(num_repeats=1):
        tic = time.time()
        for _ in range(num_repeats):
            _ = model(input_data)
        torch.cuda.synchronize()
        return (time.time() - tic) * 1000 / num_repeats

    with torch.no_grad():
        st = time.time()
        while time.time() - st < 1.0:
            get_runtime()  # warmup
        warmup_runtime = get_runtime()
        num_repeats = max(1, int(1000 / warmup_runtime))
        times = get_runtime(num_repeats)
    return np.mean(times)


model_path = "1bitLLM/bitnet_b1_58-3B"


def main():
    model = (
        BitnetForCausalLM.from_pretrained(
            model_path,
            use_flash_attention_2=False,
            torch_dtype=torch.float16,
        )
        .cuda()
        .half()
    )

    tokenizer = BitnetTokenizer.from_pretrained(model_path, use_fast=False)
    input_id = tokenizer("Hello")["input_ids"]
    input_id = torch.tensor(input_id).unsqueeze(0).cuda()

    print("original model generated text:")
    print(generate_text(model, tokenizer, "Hello", max_length=100))

    model.quantize()
    print("quantized model generated text:")
    print(generate_text(model, tokenizer, "Hello", max_length=100))


if __name__ == "__main__":
    main()
