import argparse
import torch

from modeling_bitnet import BitnetForCausalLM

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--hf_path", default="1bitLLM/bitnet_b1_58-3B", type=str)


def profile(model, input_data):
    import time

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


def main():
    model = BitnetForCausalLM.from_pretrained(
        "1bitLLM/bitnet_b1_58-3B",
        device_map="auto",
        low_cpu_mem_usage=True,
        use_flash_attention_2=True,
        torch_dtype=torch.float16,
    ).half()
    with torch.no_grad():
        model.quantize()
        model = torch.compile(model)

    benchmark_sets = [(1024, 1), (1, 2048)]
    for batch_size, seq_len in benchmark_sets:
        input_id = torch.ones(batch_size, seq_len).long().cuda()
        latency = profile(model, input_id)
        print(f"Batch size: {batch_size}, Seq len: {seq_len}, Latency: {latency}")


if __name__ == "__main__":
    main()
