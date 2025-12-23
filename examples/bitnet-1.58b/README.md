---
license: mit
---

This is a Tilelang Implementation for the reproduced 1.58bit model from [1bitLLM/bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B). We replaced the original simulated Int8x3bit Quantized Inference Kernel with INT8xINT2 Kernel. We also evaluated the model's correctness and performance through `eval_correctness.py` and `benchmark_inference_latency.py`.

## Make Checkpoints for vLLM

We provide two scripts to make the checkpoints for vLLM. The first script is `generate_bitnet_model_native_format.sh`, which is used to make a checkpoint with fp16 uncompressed metaadta, the main difference with the original checkpoint is the `quant_config.json`, which allow vLLM to load the model and execute with a quant extension.

```bash
# move to the integration directory
cd /root/to/BitBLAS/integration/BitNet
# make the checkpoint
./maint/generate_bitnet_model_native_format.sh
# the output ckpy will be saved in the `./models/ckpt_bitnet_b1_58-3B` directory
```

The second script is `generate_bitnet_model_bitblas_format.sh`, which is used to make a checkpoint with BitBLAS compressed metadata, which can avoid the online dequantize sage for the profiling of vLLM, which lead to more efficient memory utilization.

```bash
./maint/generate_bitnet_model_bitblas_format.sh ./models/ckpt_bitnet_b1_58-3B ./models/ckpt_bitnet_b1_58-3B_bitblas
# the output ckpy will be saved in the `./models/ckpt_bitnet_b1_58-3B_bitblas` directory
```

Finnaly, you can use the ckpt in vLLM with:

```bash
cd vllm_workspace
# inference with the ckpt with fp16 uncompressed metadata
python3 inference_with_native_format.py
# inference with the ckpt with BitBLAS compressed metadata
python3 inference_with_bitblas_format.py
```

**Benchmark results of vLLM**

| Model                  | Framework                | BS16IN32OUT128 | BS1IN512OUT1024 | BS32IN32OUT128 |
|------------------------|--------------------------|----------------|-----------------|----------------|
| bitnet-3b-1.58bits     | pytorch                  | 106.83         | 49.34           | 209.03         |
| bitnet-3b-1.58bits     | pytorch-tilelang         | 240.33         | 103.09          | 493.31         |
| bitnet-3b-1.58bits     | vllm-tilelang            | 379.25         | 117.43          | 752.55         |
| bitnet-3b-1.58bits     | vllm-tilelang-cuda-graph | 2543.58        | 1621.08         | 2731.79        |

## BitBLAS Results

### Performance

**Note:** To reproduce the results of BitBLAS, Please checkout the `benchmark_inference_latency.py`. To reproduce the results of the original model, Please checkout the [1bitLLM/bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) repo.

|      Model      | Device | batchsize | in_seq |   model  | bitnet-1.58b-3b-huggingface | bitnet-1.58b-3b-bitblas |
|:---------------:|:------:|:---------:|:------:|:--------:|:---------------------------:|:-----------------------:|
| bitnet_b1_58-3B |  A100  |     1     |    1   | LLAMA-3B |         177.6729107         |       64.17962909       |
| bitnet_b1_58-3B |  A100  |    128    |    1   | LLAMA-3B |         188.6145592         |       63.48158518       |
| bitnet_b1_58-3B |  A100  |     1     |  2048  | LLAMA-3B |         348.7066031         |       202.6877999       |

### On-the-Fly GPU Memory Footprint

We measured the GPU memory footprint through the `nvidia-smi` command. Please checkout `nvidia_measure_memory.sh` to get the real-time GPU memory usage. And then start a `benchmark_model_10k_loops.py` workload to measure the overall GPU memory usage.

|    **Model**    | **Device** | **batchsize** | **in_seq** | **bitnet-1.58b-3b-huggingface** | **bitnet-1.58b-3b-bitblas** |
|:---------------:|:----------:|:-------------:|:----------:|:-------------------------------:|:---------------------------:|
| bitnet_b1_58-3B |    A100    |       1       |      1     |             7595 MB             |           1729 MB           |
| bitnet_b1_58-3B |    A100    |      128      |      1     |             7677 MB             |           1789 MB           |
| bitnet_b1_58-3B |    A100    |       1       |    2048    |             8731 MB             |           3163 MB           |

## PPL and Zero-shot Accuracy

The number is Reported from the [1bitLLM/bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B), Please checkout the `eval_ppl.py`.

PPL and zero-shot accuracy:
| Models | PPL| ARCe| ARCc| HS | BQ | OQ | PQ | WGe | Avg
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| FP16 700M (reported) | 12.33 | 54.7 | 23.0 | 37.0 | 60.0 | 20.2 | 68.9 | 54.8 | 45.5 |
| BitNet b1.58 700M (reported) | 12.87 | 51.8 | 21.4 | 35.1 | 58.2 | 20.0 | 68.1 | 55.2 | 44.3 |
| BitNet b1.58 700M (reproduced) | 12.78 | 51.4 | 21.8 | 35.0 | 59.6 | 20.6 | 67.5 | 55.4 | 44.5 |
| FP16 1.3B (reported)    | 11.25  | 56.9 | 23.5 | 38.5 | 59.1 | 21.6 | 70.0 | 53.9 | 46.2
| BitNet b1.58 1.3B (reported)    | 11.29  | 54.9 | 24.2 | 37.7 | 56.7 | 19.6 | 68.8 | 55.8 | 45.4 |
| BitNet b1.58 1.3B (reproduced)    | 11.19 | 55.8 | 23.7 | 37.6 | 59.0 | 20.2 | 69.2 | 56.0 | 45.9
| FP16 3B (reported)    | 10.04   | 62.1 | 25.6 | 43.3 | 61.8 | 24.6 | 72.1 | 58.2 | 49.7
| BitNet b1.58 3B (reported)    | 9.91   | 61.4 | 28.3 | 42.9 | 61.5 | 26.6 | 71.5 | 59.3 | 50.2
| BitNet b1.58 3B (reproduced)    | 9.88 | 60.9 | 28.0 | 42.3 | 58.3 | 26.0 | 71.4 | 60.3 | 49.6 |

The differences between the reported numbers and the reproduced results are possibly variances from the training data processing, seeds, or other random factors.

## Citations

```bibtex
@article{ma2024era,
  title={The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits},
  author={Ma, Shuming and Wang, Hongyu and Ma, Lingxiao and Wang, Lei and Wang, Wenhui and Huang, Shaohan and Dong, Li and Wang, Ruiping and Xue, Jilong and Wei, Furu},
  journal={arXiv preprint arXiv:2402.17764},
  year={2024}
}
```
