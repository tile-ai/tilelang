# pylint: disable=missing-docstring, invalid-name
"""This is modified from https://huggingface.co/1bitLLM/bitnet_b1_58-3B/blob/main/utils_quant.py to work with BitBLAS."""

import torch
from torch import nn
from bitblas.cache import global_operator_cache, get_database_path
from bitblas import Matmul, MatmulConfig
from bitblas import auto_detect_nvidia_target
from logging import getLogger

logger = getLogger(__name__)
BITBLAS_TARGET = auto_detect_nvidia_target()
BITBLAS_DATABASE_PATH = get_database_path()


def weight_quant(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s = 1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1) / s
    return result.type(dtype)


def activation_quant(x, num_bits=8):
    dtype = x.dtype
    x = x.float()
    Qn = -(2 ** (num_bits - 1))
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp) / s
    return result.type(dtype)


class BitLinearBitBLAS(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_bits=1,
        input_bits=8,
        **kwargs,
    ):
        super().__init__()
        """
        RMSNorm is placed outside BitLinear
        """
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        matmul_config = MatmulConfig(
            N=self.out_features,  # N dimension
            K=self.in_features,  # K dimension
            A_dtype="int8",  # activation A dtype
            W_dtype="int2",  # weight W dtype
            accum_dtype="int32",  # accumulation dtype
            out_dtype="float32",  # output dtype
            layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
            with_bias=False,  # bias
            # configs for weight only quantization
            group_size=None,  # setting for grouped quantization
            with_scaling=False,  # setting for scaling factor
            with_zeros=False,  # setting for zeros
            zeros_mode=None,  # setting for how to calculating zeros
        )
        ENABLE_TUNING = True
        self.bitblas_matmul = self._get_or_create_bitblas_operator(matmul_config, ENABLE_TUNING)

        self.format = "bitnet"
        self.Qp = 2 ** (self.input_bits - 1) - 1

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
            logger.info(f"Loaded {global_operator_cache.size()} operators from database.")

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            # should disable tuning for the first time because we may require loading bitblas operator from database.
            bitblas_matmul = Matmul(config, target=BITBLAS_TARGET, enable_tuning=False)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
                print("BitBLAS Tuning done, appended operator to global_operator_cache.")
            else:
                print("BitBLAS Operator created.")
        else:
            print("BitBLAS Operator found in global_operator_cache.")
        return bitblas_matmul

    def replace_weight_param_with_qweight(self):
        if hasattr(self, "weight"):
            del self.weight
        quant_weight = torch.empty(self.bitblas_matmul.retrieve_weight_shape())
        self.qweight = nn.Parameter(quant_weight, requires_grad=False)
        self.format = "bitblas"

    @classmethod
    def from_bit_linear(cls, bitlinear, weight_group=1):
        bitblas_linear = cls(bitlinear.in_features, bitlinear.out_features, weight_bits=1, input_bits=8)
        sw, qweight = bitblas_linear.create_bitblas_weights(bitlinear.weight, weight_group)
        bitblas_linear.register_buffer("qweight", qweight)
        bitblas_linear.register_buffer("sw", sw)
        if bitlinear.bias is not None:
            bitblas_linear.register_buffer("bias", bitlinear.bias)
        else:
            bitblas_linear.bias = None
        return bitblas_linear

    def create_bitblas_weights(self, weight, weight_group=1):
        if weight_group:
            hidden_size = weight.size(0)
            group_size = hidden_size // weight_group

            sw_list = []
            qweight_list = []

            for i in range(weight_group):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size

                sw = 1 / weight[start_idx:end_idx].abs().mean().clamp(min=1e-5)
                sw_list.append(sw.repeat(group_size))

                qweight = self.weight_quant(weight[start_idx:end_idx]).detach()
                qweight_list.append(qweight)

            sw = torch.cat(sw_list, dim=0)
            qweight = torch.cat(qweight_list, dim=0)
        else:
            sw = 1 / weight.abs().mean().clamp(min=1e-5)
            qweight = self.weight_quant(weight).detach()
        qweight = self.bitblas_matmul.transform_weight(qweight)
        qweight = nn.Parameter(qweight, requires_grad=False)
        return sw, qweight

    def post_process_weights(self):
        sw = 1 / self.weight.abs().mean().clamp(min=1e-5)
        self.sw = sw
        quant_weight = self.weight_quant(self.weight).detach()
        quant_weight = self.bitblas_matmul.transform_weight(quant_weight)
        # remove self.weight and replace it with quant_weight
        if hasattr(self, "weight"):
            del self.weight
        self.qweight = nn.Parameter(quant_weight, requires_grad=False)
        self.format = "bitblas"

    @staticmethod
    def weight_quant(weight):
        weight = weight.float()
        s = 1 / weight.abs().mean().clamp(min=1e-5)
        result = (weight * s).round().clamp(-1, 1)
        return result.type(torch.int8)

    @torch.compile
    def activation_quant(self, x, num_bits=8):
        x = x.float()
        Qn = -(2 ** (num_bits - 1))
        Qp = 2 ** (num_bits - 1) - 1
        s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (x * s).round().clamp(Qn, Qp)
        return result.type(torch.int8), s

    @torch.compile
    def post_quant_process(self, input, si, sw):
        out = input / si
        out = out / sw
        out = out.half()
        return out

    # for the correctness evaluation.
    def native_forward(self, input):
        quant_input = input + (activation_quant(input, self.input_bits) - input).detach()
        quant_weight = self.weight + (weight_quant(self.weight, self.weight_bits) - self.weight).detach()

        out = nn.functional.linear(quant_input, quant_weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def forward_fp32_simulated(self, input):
        quant_input, si = self.activation_quant(input, self.input_bits).detach()
        quant_weight = self.weight_quant(self.weight).detach()

        fp32_simulated_input = quant_input.float()
        fp32_simulated_weight = quant_weight.float()
        fp32_simulated_out = nn.functional.linear(fp32_simulated_input, fp32_simulated_weight)

        sw = 1 / self.weight.abs().mean().clamp(min=1e-5)
        # if / (si * sw) it will inf in some cases
        out = fp32_simulated_out / si
        out = out / sw
        out = out.half()
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def forward(self, input):
        # return self.forward_fp32_simulated(input)
        quant_input, si = self.activation_quant(input, self.input_bits)
        fp32_out = self.bitblas_matmul(quant_input, self.qweight)
        sw = self.sw
        # if / (si * sw) it will inf in some cases
        out = self.post_quant_process(fp32_out, si, sw)

        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
        return out


# Naive BitLinear from HuggingFace
class BitLinear(nn.Linear):
    def __init__(self, *kargs, weight_bits=1, input_bits=8, **kwargs):
        super(BitLinear, self).__init__(*kargs, **kwargs)
        """
        RMSNorm is placed outside BitLinear
        """
        self.weight_bits = weight_bits
        self.input_bits = input_bits

    def forward(self, input):
        quant_input = input + (activation_quant(input, self.input_bits) - input).detach()
        quant_weight = self.weight + (weight_quant(self.weight, self.weight_bits) - self.weight).detach()

        out = nn.functional.linear(quant_input, quant_weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
