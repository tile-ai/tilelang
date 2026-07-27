# pylint: disable=missing-docstring, invalid-name
"""This is modified from https://huggingface.co/1bitLLM/bitnet_b1_58-3B/blob/main/utils_quant.py."""

import math
import argparse
import torch
import random

from eval_utils import get_test_dataset
from modeling_bitnet import BitnetForCausalLM
from tokenization_bitnet import BitnetTokenizer

from tqdm import tqdm

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--hf_path", default="1bitLLM/bitnet_b1_58-3B", type=str)
parser.add_argument("--seqlen", default=2048, type=int)


def calulate_loss(model, input, loss_fct):
    output = model(input, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
    shift_logits = output[:, :-1, :].contiguous()
    shift_labels = input[:, 1:]
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def main(args):
    datasets = ["c4", "wikitext2"]
    model = (
        BitnetForCausalLM.from_pretrained(
            args.hf_path,
            use_flash_attention_2=True,
            torch_dtype=torch.float16,
        )
        .cuda()
        .half()
    )
    with torch.no_grad():
        model._post_process_weights()
    tokenizer = BitnetTokenizer.from_pretrained(args.hf_path, use_fast=False)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum").cuda()

    ppl = []
    for dataset in datasets:
        testdata = get_test_dataset(dataset, tokenizer, seqlen=args.seqlen)
        acc_loss, count = 0.0, 0
        progress = tqdm(range(len(testdata)))
        for ii in progress:
            input = torch.Tensor(testdata[ii]).long().cuda().view(1, -1)
            loss = calulate_loss(model, input, loss_fct)
            count += input.size(-1) - 1
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss / count / math.log(2)}")

        avg_loss = acc_loss / count / math.log(2)
        ppl.append(2**avg_loss)
        print("{} PPL: {}".format(dataset, ppl[-1]))

    print(ppl)
    print("Avg PPL:", sum(ppl) / len(ppl))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
