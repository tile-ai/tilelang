# ruff: noqa
import torch

import numpy as np
import torch.nn.functional as F

from lm_eval.base import BaseLM
from datasets import load_dataset


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_test_dataset(dataset_name, tokenizer, seqlen=2048):
    if dataset_name == "wikitext2":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testdata = "".join(testdata["text"]).split("\n")
    elif dataset_name == "c4":
        testdata = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")[
            "text"
        ]
    else:
        raise NotImplementedError

    testdata = [item for item in testdata if item != ""]
    tokenized_text = [tokenizer(item, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id] for item in testdata]

    data, doc = [], [tokenizer.bos_token_id]
    for sen in tokenized_text:
        if len(sen) > seqlen:
            continue
        if len(doc) + len(sen) > seqlen:
            data.append(doc)
            doc = [tokenizer.bos_token_id]
        doc.extend(sen)
    if len(doc) > 1 and len(doc) <= seqlen:
        data.append(doc)
    return data


class LMEvalAdaptor(BaseLM):
    def __init__(self, model_name, model, tokenizer, batch_size=1, max_length=-1):
        super().__init__()

        assert isinstance(batch_size, int)

        self.model_name = model_name
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length != -1:
            return self._max_length
        if hasattr(self.model.config, "n_ctx"):
            return self.model.config.n_ctx
        elif hasattr(self.model.config, "max_position_embeddings"):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "n_positions"):
            return self.model.config.n_positions
        elif "bloom" in self.model_name:
            return 2048
        elif "llama" in self.model_name:
            return 2048  # TODO: did not check this
        elif "mpt" in self.model_name:
            return 2048
        elif "falcon" in self.model_name:
            return 2048
        else:
            print(self.model.config)
            raise NotImplementedError

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str, add_special_tokens=True):
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            context, continuation = context.strip(), continuation.strip()
            if context == "":
                # end of text as context
                context_enc = [self.eot_token_id]
            else:
                context_enc = self.tok_encode(context, add_special_tokens=True)

            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            out = self.model(inps)[0]
        return out

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False)
