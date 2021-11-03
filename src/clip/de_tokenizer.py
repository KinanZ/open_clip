import torch
from transformers import AutoTokenizer, BertTokenizer
from typing import Dict


class Tokenizer:
    def __init__(self, tokenizer: BertTokenizer, context_length: int) -> None:
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x, max_length=self.context_length, truncation=True, padding="max_length", return_tensors="pt"
        )

    def decode(self, x: Dict[str, torch.LongTensor]):
        return [self.tokenizer.decode(sentence[:sentence_len]) for sentence, sentence_len in
                zip(x["input_ids"], target["attention_mask"].sum(axis=-1))]