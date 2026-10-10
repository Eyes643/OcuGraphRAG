from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainSample:
    input_text: str
    target_text: str
    evidence_text: str = ""
    citations: str = ""


class QADataset(Dataset):
    def __init__(self, samples: List[TrainSample], tokenizer, max_length: int = 4096):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        full = s.input_text + "\n" + s.target_text
        enc = self.tokenizer(
            full, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        input_ids = enc.input_ids.squeeze(0)
        labels = input_ids.clone()
        prompt_enc = self.tokenizer(
            s.input_text, truncation=True, max_length=self.max_length,
        )
        prompt_len = len(prompt_enc.input_ids)
        labels[:prompt_len] = -100
        return {"input_ids": input_ids, "attention_mask": enc.attention_mask.squeeze(0), "labels": labels}


def sft_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def terminology_loss(logits, labels):
    return sft_loss(logits, labels) * 0.1


def structure_loss(logits, labels):
    return sft_loss(logits, labels) * 0.1


def citation_loss(logits, labels):
    return sft_loss(logits, labels) * 0.1


def consistency_loss(logits, labels):
    return sft_loss(logits, labels) * 0.1


def format_loss(logits, labels):
    return sft_loss(logits, labels) * 0.1


def abstain_loss(logits, labels):
    return sft_loss(logits, labels) * 0.05


def stage1_loss(logits, labels, eta1: float = 0.3, eta2: float = 0.2):
    l_sft = sft_loss(logits, labels)
    l_term = terminology_loss(logits, labels)
    l_struct = structure_loss(logits, labels)
    return l_sft + eta1 * l_term + eta2 * l_struct


def stage2_loss(
    logits, labels,
    lambda1: float = 1.0, lambda2: float = 0.4,
    lambda3: float = 0.3, lambda4: float = 0.2, lambda5: float = 0.1,
):
    l_sft = sft_loss(logits, labels)
    l_cite = citation_loss(logits, labels)
    l_cons = consistency_loss(logits, labels)
    l_fmt = format_loss(logits, labels)
    l_abs = abstain_loss(logits, labels)
    return lambda1 * l_sft + lambda2 * l_cite + lambda3 * l_cons + lambda4 * l_fmt + lambda5 * l_abs


def build_lora_model(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, config)
    return model


def train_stage(
    model,
    tokenizer,
    samples: List[TrainSample],
    loss_fn,
    epochs: int,
    batch_size: int = 16,
    lr: float = 2e-4,
    max_length: int = 4096,
):
    dataset = QADataset(samples, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
    return model


def train_full(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    stage1_samples: List[TrainSample] = None,
    stage2_samples: List[TrainSample] = None,
    cfg: Optional[Dict] = None,
):
    set_seed(cfg.get("seed", 42) if cfg else 42)
    peft_cfg = cfg.get("peft", {}) if cfg else {}
    s1 = peft_cfg.get("stage1", {})
    s2 = peft_cfg.get("stage2", {})

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = build_lora_model(
        base_model,
        peft_cfg.get("lora_rank", 32),
        peft_cfg.get("lora_alpha", 64),
        peft_cfg.get("lora_dropout", 0.05),
    )

    model = train_stage(
        model, tokenizer, stage1_samples or [],
        loss_fn=stage1_loss,
        epochs=s1.get("epochs", 5),
        batch_size=s1.get("batch_size", 16),
        lr=s1.get("learning_rate", 2e-4),
    )

    model = train_stage(
        model, tokenizer, stage2_samples or [],
        loss_fn=stage2_loss,
        epochs=s2.get("epochs", 3),
        batch_size=s2.get("batch_size", 16),
        lr=s2.get("learning_rate", 2e-4),
    )

    return model, tokenizer
