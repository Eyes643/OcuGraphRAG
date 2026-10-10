from __future__ import annotations

from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        max_context: int = 4096,
        max_new_tokens: int = 1024,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are an ophthalmic guideline expert."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_context,
        ).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_ids = output_ids[0][len(inputs.input_ids[0]):]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)
