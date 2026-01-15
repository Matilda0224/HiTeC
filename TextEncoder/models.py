import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import List
from data.loader import DatasetLoader,load_splits
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from augment.augment import get_all_one_hop_neighbors, generate_prompt_for_node,build_all_prompts,build_negative_texts,build_negative_prompts
import time
import os
import os.path as osp
from evaluation.evaluation import linear_evaluation
import numpy as np
from collections import defaultdict
from torch import Tensor

plm_map = {
    'bert' : 'bert-base-uncased',
    'bert_large':'bert-large-uncased',
    'roberta': 'roberta-base',
    'roberta-large':'roberta-large',
    'distilbert':'distilbert-base-uncased',
    'deberta': 'microsoft/deberta-v3-base'

}
class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert",
        projector_dim: int = 4096,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        training: bool = False,
        pooling: str = "cls", 
    ):
        super().__init__()
        if model_name == 'deberta':
            self.tokenizer = AutoTokenizer.from_pretrained(plm_map[model_name], use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(plm_map[model_name])
        self.encoder = AutoModel.from_pretrained(plm_map[model_name])
        self.training = training
        self.pooling = pooling

        if use_lora:
            if model_name in ['deberta']:
                target_modules = ["query_proj", "key_proj", "value_proj"] 
            elif model_name in ['distilbert']:
                target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
            else:
                target_modules = ["query", "key"]

            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                # inference_mode=not train_encoder,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules = target_modules,
                bias="none"
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
      
        for param in self.encoder.parameters():
            param.requires_grad = False


        encoder_hidden = self.encoder.config.hidden_size
        
        self.projector = nn.Sequential(
            nn.Linear(encoder_hidden, projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, encoder_hidden),
        )
        self.print_num_parameters()
        
    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "cls":
            emb = outputs.last_hidden_state[:, 0] 
        elif self.pooling == "mean":
            last_hidden = outputs.last_hidden_state  # [B, L, D]
            mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
            summed = (last_hidden * mask).sum(1)
            lengths = mask.sum(1)  # [B, 1]
            emb = summed / lengths.clamp(min=1e-9)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        projected = self.projector(emb)

        return projected

    def encode(self, texts: List[str], batch_size=32, max_length=512) -> torch.Tensor:

        self.eval()
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                outputs = self.forward(inputs['input_ids'], inputs['attention_mask'])
                all_embeddings.append(outputs.cpu())
        return torch.cat(all_embeddings, dim=0)
    
    def print_num_parameters(self):
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        encoder_total = sum(p.numel() for p in self.encoder.parameters())

        projector_trainable = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        projector_total = sum(p.numel() for p in self.projector.parameters())

        total_trainable = encoder_trainable + projector_trainable
        total_params = encoder_total + projector_total

        print(f"[Encoder]    Trainable: {encoder_trainable:,} / {encoder_total:,} ({encoder_trainable / encoder_total:.2%})")
        print(f"[Projector]  Trainable: {projector_trainable:,} / {projector_total:,} ({projector_trainable / projector_total:.2%})")
        print(f"[Total]      Trainable: {total_trainable:,} / {total_params:,} ({total_trainable / total_params:.2%})")



