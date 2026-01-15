import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import List
from data.loader import DatasetLoader,load_splits
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from augment.augment import get_all_one_hop_neighbors, generate_prompt_for_node,build_negative_texts
import time
import os
from evaluation.evaluation import linear_evaluation
import numpy as np
from TextEncoder.models import TextEncoder


class ContrastiveHardTextDataset(Dataset):
    def __init__(self, raw_texts, aug_texts_list, nei_text_list, model, tokenizer_name="bert-base-uncased", max_length=512, agg_method="mean", device="cuda"):
        assert len(raw_texts) == len(aug_texts_list)
        self.raw_texts = raw_texts
        self.aug_texts_list = aug_texts_list
        self.nei_texts_list = aug_texts_list
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.agg_method = agg_method.lower()
        self.device = device
        self.model = model.to(device)
        self.model.train()  # <<< 训练模式 ✅

    def __len__(self):
        return len(self.raw_texts)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """
        编码多个文本 -> 聚合后返回，保留梯度。
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(inputs['input_ids'], inputs['attention_mask'])  # [B, H]

        if self.agg_method == "mean":
            return outputs.mean(dim=0)       # 聚合邻居
        elif self.agg_method == "cls":
            return outputs[0]                # 第一个 token 表示
        else:
            raise ValueError("Unknown agg method")

    def __getitem__(self, idx):
        raw_text = self.raw_texts[idx]
        neighbor_texts = self.aug_texts_list[idx]
        nei_texts = self.nei_texts_list[idx]

        z_raw = self.encode_texts([raw_text])         # shape: [H]
        z_aug = self.encode_texts(neighbor_texts)     # shape: [H]
        z_nei = self.encode_texts(nei_texts)
        return {
            "z_raw": z_raw,
            "z_aug": z_aug,
            "z_nei": z_nei,
            "idx": idx
        }

class ContrastiveTextDataset(Dataset):
    def __init__(self, raw_texts, aug_texts_list, model, tokenizer_name="bert-base-uncased", max_length=512, agg_method="mean", device="cuda"):
        assert len(raw_texts) == len(aug_texts_list)
        self.raw_texts = raw_texts
        self.aug_texts_list = aug_texts_list
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.agg_method = agg_method.lower()
        self.device = device
        self.model = model.to(device)
        self.model.train()  # <<< 训练模式 ✅

    def __len__(self):
        return len(self.raw_texts)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """
        编码多个文本 -> 聚合后返回，保留梯度。
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(inputs['input_ids'], inputs['attention_mask'])  # [B, H]

        if self.agg_method == "mean":
            return outputs.mean(dim=0)       # 聚合邻居
        elif self.agg_method == "cls":
            return outputs[0]                # 第一个 token 表示
        else:
            raise ValueError("Unknown agg method")

    def __getitem__(self, idx):
        raw_text = self.raw_texts[idx]
        neighbor_texts = self.aug_texts_list[idx]

        z_raw = self.encode_texts([raw_text])         # shape: [H]
        z_aug = self.encode_texts(neighbor_texts)     # shape: [H]

        return {
            "z_raw": z_raw,
            "z_aug": z_aug,
            "idx": idx
        }

def contrastive_loss_with_hard_negative(z1, z2, temperature=0.5):
    """
    z1: [B, D], z2: [B, D] - 原始和增强视图的 node embedding
    每个正样本对之外，构造一个hard negative（为所有负样本的平均聚合）
    """
    # 1. normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)

    # 2. 正常计算原始的 logit 矩阵（[B, B]）
    logits = torch.mm(z1, z2.T) / temperature

    # 3. 构造 Hard Negative 表示：z2 除去当前索引，再聚合成一个向量（对每个样本）
    hard_negatives = []
    for i in range(batch_size):
        mask = torch.ones(batch_size, dtype=torch.bool, device=z1.device)
        mask[i] = False
        hn = z2[mask].mean(dim=0)  # mean aggregation
        hard_negatives.append(hn)

    # [B, D]
    hard_negatives = torch.stack(hard_negatives, dim=0)

    # 4. 计算 hard negative 的相似度并拼接（[B, B+1]）
    hard_logits = (z1 * hard_negatives).sum(dim=1, keepdim=True) / temperature  # [B, 1]
    logits = torch.cat([logits, hard_logits], dim=1)  # [B, B+1]

    # 5. label 仍然是 diag 的位置 → 即索引位置不变
    labels = torch.arange(batch_size, device=z1.device)

    # 6. 交叉熵
    return F.cross_entropy(logits, labels)

def contrastive_loss(z1, z2, temperature=0.5):
    """
    z1: [B, D], z2: [B, D] → 原始和增强视图的 node embedding
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)
    logits = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(batch_size).to(z1.device)
    return F.cross_entropy(logits, labels)

# 给每个sample + 1 hard_neg
def _loss(z1,z2,z3,temperature, hard:bool):
    if hard:
        # use infonce_with_hard_neg
        loss1 =  infonce_with_hard_neg(z1, z2, z3, temperature)
        loss2 =  infonce_with_hard_neg(z2, z1, z3, temperature)
    else:
        # use contrastive_loss
        loss1 =  contrastive_loss(z1, z2, temperature)
        loss2 =  contrastive_loss(z2, z1, temperature)
    loss = 0.5 * (loss1 + loss2)
    return loss.mean()

def infonce_with_hard_neg(anchor, positive, hard_negative, temperature=0.2):
    """
    anchor: [B, D]
    positive: [B, D]
    hard_negative: [B, D]  # 每个样本一个困难负样本

    return: InfoNCE loss
    """

    B = anchor.size(0)
    # 归一化（推荐）
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    hard_negative = F.normalize(hard_negative, dim=-1)

    # full-batch 正负相似度（anchor vs positive[j]）
    logits_pos = torch.matmul(anchor, positive.T)  # [B, B]
    # 每个 anchor 对应的 hard negative
    logits_hard = torch.sum(anchor * hard_negative, dim=-1, keepdim=True)  # [B, 1]

    # 拼接：[B, B+1]
    logits = torch.cat([logits_pos, logits_hard], dim=1)  # [B, B+1]

    # 所有 anchor 的正样本 index 都在第 i 个位置（即 logits[i][i]）
    labels = torch.arange(B, device=anchor.device)

    # temperature scaling
    logits = logits / temperature

    # CrossEntropy 会自动做 softmax
    loss = F.cross_entropy(logits, labels)

    return loss

def train_textencoder_hard_neg(encoder, dataloader, epochs=5, lr=2e-5, temperature=0.5):
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)

    print("[•] Training textencoder started.")
    start_time = time.time()  # 记录开始时间
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # raw_emb = model(raw_batch)  # shape: [B, D]
            # aug_emb = model(aug_batch)
            # print(f'raw_emb: {raw_emb.shape}, aug_emb: {aug_emb.shape}')
            raw_emb = batch["z_raw"]  # [B, D]
            aug_emb = batch["z_aug"]  # [B, D]
            nei_emb = batch['z_nei']
            # loss = contrastive_loss(aug_emb, nei_emb) # infoNCE
            # loss = triplet_loss(raw_emb, aug_emb, nei_emb)
            # loss = infonce_with_negs(raw_emb, aug_emb, nei_emb)
            loss = infonce_with_hard_neg(raw_emb, aug_emb, nei_emb)
            # loss = _loss(raw_emb, aug_emb, nei_emb,temperature, True) #交换了效果反而差 result save at hard2
            # loss = loss()
            # break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            del loss, raw_emb, aug_emb, nei_emb
            torch.cuda.empty_cache()
      
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\n[✓] Training textencoder completed in {int(minutes)} min {int(seconds)} sec.")
    # 保存 TextEncoder 整体

    return total_loss

def train_textencoder(encoder, dataloader, epochs=5, lr=2e-5, temperature=0.5):
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)

    print("[•] Training textencoder started.")
    start_time = time.time()  # 记录开始时间
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # raw_emb = model(raw_batch)  # shape: [B, D]
            # aug_emb = model(aug_batch)
            # print(f'raw_emb: {raw_emb.shape}, aug_emb: {aug_emb.shape}')
            raw_emb = batch["z_raw"]  # [B, D]
            aug_emb = batch["z_aug"]  # [B, D]
            # loss = contrastive_loss(raw_emb, aug_emb)
            # loss = _loss(raw_emb, aug_emb, None, 0.5, False)
            loss = contrastive_loss_with_hard_negative(raw_emb, aug_emb, 0.5)
            # break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # break
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\n[✓] Training textencoder completed in {int(minutes)} min {int(seconds)} sec.")
    return total_loss

def encode_and_save(texts, text_encoder, batch_size=32, device="cuda"):
    """
    使用训练好的 TextEncoder 对 texts 编码并保存为 .pt 文件
    Args:
        texts (List[str]): 输入的原始文本列表
        text_encoder (nn.Module): 已训练好的文本编码器
        save_path (str): 输出保存路径，如 "node_embeddings.pt"
        batch_size (int): 批处理大小
        device (str): 使用的设备
    """
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = text_encoder.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            embeddings = text_encoder(inputs['input_ids'], inputs['attention_mask'])  # [B, D]
            all_embeddings.append(embeddings.cpu())

    node_embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]

    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # torch.save(node_embeddings, save_path)
    # print(f"[✓] Node embeddings saved to: {save_path}")
    return node_embeddings

# def load_split(seed):
#     split_path = f'../../../../TAHG_Datasets/cora/splits/{seed}.pt'
#     masks = torch.load(split_path)
#     return masks

def node_evaluation(data, node_embedding, device, num_splits = 20):
    if data.name == 'pubmed':
        lr = 0.005
        max_epoch = 300
    elif data.name == 'cora':
        lr = 0.005
        max_epoch = 100
    else:
        lr = 0.005
        max_epoch = 800
    masks_list = load_splits(data.name, num_splits=20, device=device)
    accs = []
    for i in range(int(num_splits)):
        masks = masks_list[i]
        node_embedding = node_embedding.to(device)
        labels = data.labels.to(device)
        acc= linear_evaluation(node_embedding, labels, masks, lr=lr, max_epoch=max_epoch) # list(train, val, test)
        accs.append(acc)
        print(f'on split: {i}, train_acc: {acc[0]:.2f}, ' f'valid_acc: {acc[1]:.2f}, test_acc: {acc[2]:.2f}')
    
    acc_mean, acc_std = np.mean(accs, axis=0), np.std(accs, axis=0)
    # accs_mean = list(np.mean(accs, axis=0))
    # accs_std = list(np.std(accs, axis=0))
    print(f'[Final] test_acc: {acc_mean[2]:.4f}+-{acc_std[2]:.4f}')

def build_all_prompts(
    data,
    dataset_name: str,
    max_neighbors: int = 5,
    prompt_style: str = "full"
):
    """
    遍历超图中的每个节点，构造结构感知+任务感知的 prompt。
    返回一个 list，长度为 num_nodes。
    """
    all_prompts = []
    num_nodes = len(data.texts)

    # 创建邻接表：node -> neighbor nodes
    # node_neighbors = [[] for _ in range(num_nodes)]
    # edge_index = data.edge_index
    # for src, dst in edge_index.T.tolist():
    #     node_neighbors[src].append(dst)
    node_neighbors = [[] for _ in range(num_nodes)]
    for hedge_nodes in data.hypergraph.values():
        for i in range(len(hedge_nodes)):
            center = hedge_nodes[i]
            for j in range(len(hedge_nodes)):
                if i != j:
                    node_neighbors[center].append(hedge_nodes[j])
    # 去重（可选）
    node_neighbors = [list(set(neighs)) for neighs in node_neighbors]

    # 超边 membership（node -> hyperedge id list）
    if hasattr(data, "node2hyper"):
        node2hyper = data.node2hyper
    else:
        # 从 hypergraph 反构建
        node2hyper = {i: [] for i in range(num_nodes)}
        for he_id, node_list in data.hypergraph.items():
            for nid in node_list:
                node2hyper[nid].append(he_id)

    # 遍历每个节点
    for node_id in range(num_nodes):
        central_text = data.texts[node_id]
        neighbors = node_neighbors[node_id]
        neighbor_texts = [data.texts[n] for n in neighbors]
        degree = len(neighbors)
        memberships = node2hyper[node_id]

        prompt = generate_prompt_for_node(
            dataset_name=dataset_name,
            node_text=central_text,
            neighbor_texts=neighbor_texts,
            degree=degree,
            memberships=memberships,
            max_neighbors=max_neighbors,
            prompt_style=prompt_style
        )
        all_prompts.append(prompt)

    return all_prompts


def save_text_encoder(model, save_dir: str):
    """
    保存 TextEncoder 模型，包括：
    - encoder（支持 LoRA）
    - projector
    - tokenizer 配置

    Args:
        model (TextEncoder): 待保存模型
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 保存 projector 权重
    torch.save(model.projector.state_dict(), os.path.join(save_dir, "projector.pt"))

    # 保存 encoder（包括 LoRA adapter）
    model.encoder.save_pretrained(os.path.join(save_dir, "encoder"))

    # 保存 tokenizer 配置
    model.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

    print(f"✅ TextEncoder saved to {save_dir}")

from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig

def load_text_encoder(model_class, load_dir: str, device: str = "cuda"):
    """
    加载 TextEncoder 模型（需传入原始 TextEncoder 类定义）

    Args:
        model_class: TextEncoder 类（注意要和保存时一致）
        load_dir (str): 保存目录
        device (str): 加载设备

    Returns:
        TextEncoder: 加载后的模型
    """
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_dir, "tokenizer"))

    # 加载 encoder
    encoder = AutoModel.from_pretrained(os.path.join(load_dir, "encoder"))

    # 初始化 TextEncoder 模型实例
    model = model_class
    model.tokenizer = tokenizer
    model.encoder = encoder
    model.device = device
    model.to(device)

    # 加载 projector 权重
    projector_path = os.path.join(load_dir, "projector.pt")
    model.projector.load_state_dict(torch.load(projector_path, map_location=device))

    print(f"✅ TextEncoder loaded from {load_dir}")
    return model

if __name__ == '__main__':
    
    model = TextEncoder(train_encoder=True, use_lora=True)
    model.print_num_parameters()
    
    device = 'cuda:7'
    data = DatasetLoader().load('pubmed').to(device)
    raw_texts = data.texts # [L]
    augment_texts = get_all_one_hop_neighbors(hypergraph = data.hypergraph, raw_text_list = raw_texts, max_neighbors=5) # [L , ~10]
    # hard_neg_texts = build_negative_texts(data=data,k=2,num_negatives=5)
    dataset = ContrastiveTextDataset(raw_texts, augment_texts, model, device =device)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # dataset = ContrastiveHardTextDataset(raw_texts, augment_texts, hard_neg_texts, model, device =device)
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    print(f"len of raw text: {len(raw_texts)}, len of augment_texts: {len(augment_texts)}")

    # optimizer = torch.optim.Adam(model.projector.parameters(), lr=1e-4)
    # train model
  

    # train_textencoder(model, dataloader,epochs=5, lr=2e-5, temperature=0.5) # 训练textencoder 
    # train_textencoder_hard_neg(model, dataloader, epochs=5, lr=2e-5, temperature=0.5)
    save_model_path = 'checkpoints/textencoder/pubmed/hard' #2代表加上正反loss
    # save_text_encoder(model, save_model_path)
    model = load_text_encoder(model, save_model_path) # 加载textencoder
    # PLM
    prompts = build_all_prompts(data, 'pubmed', 5, 'full')
    print(f"len of prompts: {len(prompts)}")
    # print(f"******prompt 0 :{prompts[0]}")
    # save_path = f"out/cora/node_emb.pt"
    node_emb = encode_and_save(prompts, text_encoder=model, device=device) #在cpu

    # 评估PLM
    node_evaluation(data, node_emb, device)
 