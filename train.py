from TriCL.train_tricl import train_tricl
from TriCL.utils import get_semantic_score
from TextEncoder.models import TextEncoder
from TextEncoder.train_textencoder import train_textencoder,  ContrastiveHardTextDataset,train_textencoder_hard_neg
from data.loader import DatasetLoader
from TriCL.models import HyperEncoder, TriCL
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import random
import yaml
from tqdm import tqdm
import numpy as np
from augment.augment import get_all_one_hop_neighbors, build_negative_texts, build_all_prompts
import os
import os.path as osp
from evaluation.evaluation import node_classification_eval, edge_prediction_eval
import time
from typing import List
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
import torch.nn.functional as F
from TriCL.utils import drop_features, drop_incidence, valid_node_edge_mask, hyperedge_index_masking, drop_incidence_with_semantic_score,sample_subhypergraph_rw,sample_subhypergraph_swalk,sample_important_subhypergraph_swalk, select_important_nodes,drop_incidence_with_structure_score
import math

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def encode_and_save(texts, text_encoder, save_path, batch_size=128, max_length = 512,  device="cuda"):

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
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            embeddings = text_encoder(inputs['input_ids'], inputs['attention_mask'])  # [B, D]
            all_embeddings.append(embeddings.cpu())

    node_embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(node_embeddings, save_path)
    return node_embeddings

def semantic_filter_hyperedge_index(
    hyperedge_index: torch.Tensor,   # [2, E]
    x: torch.Tensor,                 # [N, F]
    similarity_threshold: float,
    num_nodes: int,
    num_edges: int,
    device: torch.device = torch.device('cpu')
) -> (torch.Tensor, dict):
    """
    对 hyperedge_index 进行语义过滤，仅保留语义相似的 node-edge 连接，不再添加自环。
    返回:
        - filtered_hyperedge_index: Tensor[2, E_filtered]
        - filtered_hypergraph: dict, key: edge_id, value: list of node_ids (经过过滤保留)
    """
    # 构造 edge2nodes 字典（只使用 edge_id < num_edges 的原始边）
    edge2nodes = {eid: [] for eid in range(num_edges)}
    row, col = hyperedge_index  # row: node_id, col: edge_id
    for nid, eid in zip(row.tolist(), col.tolist()):
        if eid < num_edges:  # 只考虑原始边，不包含自环（假设自环在预处理中已经固定添加且不参与过滤）
            edge2nodes[eid].append(nid) # hypergraph结构 已有

    filtered_pairs = []  # 存放 (node_id, edge_id) 对
    # 同时构造 filtered_hypergraph 字典：key 为 edge_id，value 为经过过滤的节点列表
    filtered_hypergraph = {}

    connected_nodes = set()

    # 对每个 edge 进行语义过滤
    for eid, node_list in edge2nodes.items():
        if len(node_list) == 0:
            continue
        # 提取该边中所有节点的特征，计算超边中心
        node_feats = x[node_list]   # shape: [k, F]
        center_feat = node_feats.mean(dim=0, keepdim=True) # 超边语义中心
        sims = F.cosine_similarity(node_feats, center_feat, dim=-1)  # [k]
        
        # 选出语义相似度大于阈值的连接，并保存至 filtered_pairs 和 filtered_hypergraph
        for i, nid in enumerate(node_list):
            if sims[i] >= similarity_threshold:
                filtered_pairs.append((nid, eid))
                connected_nodes.add(nid)
                # 保存到 filtered_hypergraph 对应的 eid 列表中
                if eid not in filtered_hypergraph:
                    filtered_hypergraph[eid] = []
                filtered_hypergraph[eid].append(nid)

    all_nodes = set(range(num_nodes))
    missing_nodes = all_nodes - connected_nodes

    print(f"[Semantic Filter] Filtered connections retained: {len(filtered_pairs)}")
    print(f"[Semantic Filter] Nodes with no remaining edge: {len(missing_nodes)}")
    print(f"⚠️ Note: Self-loops should already exist in preprocessed hypergraph.")

    # 转为 tensor 格式
    if len(filtered_pairs) > 0:
        f_row = torch.tensor([p[0] for p in filtered_pairs], dtype=torch.long, device=device)
        f_col = torch.tensor([p[1] for p in filtered_pairs], dtype=torch.long, device=device)
        filtered_hyperedge_index = torch.stack([f_row, f_col], dim=0)
    else:
        # 如果 filtered_pairs 为空，则返回一个空 tensor（形状 [2, 0]）
        filtered_hyperedge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    return filtered_hyperedge_index, filtered_hypergraph

if __name__ == '__main__':

    parser = argparse.ArgumentParser('unsupervised learning script.')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'citeseer2', 'history', 'photo','computers','fitness'])
    parser.add_argument('--device', type=int, default=7, choices = [0,1,2,3,4,5,6,7])

    parser.add_argument('--plm_type', type=str, default='bert', choices = ['bert', 'roberta', 'distilbert', 'deberta'])
    parser.add_argument('--num_neighbors', type=int, default=10, help='length of node sequence')
    parser.add_argument('--train_textencoder',  action='store_true')
    parser.add_argument('--pooling', type=str, default='cls', choices = ['cls', 'mean'])
    parser.add_argument('--use_lora',  action='store_true')
    parser.add_argument('--encode_emb',  action='store_true')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--pro_dim',  type=int, default=4096, help='output dim of projector')
    parser.add_argument('--save_model_path', type=str, default='checkpoints/textencoder')
    parser.add_argument('--save_emb_path', type=str, default='emb')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--plm_epochs', type=int, default=5)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    # tricl para
    parser.add_argument('--model_type', type=str, default='tricl_ngs', choices=['tricl_n', 'tricl_ng', 'tricl_ngs'])
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--s_walk', type=int, default=5)
    parser.add_argument('--sub_rate', type=float, default=0.3)
    parser.add_argument('--similarity_threshold', type=float, default=0.9)

    parser.set_defaults(train_textencoder=False) 
    parser.set_defaults(use_lora=True)
    parser.set_defaults(encode_emb=False)
    

    args = parser.parse_args()
    print(args) 
    params = yaml.safe_load(open('config.yaml'))[args.dataset]
    print(params) 
    params['sub_rate'] = args.sub_rate

    # read dataset
    device = f'cuda:{args.device}'
    data = DatasetLoader().load(args.dataset, device)

    raw_texts = data.texts # [L]]
    hyperedge_index = data.hyperedge_index

    edge2nodes = defaultdict(set) # 
    node2edges = defaultdict(set)
    for i in range(hyperedge_index.shape[1]):
        nid = hyperedge_index[0, i].item()
        eid = hyperedge_index[1, i].item()
        edge2nodes[eid].add(nid)
        node2edges[nid].add(eid)
    print(f'get edge2nodes:{len(edge2nodes)} and node2edges:{len(node2edges)}')


    if args.encode_emb:
        if args.train_textencoder:
            textencoder = TextEncoder(model_name=args.plm_type,  use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout = args.lora_dropout, pooling = args.pooling).to(device)
            textencoder.print_num_parameters()

            augment_texts = get_all_one_hop_neighbors(nodes2edge=node2edges, edge2nodes=edge2nodes,  raw_text_list = raw_texts, text_embs = data.features, max_neighbors=5) # [L , ~10]
            print(f"len of raw texts:{len(raw_texts)}, aug_texts:{len(augment_texts)}")
            hard_neg_texts = build_negative_texts(data=data, k=3, num_negatives=3)
            dataset = ContrastiveHardTextDataset(raw_texts, augment_texts, hard_neg_texts, textencoder, max_length=args.max_length, device = device)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) 

            textencoder_loss = train_textencoder_hard_neg(textencoder, dataloader, epochs=args.plm_epochs, lr=2e-5, temperature=0.5) 

            encoder_path = osp.join(args.save_model_path , args.dataset)

        else:
            # load from checkpoint
            textencoder = TextEncoder(model_name=args.plm_type, use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout = args.lora_dropout, pooling = args.pooling).to(device)
            textencoder.print_num_parameters()
            encoder_path = osp.join(args.save_model_path , args.dataset)
            textencoder = load_text_encoder(textencoder,  encoder_path, device=device)
        
        emb_dir = osp.join(args.save_emb_path, args.dataset)
        p_path = osp.join(emb_dir, f'prompts_emb.pt')
        r_path = osp.join(emb_dir, 'raw_emb.pt')
       
        prompts = build_all_prompts(data, args.dataset, node2edges, edge2nodes,args.num_neighbors)
        print(f"len of prompts:{len(prompts)}")

        prompts_emb = encode_and_save(prompts, text_encoder=textencoder, save_path=p_path,max_length=args.max_length, device=device).to(device) # 经过encode和save之后，emb会放到cpu上
        raw_emb = encode_and_save(raw_texts, text_encoder=textencoder, save_path=r_path, max_length=args.max_length, device=device).to(device) 
        
        data.prompts_emb = prompts_emb
        data.raw_emb = raw_emb
    else:
        emb_dir = osp.join(args.save_emb_path, args.dataset)
        p_path = osp.join(emb_dir, f'prompts_emb.pt')
        r_path = osp.join(emb_dir, 'raw_emb.pt')
        
        prompts_emb = torch.load(p_path, weights_only = False).to(device) 
        data.prompts_emb = prompts_emb
        raw_emb = torch.load(r_path, weights_only = False).to(device) 
        data.raw_emb = raw_emb
        print(f'load prompts_emb from {p_path}, and raw_emb from {r_path}.')
        
        data.prompts_score = get_semantic_score(data.hypergraph, data.prompts_emb, data.num_edges)
        data.raw_score = get_semantic_score(data.hypergraph, data.raw_emb, data.num_edges)
        print('semantic-score load')

        node_accs = []
        edge_accs = []
        for seed in range(args.num_seeds):
            fix_seed(seed)
            
            # 增强应该都预先计算
            hyperedge_index1 = drop_incidence_with_semantic_score(hyperedge_index, data.prompts_score, params['drop_incidence_rate'],0.5)
            hyperedge_index2 = drop_incidence_with_semantic_score(hyperedge_index, data.raw_score, params['drop_incidence_rate'],0.5)
            
            imp_n1 = select_important_nodes(hyperedge_index1, prompts_emb, params['sub_rate'])
            imp_n2 = select_important_nodes(hyperedge_index2, raw_emb, params['sub_rate'])
    
            imp_set1 = set(imp_n1.tolist())
            imp_set2 = set(imp_n2.tolist())
            shared_nodes = sorted(list(imp_set1 & imp_set2))  # 排序保证顺序一致

            subgraph1 = sample_important_subhypergraph_swalk(hyperedge_index=hyperedge_index1, important_nodes= shared_nodes, x=prompts_emb,walk_len=params['n_walk'],restart_prob=0.3,s=params['s_walk'])
            subgraph2 = sample_important_subhypergraph_swalk(hyperedge_index=hyperedge_index2, important_nodes= shared_nodes, x=raw_emb,walk_len=params['n_walk'],restart_prob=0.3,s=params['s_walk'])

            
            encoder = HyperEncoder(data.prompts_emb.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'])
            model = TriCL(encoder, params['proj_dim']).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            print("[•] Training hnnencoder started.")
            start_time = time.time()  
            # epoch_times = []
            for epoch in tqdm(range(1, params['epochs'] + 1)):
                # encoder_loss = train_tricl(data, model, optimizer, args.model_type, params, num_negs=None)
                t0 = time.time()
                encoder_loss = train_tricl(data, model, optimizer, args.model_type, params, hyperedge_index1, hyperedge_index2,subgraph1,subgraph2, num_negs=None)
                # t1 = time.time()
                # epoch_time = t1 - t0      # seconds
                # epoch_times.append(epoch_time)
                if (epoch % 50 == 0) or (epoch == params['epochs']):
                    print(f"epoch {epoch}: loss {encoder_loss}")
            # avg_epoch_time = np.mean(epoch_times)          # seconds
            # avg_epoch_time_ms = avg_epoch_time * 1000
            # log_epoch_time = math.log(avg_epoch_time)
            # print(
            #     f"[✓] Avg. training time per epoch: "
            #     f"{avg_epoch_time:.4f} s "
            #     f"{avg_epoch_time_ms:.4f} ms "
            # )
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            print(
                f"[✓] All. training time: "
                f"{elapsed_time:.4f} s "
                f"{minutes} min, {seconds}s "
            )
            # print(f"\n[✓] Training hnn completed in {int(minutes)} min {int(seconds)} sec.")
            
            node_acc = node_classification_eval(encoder=model, data=data)
            edge_acc = edge_prediction_eval(encoder=model, data=data, dataset =args.dataset, device=args.device)
            
            node_accs.append(node_acc)
            edge_accs.append(edge_acc)
           

            node_acc_mean, node_acc_std = np.mean(node_acc, axis=0), np.std(node_acc, axis=0)
            edge_acc_mean, edge_acc_std = np.mean(edge_acc, axis=0), np.std(edge_acc, axis=0)
            
            print(f'on seed {seed}, on task [linear_node]:  train_acc: {node_acc_mean[0]:.2f}+-{node_acc_std[0]:.2f}, '
                    f'valid_acc: {node_acc_mean[1]:.2f}+-{node_acc_std[1]:.2f}, test_acc: {node_acc_mean[2]:.2f}+-{node_acc_std[2]:.2f}')
            print(f'on seed {seed}, on task [linear_edge]:  train_acc: {edge_acc_mean[0]:.2f}+-{edge_acc_std[0]:.2f}, '
                    f'valid_acc: {edge_acc_mean[1]:.2f}+-{edge_acc_std[1]:.2f}, test_acc: {edge_acc_mean[2]:.2f}+-{edge_acc_std[2]:.2f}')
        
        node_accs = np.array(node_accs).reshape(-1, 3)
        node_accs_mean = list(np.mean(node_accs, axis=0))
        node_accs_std = list(np.std(node_accs, axis=0))
        print(f'On task:【linear_node】, [Final] dataset: {args.dataset}, test_acc: {node_accs_mean[2]:.2f}+-{node_accs_std[2]:.2f}')
        
        edge_accs = np.array(edge_accs).reshape(-1, 3)
        edge_accs_mean = list(np.mean(edge_accs, axis=0))
        edge_accs_std = list(np.std(edge_accs, axis=0))
        print(f'On task:【linear_edge】, [Final] dataset: {args.dataset}, test_acc: {edge_accs_mean[2]:.2f}+-{edge_accs_std[2]:.2f}')

        