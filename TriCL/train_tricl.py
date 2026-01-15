import argparse
import random

import yaml
from tqdm import tqdm
import numpy as np
import torch

from data.loader import DatasetLoader
from TriCL.models import HyperEncoder, TriCL
# from TriCL.utils import drop_features, drop_incidence, valid_node_edge_mask, hyperedge_index_masking
from TriCL.utils import drop_features, drop_incidence, valid_node_edge_mask, hyperedge_index_masking, drop_incidence_with_semantic_score,sample_subhypergraph_rw,sample_subhypergraph_swalk,sample_important_subhypergraph_swalk, select_important_nodes,drop_incidence_with_structure_score
from evaluation.evaluation import linear_evaluation
from collections import defaultdict, deque
from typing import List, Dict

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# def encode_subgraphs(subgraph_list, model,alpha=0.5, max_hop=5):
#     subgraph_embs = []
#     device = next(model.parameters()).device

#     for g in subgraph_list:
#         x_sub = g['x_sub'].to(device)
#         he_sub = g['hyperedge_index'].to(device)
#         center_id = g.get('center_id', 0)

#         node_emb, edge_emb = model(x_sub, he_sub)  # [n_sub, D]
#         n = model.node_projection(node_emb)         # [n_sub, D]
#         e = model.edge_projection(edge_emb)         # [e_sub, D]

#         z_subgraph = n.mean(dim=0)
#         subgraph_embs.append(z_subgraph)

#     return torch.stack(subgraph_embs, dim=0)
@torch.no_grad()
def encode_subgraphs(
    subgraph_list: List[Dict],
    x_global: torch.Tensor,           # [N, D] 全图节点嵌入 X^(ℓ)
    e_global: torch.Tensor,           # [M, D] 全图超边嵌入 Z^(ℓ)
    aggr: str = 'node',               # 'node'|'edge'|'mean'|'hop_weighted'
    alpha: float = 0.5,               # node/edge 融合权重（用于 'mean' 或 hop 融合）
    max_hop: int = 5                  # hop_weighted 的最大 hop 截断
) -> torch.Tensor:
    """
    每个子图用 indices 在 (x_global, e_global) 中索引并读出，不再跑 HNN。

    subgraph_list 中每个元素应包含：
      - 'node_idx': LongTensor [n_sub]     子图节点的全局索引
      - 'edge_idx': LongTensor [e_sub]     子图超边的全局索引
      - 'hyperedge_index': LongTensor [2, E_sub]  子图局部入射(节点局部id, 超边局部id)
      - 'center_local_id': int (可选)      锚点在子图局部节点的索引（用于 hop_weighted）
    """
    device = x_global.device
    D = x_global.size(1)
    out = []

    for g in subgraph_list:
        node_idx: torch.LongTensor = g['node_idx'].to(device)          # [n_sub]
        edge_idx: torch.LongTensor = g['edge_idx'].to(device)          # [e_sub]
        he_sub:   torch.LongTensor = g['hyperedge_index'].to(device)   # [2, E_sub] (local ids)

        # 取子图的节点/超边嵌入（局部顺序）
        n = x_global.index_select(0, node_idx)  # [n_sub, D]
        e = e_global.index_select(0, edge_idx)  # [e_sub, D]

        if aggr == 'node':
            z_sub = n.mean(dim=0) if n.numel() > 0 else torch.zeros(D, device=device, dtype=x_global.dtype)

        elif aggr == 'edge':
            z_sub = e.mean(dim=0) if e.numel() > 0 else torch.zeros(D, device=device, dtype=x_global.dtype)

        elif aggr == 'mean':
            z_node = n.mean(dim=0) if n.numel() > 0 else torch.zeros(D, device=device, dtype=x_global.dtype)
            z_edge = e.mean(dim=0) if e.numel() > 0 else torch.zeros(D, device=device, dtype=x_global.dtype)
            z_sub  = alpha * z_node + (1 - alpha) * z_edge

        elif aggr == 'hop_weighted':
            # --- 基于子图局部结构的 hop 加权（锚点在局部 id 空间） ---
            center_local = int(g.get('center_local_id', 0))
            n_sub = node_idx.numel()
            if n_sub == 0:
                z_node = torch.zeros(D, device=device, dtype=x_global.dtype)
            else:
                # 1) 建局部节点邻接（同超边两两连）
                adj = defaultdict(set)
                # he_sub: [2, E_sub]  rows = (local_node_id, local_edge_id)
                # 为每个局部超边收集节点
                # 先把每个 edge_local -> 节点列表
                edge2nodes = defaultdict(list)
                for i in range(he_sub.shape[1]):
                    v_loc = int(he_sub[0, i].item())
                    e_loc = int(he_sub[1, i].item())
                    edge2nodes[e_loc].append(v_loc)
                for nodes in edge2nodes.values():
                    for u in nodes:
                        for v in nodes:
                            if u != v:
                                adj[u].add(v)

                # 2) BFS 计算 hop 距离
                INF = 10**9
                hop = [INF] * n_sub
                if 0 <= center_local < n_sub:
                    hop[center_local] = 0
                    q = deque([center_local])
                    while q:
                        cur = q.popleft()
                        for nb in adj[cur]:
                            if hop[nb] == INF:
                                hop[nb] = hop[cur] + 1
                                if hop[nb] < max_hop:
                                    q.append(nb)

                # 3) 生成权重并归一化（越远权重越小）
                hop_t = torch.tensor(hop, device=device, dtype=n.dtype)
                hop_t[hop_t == INF] = max_hop + 1
                w = 1.0 / (hop_t + 1.0)                          # [n_sub]
                w = w / w.sum().clamp_min(1.0)
                z_node = torch.matmul(w.unsqueeze(0), n).squeeze(0)  # [D]

            z_edge = e.mean(dim=0) if e.numel() > 0 else torch.zeros(D, device=device, dtype=x_global.dtype)
            z_sub  = alpha * z_node + (1 - alpha) * z_edge

        else:
            raise ValueError(f"Unsupported aggregation mode: {aggr}")

        out.append(z_sub)

    return torch.stack(out, dim=0) if out else torch.empty(0, D, device=device, dtype=x_global.dtype)

# def train_tricl(data, model,optimizer, model_type,params, num_negs):
def train_tricl(data, model,optimizer, model_type,params, hyperedge_index1, hyperedge_index2, subgraph1, subgraph2, num_negs):
    hyperedge_index = data.hyperedge_index # feature来自prompt_text
    num_nodes, num_edges = data.num_nodes, data.num_edges
    # prompts_emb = data.features
    prompts_emb = data.prompts_emb
    raw_emb = data.raw_emb
    # neg_prompts_emb = data.neg_prompts_emb 
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Hypergraph Augmentation
    # hyperedge_index1 = drop_incidence_with_semantic_score(hyperedge_index, data.prompts_score, params['drop_incidence_rate'],0.5)
    # hyperedge_index2 = drop_incidence_with_semantic_score(hyperedge_index, data.raw_score, params['drop_incidence_rate'],0.5)


    # x1 = drop_features(prompts_emb, params['drop_feature_rate'])
    # x2 = drop_features(raw_emb, params['drop_feature_rate'])
    n1, e1 = model(prompts_emb, hyperedge_index1, num_nodes, num_edges) 
    n2, e2 = model(raw_emb, hyperedge_index2, num_nodes, num_edges)
    # 为v1 v2构造子图
    # if params['sample_method']:
    #     imp_n1 = select_important_nodes(hyperedge_index1, prompts_emb, params['sub_rate'])
    #     imp_n2 = select_important_nodes(hyperedge_index2, raw_emb, params['sub_rate'])
    
    #     imp_set1 = set(imp_n1.tolist())
    #     imp_set2 = set(imp_n2.tolist())
    #     shared_nodes = sorted(list(imp_set1 & imp_set2))  # 排序保证顺序一致
        
    #     subgraph1 = sample_important_subhypergraph_swalk(hyperedge_index=hyperedge_index1, important_nodes= shared_nodes, x=prompts_emb,walk_len=params['n_walk'],restart_prob=0.3,s=params['s_walk'])
    #     subgraph2 = sample_important_subhypergraph_swalk(hyperedge_index=hyperedge_index2, important_nodes= shared_nodes, x=raw_emb,walk_len=params['n_walk'],restart_prob=0.3,s=params['s_walk'])
    #     subgraph_embs_v1 = encode_subgraphs(subgraph1, model) 
    #     subgraph_embs_v2 = encode_subgraphs(subgraph2, model)
        
    # if params['sample_method'] == 'rw':
    #     subgraph1 = sample_subhypergraph_rw( hyperedge_index1, prompts_emb, walk_len=params['n_walk'],important_nodes=shared_nodes)
    #     subgraph2 = sample_subhypergraph_rw( hyperedge_index2, raw_emb, walk_len=params['n_walk'],important_nodes=shared_nodes)

    # elif params['sample_method'] == 's_walk':
    #     subgraph1 = sample_subhypergraph_swalk(hyperedge_index=hyperedge_index1,x=prompts_emb,walk_len=params['n_walk'],restart_prob=0.3,s=params['s_walk'])
    #     subgraph2 = sample_subhypergraph_swalk(hyperedge_index=hyperedge_index2,x=raw_emb,walk_len=params['n_walk'],restart_prob=0.3,s=params['s_walk'])

    # elif params['sample_method'] == 'important':
    #     subgraph1 = sample_important_subhypergraph_swalk(hyperedge_index=hyperedge_index1, important_nodes= shared_nodes, x=prompts_emb,walk_len=params['n_walk'],restart_prob=0.3,s=params['s_walk'])
    #     subgraph2 = sample_important_subhypergraph_swalk(hyperedge_index=hyperedge_index2, important_nodes= shared_nodes, x=raw_emb,walk_len=params['n_walk'],restart_prob=0.3,s=params['s_walk'])

    # else:
    #     subgraph1 = None
    #     subgraph2 = None
    if subgraph1 and  subgraph2:
        subgraph_embs_v1 = encode_subgraphs(subgraph1, n1, e1) 
        subgraph_embs_v2 = encode_subgraphs(subgraph2, n2, e2)

    else:
        subgraph_embs_v1 = torch.empty(0)
        subgraph_embs_v2 = torch.empty(0)


    node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
    node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
    node_mask = node_mask1 & node_mask2
    edge_mask = edge_mask1 & edge_mask2

    # Encoder
    # n1, e1 = model(prompts_emb, hyperedge_index1, num_nodes, num_edges) 
    # n2, e2 = model(raw_emb, hyperedge_index2, num_nodes, num_edges)
    

    # Projection Head
    # n1, n2 = model.node_projection(n1), model.node_projection(n2)
    # e1, e2 = model.edge_projection(e1), model.edge_projection(e2)

    if model_type in ['tricl_n', 'tricl_ng', 'tricl_ngs']:
        loss_n = model.node_level_loss(n1, n2, params['tau_n'], batch_size=params['batch_size_1'], num_negs=num_negs)
    else:
        loss_n = 0

    if model_type in ['tricl_ng', 'tricl_ngs']:
        loss_g = model.group_level_loss(e1[edge_mask], e2[edge_mask], params['tau_g'], batch_size=params['batch_size_1'], num_negs=num_negs)
    else:
        loss_g = 0

    if model_type in ['tricl_ngs']:
        # imp_n1 = select_important_nodes(hyperedge_index1, prompts_emb, params['sub_rate'])
        # imp_n2 = select_important_nodes(hyperedge_index2, raw_emb, params['sub_rate'])
    
        # imp_set1 = set(imp_n1.tolist())
        # imp_set2 = set(imp_n2.tolist())
        # shared_nodes = sorted(list(imp_set1 & imp_set2))  # 排序保证顺序一致
        
        # subgraph1 = sample_important_subhypergraph_swalk(hyperedge_index=hyperedge_index1, important_nodes= shared_nodes, x=prompts_emb,walk_len=params['n_walk'],restart_prob=0.3,s=params['s_walk'])
        # subgraph2 = sample_important_subhypergraph_swalk(hyperedge_index=hyperedge_index2, important_nodes= shared_nodes, x=raw_emb,walk_len=params['n_walk'],restart_prob=0.3,s=params['s_walk'])
        # subgraph_embs_v1 = encode_subgraphs(subgraph1, model) 
        # subgraph_embs_v2 = encode_subgraphs(subgraph2, model)

        # if subgraph_embs_v1.numel() > 0 and subgraph_embs_v2.numel() > 0:
        loss_sub = model.node_level_loss(subgraph_embs_v1,subgraph_embs_v2, params['tau_sub'], batch_size=params['batch_size_1'], num_negs=num_negs)

    else:
        loss_sub = 0
    

    loss = loss_n + params['w_g'] * loss_g + params['w_sub'] * loss_sub
    loss.backward()
    optimizer.step()

    return loss.item()




