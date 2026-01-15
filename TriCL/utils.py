import random
from itertools import permutations

import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add
import torch.nn.functional as F
from data.loader import DatasetLoader
from collections import defaultdict
from typing import List

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def drop_features(x: Tensor, p: float):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]


def drop_incidence(hyperedge_index: Tensor, p: float = 0.2):
    if p == 0.0:
        return hyperedge_index
    
    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p
    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)
    return hyperedge_index


def drop_nodes(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_nodes, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[drop_idx, :] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def drop_hyperedges(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_edges, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[:, drop_idx] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def valid_node_edge_mask(hyperedge_index: Tensor, num_nodes: int, num_edges: int):
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1])
    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def common_node_edge_mask(hyperedge_indexs: list[Tensor], num_nodes: int, num_edges: int):
    hyperedge_weight = hyperedge_indexs[0].new_ones(num_edges)
    node_mask = hyperedge_indexs[0].new_ones((num_nodes,)).to(torch.bool)
    edge_mask = hyperedge_indexs[0].new_ones((num_edges,)).to(torch.bool)

    for index in hyperedge_indexs:
        Dn = scatter_add(hyperedge_weight[index[1]], index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(index.new_ones(index.shape[1]), index[1], dim=0, dim_size=num_edges)
        node_mask &= Dn != 0
        edge_mask &= De != 0
    return node_mask, edge_mask


def hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask, edge_mask):
    if node_mask is None and edge_mask is None:
        return hyperedge_index

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = H[node_mask].to_sparse().indices()
    return masked_hyperedge_index


def clique_expansion(hyperedge_index: Tensor):
    edge_set = set(hyperedge_index[1].tolist())
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)
    
    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    return adjacency_matrix.to(hyperedge_index.device)

def semantic_filter_hyperedge_index(
    hyperedge_index: torch.Tensor,   # [2, E]
    x: torch.Tensor,                 # [N, F]
    similarity_threshold: float,
    num_nodes: int,
    num_edges: int,
    device: torch.device = torch.device('cpu')
) -> (torch.Tensor, dict):

    edge2nodes = {eid: [] for eid in range(num_edges)}
    row, col = hyperedge_index  # row: node_id, col: edge_id
    for nid, eid in zip(row.tolist(), col.tolist()):
        if eid < num_edges: 
            edge2nodes[eid].append(nid) 

    filtered_pairs = []
    filtered_hypergraph = {}

    connected_nodes = set()

    for eid, node_list in edge2nodes.items():
        if len(node_list) == 0:
            continue
        node_feats = x[node_list]   # shape: [k, F]
        center_feat = node_feats.mean(dim=0, keepdim=True) 
        sims = F.cosine_similarity(node_feats, center_feat, dim=-1)  # [k]
        
        for i, nid in enumerate(node_list):
            if sims[i] >= similarity_threshold:
                filtered_pairs.append((nid, eid))
                connected_nodes.add(nid)
                if eid not in filtered_hypergraph:
                    filtered_hypergraph[eid] = []
                filtered_hypergraph[eid].append(nid)

    all_nodes = set(range(num_nodes))
    missing_nodes = all_nodes - connected_nodes

@torch.no_grad()
def get_semantic_score(
    hypergraph: dict,
    node_emb: Tensor,           # [num_nodes, hidden_dim]
    num_hyperedges: int
) -> Tensor:
    semantic_scores = torch.zeros(num_hyperedges, device=node_emb.device)

    for he_id, nodes in hypergraph.items():
        if len(nodes) < 2:
            semantic_scores[he_id] = 0.0
            continue
        
        node_indices = torch.tensor(nodes, device=node_emb.device)
        emb = node_emb[node_indices]  # [k, D]

        emb = F.normalize(emb, p=2, dim=1)  # [k, D]
        sim_matrix = torch.matmul(emb, emb.T)  # [k, k]
        
        k = len(nodes)
        mask = torch.triu(torch.ones(k, k, device=node_emb.device), diagonal=1).bool()
        sim_scores = sim_matrix[mask]  # shape: [k*(k-1)/2]

        if sim_scores.numel() > 0:
            score = sim_scores.mean()
        else:
            score = 0.0

        semantic_scores[he_id] = score

    return semantic_scores 

def drop_incidence_with_semantic_score(
    hyperedge_index: Tensor,            
    semantic_scores: Tensor,             
    drop_rate: float = 0.2,             
    temperature: float = 0.5            
) -> Tensor:

    row, col = hyperedge_index  
    device = hyperedge_index.device

    keep_probs = torch.sigmoid((semantic_scores - 0.5) / temperature) 

    edge_probs = keep_probs[col]  # [E]

    rand_vals = torch.rand(edge_probs.size(), device=device)
    keep_mask = rand_vals < (edge_probs * (1 - drop_rate))

    new_row = row[keep_mask]
    new_col = col[keep_mask]

    return torch.stack([new_row, new_col], dim=0)  # [2, E']



def drop_incidence_with_structure_score(hyperedge_index: torch.Tensor, drop_rate: float = 0.2) -> torch.Tensor:
    row, col = hyperedge_index  # row: node_id, col: hyperedge_id
    device = hyperedge_index.device
    num_nodes = int(row.max()) + 1
    num_edges = int(col.max()) + 1
    num_connections = row.size(0)

    node_degree = torch.bincount(row, minlength=num_nodes).float()   # shape: [num_nodes]
    edge_degree = torch.bincount(col, minlength=num_edges).float()  # shape: [num_hyperedges]

    node_deg = node_degree[row]   # shape: [E]
    edge_deg = edge_degree[col]   # shape: [E]

    node_keep_score = 1.0 - node_deg / (node_deg.max() + 1e-6)     # 度越高，越接近0（不容易drop）
    edge_keep_score = 1.0 - edge_deg / (edge_deg.max() + 1e-6)

    structural_drop_prob = (node_keep_score + edge_keep_score) / 2  # 越大表示越容易被drop

    final_keep_prob = 1.0 - structural_drop_prob * drop_rate

    rand_vals = torch.rand(num_connections, device=device)
    keep_mask = rand_vals < final_keep_prob

    new_row = row[keep_mask]
    new_col = col[keep_mask]

    return torch.stack([new_row, new_col], dim=0)

def sample_subhypergraph_swalk(hyperedge_index, x, walk_len=4, restart_prob=0.3, s=2):

    edge2nodes = defaultdict(set) 
    node2edges = defaultdict(set)

    for i in range(hyperedge_index.shape[1]):
        nid = hyperedge_index[0, i].item()
        eid = hyperedge_index[1, i].item()
        edge2nodes[eid].add(nid)
        node2edges[nid].add(eid)

    num_nodes = x.size(0)
    subgraph_list = []

    for anchor in range(num_nodes):
        visited_nodes = set([anchor])
        visited_edges = set()
        current_node = anchor
        prev_edge = None 

        # path = f'sart: node {anchor}'
        for _ in range(walk_len):
            if random.random() < restart_prob:
                current_node = anchor
                prev_edge = None

            candidate_edges = list(node2edges.get(current_node, []))
            if prev_edge is not None:
                prev_nodes = edge2nodes[prev_edge]
                candidate_edges = [e for e in candidate_edges if len(prev_nodes & edge2nodes[e]) >= s] 
          
            if not candidate_edges:
                break
            edge = random.choice(candidate_edges) 
            visited_edges.add(edge)
            prev_edge = edge

            node_candidates = list(edge2nodes[edge]) 
            current_node = random.choice(node_candidates)
            visited_nodes.add(current_node)


        if len(visited_edges) == 0 or len(visited_nodes) <= 1:
            x_sub = x[anchor].unsqueeze(0)
            sub_hyperedge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            sub_nodes = sorted(list(visited_nodes))
            sub_edges = sorted(list(visited_edges))
            node_map = {nid: i for i, nid in enumerate(sub_nodes)}
            edge_map = {eid: i for i, eid in enumerate(sub_edges)}

            sub_hyperedge_index = []
            for eid in sub_edges:
                for nid in edge2nodes[eid]:
                    if nid in node_map:
                        sub_hyperedge_index.append([node_map[nid], edge_map[eid]])

            sub_hyperedge_index = (
                torch.tensor(sub_hyperedge_index).T
                if sub_hyperedge_index else torch.empty((2, 0), dtype=torch.long)
            )
            x_sub = x[sub_nodes]

        subgraph_list.append({
            'x_sub': x_sub,
            'hyperedge_index': sub_hyperedge_index
        })

    return subgraph_list

def sample_subhypergraph_rw(
    hyperedge_index: torch.Tensor,     # [2, E]
    x: torch.Tensor,                   # [N, F]
    walk_len: int = 4,
    restart_prob: float = 0.3,
    important_nodes: List[int] = None 
):

    edge2nodes = defaultdict(set)
    node2edges = defaultdict(set)

    for i in range(hyperedge_index.shape[1]):
        nid = hyperedge_index[0, i].item()
        eid = hyperedge_index[1, i].item()
        edge2nodes[eid].add(nid)
        node2edges[nid].add(eid)

    num_nodes = x.size(0)
    if important_nodes is None:
        important_nodes = list(range(num_nodes)) 

    subgraph_list = []

    for anchor in important_nodes:
        visited_nodes = set([anchor])
        visited_edges = set()
        current_node = anchor

        for _ in range(walk_len):
            if random.random() < restart_prob:
                current_node = anchor

            edge_candidates = list(node2edges.get(current_node, []))
            if not edge_candidates:
                break
            edge = random.choice(edge_candidates)
            visited_edges.add(edge)

            node_candidates = list(edge2nodes.get(edge, []))
            if not node_candidates:
                break
            current_node = random.choice(node_candidates)
            visited_nodes.add(current_node)

        if len(visited_edges) == 0 or len(visited_nodes) <= 1:
            x_sub = x[anchor].unsqueeze(0)
            sub_hyperedge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            sub_nodes = sorted(list(visited_nodes))
            sub_edges = sorted(list(visited_edges))
            node_map = {nid: i for i, nid in enumerate(sub_nodes)}
            edge_map = {eid: i for i, eid in enumerate(sub_edges)}

            sub_hyperedge_index = []
            for eid in sub_edges:
                for nid in edge2nodes[eid]:
                    if nid in node_map:
                        sub_hyperedge_index.append([node_map[nid], edge_map[eid]])

            sub_hyperedge_index = (
                torch.tensor(sub_hyperedge_index).T
                if sub_hyperedge_index else torch.empty((2, 0), dtype=torch.long)
            )
            x_sub = x[sub_nodes]

        subgraph_list.append({
            'x_sub': x_sub,
            'hyperedge_index': sub_hyperedge_index
        })

    return subgraph_list
    
# def sample_important_subhypergraph_swalk(
#     hyperedge_index: torch.Tensor,
#     x: torch.Tensor,
#     important_nodes: list,
#     walk_len: int = 4,
#     restart_prob: float = 0.3,
#     s: int = 2
# ):
#     device = x.device

#     edge2nodes = defaultdict(set)
#     node2edges = defaultdict(set)

#     for i in range(hyperedge_index.shape[1]):
#         nid = hyperedge_index[0, i].item()
#         eid = hyperedge_index[1, i].item()
#         edge2nodes[eid].add(nid)
#         node2edges[nid].add(eid)

#     subgraph_list = []

#     for anchor in important_nodes:  
#         visited_nodes = set([anchor])
#         visited_edges = set()
#         current_node = anchor
#         prev_edge = None

#         for _ in range(walk_len):
#             if random.random() < restart_prob:
#                 current_node = anchor
#                 prev_edge = None

#             candidate_edges = list(node2edges.get(current_node, []))
#             if prev_edge is not None:
#                 prev_nodes = edge2nodes[prev_edge]
#                 candidate_edges = [
#                     e for e in candidate_edges
#                     if len(prev_nodes & edge2nodes[e]) >= s
#                 ]

#             if not candidate_edges:
#                 break

#             edge = random.choice(candidate_edges)
#             visited_edges.add(edge)
#             prev_edge = edge

#             node_candidates = list(edge2nodes[edge])
#             current_node = random.choice(node_candidates)
#             visited_nodes.add(current_node)

#         if len(visited_edges) == 0 or len(visited_nodes) <= 1:
#             x_sub = x[anchor].unsqueeze(0).to(device)
#             sub_hyperedge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
#             center_id = 0
#         else:
#             sub_nodes = sorted(list(visited_nodes))
#             sub_edges = sorted(list(visited_edges))
#             node_map = {nid: i for i, nid in enumerate(sub_nodes)}
#             edge_map = {eid: i for i, eid in enumerate(sub_edges)}

#             sub_hyperedge_index = []
#             for eid in sub_edges:
#                 for nid in edge2nodes[eid]:
#                     if nid in node_map:
#                         sub_hyperedge_index.append([node_map[nid], edge_map[eid]])

#             sub_hyperedge_index = (
#                 torch.tensor(sub_hyperedge_index, dtype=torch.long, device=device).T
#                 if sub_hyperedge_index else torch.empty((2, 0), dtype=torch.long, device=device)
#             )
#             x_sub = x[sub_nodes].to(device)
#             center_id = node_map[anchor]

#         subgraph_list.append({
#             'x_sub': x_sub,
#             'hyperedge_index': sub_hyperedge_index,
#             'center_id': center_id
#         })

#     return subgraph_list
import torch
import random
from collections import defaultdict, deque
from typing import List, Dict, Union

def sample_important_subhypergraph_swalk(
    hyperedge_index: torch.Tensor,   # [2, E_inc], (global_node_id, global_edge_id)
    x: torch.Tensor,                 # [N, F] 仅用于确定 N 和 device
    important_nodes: Union[List[int], torch.Tensor],
    walk_len: int = 4,
    restart_prob: float = 0.3,
    s: int = 2
) -> List[Dict]:
    """
    仅为 important_nodes 构建基于 s-walk 的子图。
    返回的每个子图均使用原图的全局索引，便于后续读出：
      - 'node_idx': LongTensor [n_sub]         子图节点的全局ID（升序）
      - 'edge_idx': LongTensor [e_sub]         子图超边的全局ID（升序）
      - 'hyperedge_index': LongTensor [2, E_sub]  保留在子图内的 (global_node_id, global_edge_id) 对
      - 'center_local_id': int                 锚点在 node_idx 排序后的位置
    """
    device = x.device
    N = int(x.size(0))

    # 支持 tensor/list 的 important_nodes
    if isinstance(important_nodes, torch.Tensor):
        important_nodes = important_nodes.tolist()

    # —— 构建全局 incidence 的辅助映射 ——
    edge2nodes = defaultdict(set)   # eid -> {nid}
    node2edges = defaultdict(set)   # nid -> {eid}
    E_inc = hyperedge_index.size(1)
    for i in range(E_inc):
        nid = int(hyperedge_index[0, i].item())
        eid = int(hyperedge_index[1, i].item())
        edge2nodes[eid].add(nid)
        node2edges[nid].add(eid)

    subgraph_list: List[Dict] = []

    for anchor in important_nodes:
        if anchor < 0 or anchor >= N:
            # 跳过非法锚点
            continue

        visited_nodes = set([anchor])
        visited_edges = set()
        current_node = anchor
        prev_edge = None

        for _ in range(walk_len):
            # 随机重启
            if random.random() < restart_prob:
                current_node = anchor
                prev_edge = None

            # 当前节点的候选超边
            cand_edges = list(node2edges.get(current_node, []))
            if prev_edge is not None:
                prev_nodes = edge2nodes[prev_edge]
                cand_edges = [e for e in cand_edges
                              if len(prev_nodes & edge2nodes[e]) >= s]

            if not cand_edges:
                break

            # 选择下一条超边
            edge = random.choice(cand_edges)
            visited_edges.add(edge)
            prev_edge = edge

            # 在该超边内随机跳转一个节点
            node_candidates = list(edge2nodes[edge])
            current_node = random.choice(node_candidates)
            visited_nodes.add(current_node)

        # —— 组装子图（全局索引） ——
        if len(visited_nodes) == 0:
            # 理论上不会发生；安全兜底
            node_idx = torch.tensor([anchor], dtype=torch.long, device=device)
            edge_idx = torch.empty(0, dtype=torch.long, device=device)
            he_sub_glob = torch.empty(2, 0, dtype=torch.long, device=device)
            center_local_id = 0
        else:
            node_idx = torch.tensor(sorted(visited_nodes), dtype=torch.long, device=device)
            edge_idx = torch.tensor(sorted(visited_edges), dtype=torch.long, device=device)
            node_set = set(node_idx.tolist())
            edge_set = set(edge_idx.tolist())

            # 保留 (node, edge) 同时在子图集合中的入射对（使用全局ID）
            pairs = []
            for eid in edge_set:
                for nid in edge2nodes[eid]:
                    if nid in node_set:
                        pairs.append([nid, eid])
            he_sub_glob = (torch.tensor(pairs, dtype=torch.long, device=device).T
                           if pairs else torch.empty(2, 0, dtype=torch.long, device=device))

            # 锚点在子图中的局部位置
            lookup = {n: i for i, n in enumerate(node_idx.tolist())}
            center_local_id = int(lookup.get(anchor, 0))

        subgraph_list.append({
            "node_idx": node_idx,                 # [n_sub]
            "edge_idx": edge_idx,                 # [e_sub]
            "hyperedge_index": he_sub_glob,       # [2, E_sub] (global ids)
            "center_local_id": center_local_id,   # int
        })

    return subgraph_list


def select_important_nodes(
    hyperedge_index: torch.Tensor,
    text_emb: torch.Tensor,         
    keep_rate: float = 0.3,           
) -> torch.Tensor:
    
    num_nodes = text_emb.size(0)
    device = text_emb.device

    node_degree = torch.zeros(num_nodes, device=device)
    node_degree.scatter_add_(0, hyperedge_index[0], torch.ones_like(hyperedge_index[0], dtype=torch.float))

    norm_degree = (node_degree - node_degree.min()) / (node_degree.max() - node_degree.min() + 1e-8) #degree归一化到0-1区间

    importance_score = norm_degree

    topk = int(keep_rate * num_nodes)
    important_nodes = torch.topk(importance_score, k=topk).indices  # [topk]

    return important_nodes
