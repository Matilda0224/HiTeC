import torch
from torch import Tensor
from collections import deque
import random
from typing import Dict, List
from torch_sparse import coalesce
import scipy.sparse as sp
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.loader import DatasetLoader
from collections import defaultdict

def cosine_similarity(a: torch.Tensor, b: torch.Tensor):
    """
    :param a: shape [hidden_dim]
    :param b: shape [hidden_dim]
    :return: float
    """
    sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return sim

def get_one_hop_neighbors(node_id, num_to_edge, edge_to_num, raw_text_list, text_embs, max_neighbors=5):
    neighbors = set()
    for he_id in num_to_edge[node_id]:          
        neighbors.update(edge_to_num[he_id])     

    neighbors = list(neighbors)

    selected = [
            nid for nid in neighbors 
            if cosine_similarity(text_embs[nid], text_embs[node_id]) > 0.9
        ]

    if len(neighbors) >= max_neighbors:
        selected = selected[:max_neighbors]
    else:
        selected = selected + [node_id] * (max_neighbors - len(selected))
    

    return [raw_text_list[nid] for nid in selected]

def get_all_one_hop_neighbors(nodes2edge, edge2nodes,  raw_text_list, text_embs, max_neighbors=10):
    node_sequence_list = []

    for i in range(len(raw_text_list)):
        node_sequence = get_one_hop_neighbors(i, nodes2edge, edge2nodes,  raw_text_list, text_embs, max_neighbors=max_neighbors)
        node_sequence_list.append(node_sequence)

    return node_sequence_list

def generate_prompt_for_node(
    dataset_name: str,
    node_text: str,
    neighbor_texts: list[str],
    degree: int,
    memberships: list[int],
    label_options: list[str] = None,
    max_neighbors: int = 5,
    prompt_style: str = 'full'
) -> str:
    # 1. domain header
    domain_intro = {
        "cocitation": "You are in an academic hypergraph where nodes are papers and hyperedges are sharing cocitation relationship.",
        "books":"You are in a book hypergraph where nodes represent books and their associated texts are book descriptions. Hyperedges are formed by applying a maximal clique algorithm over the original co-purchase or co-viewed relationships between books.",
        "sports":"You are in an sports hypergraph where nodes are ﬁtness-related items. Hyperedges are formed by applying a maximal clique algorithm over the original co-purchase or co-viewed relationships between items.",
        "ecommerce": "You are in an e-commerce hypergraph where nodes represent electronics-related products, each described by a top-voted or representative user review. Hyperedges are constructed using a maximal clique algorithm based on co-purchase and co-viewed relationships among products.",
        "general": "You are in a general hypergraph."
    }
    if dataset_name in ['cora', 'citeseer']:
        domain_info = domain_intro['cocitation']
    elif dataset_name in ['photo', 'computers']:
        domain_info = domain_intro['ecommerce']
    elif dataset_name in ['history']:
        domain_info = domain_intro['books']
    elif dataset_name in ['fitness']:
        domain_info = domain_intro['sports']
    else:
        domain_info = domain_intro['general']
 
    degree_info = f"You has a degree of {degree}, indicating it is {'a highly connected hub' if degree > 5 else 'a sparsely connected node'}."
    if len(memberships) > 0:
        mem_str = " , ".join(map(str, memberships))
    else:
        mem_str = "zero"
    membership_info = f"You belongs to {mem_str} hyperedges."

    used_neighbors = neighbor_texts[:max_neighbors]
    neighbor_str = " . ".join(used_neighbors) if used_neighbors else "No neighbors available."
    neighbor_info = f"The following are brief descriptions of its neighboring nodes: {neighbor_str}"

    label_map = {
        "cora": ["Case_Based", "Genetic_Algorithms", "Neural_Networks","Probabilistic_Methods", "Reinforcement_Learning", "Rule_Learning", "Theory"],
        "citeseer":["Agents", "Machine Learning", "Information Retrieval", "Database", "Human Computer Interaction", "Artificial Intelligence"],
        "history":['World', 'Americas', 'Asia', 'Military', 'Europe', 'Russia', 'Africa', 'Ancient Civilizations', 'Middle East','Historical Study & Educational Resources','Arctic & Antarctica','Negroes'],
        "photo":['Accessories', 'Bags & Cases', 'Binoculars & Scopes', 'Digital Cameras', 'Film Photography', 'Flashes', 'Lenses', 'Lighting & Studio', 'Tripods & Monopods', 'Underwater Photography', 'Video', 'Video Surveillance'],
        "computers":['Computer Accessories & Peripherals', 'Computer Components', 'Computers & Tablets', 'Data Storage', 'Laptop Accessories', 'Monitors', 'Networking Products', 'Servers', 'Tablet Accessories', 'Tablet Replacement Parts'],
        "fitness":['Accessories', 'Airsoft & Paintball', 'Boating & Sailing', 'Clothing', 'Exercise & Fitness', 'Golf', 'Hunting & Fishing', 'Leisure Sports & Game Room', 'Other Sports', 'Sports Medicine', 'Swimming', 'Team Sports', 'Tennis & Racquet Sports']
    }
    label_options = label_map[dataset_name]
    label_str = " , ".join(map(str, label_options))
    task_info = f"Your task is to determine the category of this node based on its content and its structural/semantic context. The categories are {label_str}."
   

    central_info = f"Your text content is:\n\"{node_text}\""

    if prompt_style in ['full', 'full_semantic', 'full_filtered','full_semantic_filtered'] :
        prompt = domain_info + task_info + degree_info + membership_info + central_info + neighbor_info
    elif prompt_style in ['wo_domain', 'wo_domain_semantic_filtered']:
        prompt =  task_info + degree_info + membership_info + central_info + neighbor_info
    elif prompt_style in ['wo_task']:
        prompt = domain_info + degree_info + membership_info + central_info + neighbor_info
    elif prompt_style in ['wo_deg_mem', 'wo_deg_mem_semantic_filtered']:
        prompt = domain_info + task_info + central_info + neighbor_info

    elif prompt_style in ['wo_nei', 'wo_nei_semantic_filtered']:
        prompt = domain_info + task_info + degree_info + membership_info + central_info 
    else:
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")

    return prompt

def build_all_prompts(
    data,
    dataset_name: str,
    node2edges:dict,
    edge2nodes:dict,
    max_neighbors: int = 5,
    prompt_style: str = "full",
    similarity_threshold = 0.9
):
    all_prompts = []
    num_nodes = len(data.texts)
    text_embs = data.features
    filtered_pairs = []  
    
    for node_id in range(num_nodes):
        central_text = data.texts[node_id]
        memberships = list(node2edges[node_id])

        neighbors = set()
        for edge_id in memberships:
            neighbors.update(edge2nodes[edge_id])
        neighbors.discard(node_id) 

        if prompt_style in ['full_semantic', 'full_semantic_filtered', 'wo_domain_semantic_filtered', 'wo_deg_mem_semantic_filtered', 'wo_nei_semantic_filtered']:
            selected = [
                nid for nid in neighbors 
                if cosine_similarity(text_embs[nid], text_embs[node_id]) > similarity_threshold
            ]
          
            selected_set = set(selected)  
        else:
            selected = list(neighbors)

        neighbor_list = selected
        if len(neighbor_list) > max_neighbors:
            neighbor_list = neighbor_list[:max_neighbors]
        else:
            neighbor_list += [node_id] * (max_neighbors - len(neighbor_list))

        neighbor_texts = [data.texts[nid] for nid in neighbor_list]
        degree = len(neighbors)

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


def get_k_hop_neighbors(hypergraph: dict, node_id: int, k: int = 2) -> set:
    visited = set()
    queue = deque([(node_id, 0)])

    while queue:
        curr_node, depth = queue.popleft()
        if depth > k:
            break
        visited.add(curr_node)

        #BFS
        for hedge_nodes in hypergraph.values():
            if curr_node in hedge_nodes:
                for neighbor in hedge_nodes:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
    return visited 

def get_non_k_hop_neighbors_gpu(
    hyperedge_index: Tensor,   # [2, E] (node_id, edge_id)
    num_nodes: int,
    num_edges: int,
    k: int = 2
) -> Dict[int, List[int]]:

    device = hyperedge_index.device

    row, col = hyperedge_index[0], hyperedge_index[1]
    value = torch.ones_like(row, dtype=torch.float)
    H = torch.sparse_coo_tensor(torch.stack([row, col]), value, (num_nodes, num_edges), device=device)

    # A = H @ H.T  → [N, N] adjacency matrix
    A = torch.sparse.mm(H, H.transpose(0, 1)).coalesce()

    mask = A.indices()[0] != A.indices()[1]
    A = torch.sparse_coo_tensor(
        A.indices()[:, mask],
        A.values()[mask],
        A.size(),
        device=device
    ).coalesce()

    Ak = A.clone()
    for _ in range(k - 1):
        Ak = torch.sparse.mm(Ak, A).coalesce()

        mask = Ak.indices()[0] != Ak.indices()[1]
        Ak = torch.sparse_coo_tensor(
            Ak.indices()[:, mask],
            Ak.values()[mask],
            Ak.size(),
            device=device
        ).coalesce()

        Ak = Ak.coalesce()
        Ak_all_indices = torch.cat([A.indices(), Ak.indices()], dim=1)
        Ak_all_values = torch.cat([A.values(), Ak.values()])
        A = torch.sparse_coo_tensor(Ak_all_indices, Ak_all_values, A.size(), device=device).coalesce()

    khop_dict = {i: set() for i in range(num_nodes)}
    row_ids, col_ids = A.coalesce().indices()
    for i, j in zip(row_ids.tolist(), col_ids.tolist()):
        khop_dict[i].add(j)
    for i in range(num_nodes):
        khop_dict[i].add(i)

    non_khop_neighbors = {
        i: list(set(range(num_nodes)) - khop_dict[i])
        for i in range(num_nodes)
    }

    return non_khop_neighbors

def get_non_k_hop_neighbors(
    hyperedge_index: Tensor,   # [2, num_entries]  (node_id, edge_id)
    num_nodes: int,
    num_edges: int,
    k: int = 2
) -> Dict[int, List[int]]:

    row, col = hyperedge_index[0], hyperedge_index[1]
    value = torch.ones(row.size(0))
    row = row.cpu()
    col = col.cpu()
    value = value.cpu()
    H = sp.coo_matrix((value.numpy(), (row.numpy(), col.numpy())), shape=(num_nodes, num_edges))  # N x E

    A = H @ H.T  # [N, N]
    A.setdiag(0)
    A.eliminate_zeros()

    A_k = A.copy()
    Ak = A.copy()
    for _ in range(k - 1):
        Ak = Ak @ A
        Ak.setdiag(0)
        Ak.eliminate_zeros()
        A_k = A_k + Ak

    A_k = A_k.tocsr()

    non_khop_dict = {}
    all_nodes = set(range(num_nodes))
    for i in range(num_nodes):
        khop_neighbors = set(A_k[i].indices)
        khop_neighbors.add(i)  
        non_khop_nodes = list(all_nodes - khop_neighbors)
        non_khop_dict[i] = non_khop_nodes

    return non_khop_dict

def build_negative_texts(data, k=2, num_negatives=5):
    num_nodes = len(data.texts)
    neg_texts_list = []
    non_khop_neighbors = get_non_k_hop_neighbors(data.hyperedge_index, data.num_nodes, data.num_edges, k=k)

    for node_id in range(num_nodes):
        candidate_ids = non_khop_neighbors[node_id]
        if len(candidate_ids) == 0:
            neg_texts = ["" for _ in range(num_negatives)]  # fallback
        else:
            sampled_ids = random.sample(candidate_ids, min(num_negatives, len(candidate_ids)))
            neg_texts = [data.texts[i] for i in sampled_ids]

        neg_texts_list.append(neg_texts)

    return neg_texts_list 

def build_negative_prompts(hard_neg_texts: list[list[str]]) -> list[str]:

    neg_prompts = []
    for text_list in hard_neg_texts:
        combined_text = " ".join(text_list)
        prompt = f'Some node far away from you said: {combined_text}'
        neg_prompts.append(prompt)
    return neg_prompts
