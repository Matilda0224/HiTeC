import argparse
import os
import os.path as osp
import random
import sys
from pathlib import Path

# Allow `python train.py` from any working directory.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from HiTeC.paths import ROOT, CONFIG_PATH, EMB_DIR, resolve_device, torch_load, missing_data_hint
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import numpy as np
import time
from collections import defaultdict

def fix_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    """Semantic filter on hyperedge_index; keep similar node-edge pairs only."""
    edge2nodes = {eid: [] for eid in range(num_edges)}
    row, col = hyperedge_index
    for nid, eid in zip(row.tolist(), col.tolist()):
        if eid < num_edges:
            edge2nodes[eid].append(nid)

    filtered_pairs = []
    filtered_hypergraph = {}

    connected_nodes = set()

    for eid, node_list in edge2nodes.items():
        if len(node_list) == 0:
            continue
        node_feats = x[node_list]
        center_feat = node_feats.mean(dim=0, keepdim=True)
        sims = F.cosine_similarity(node_feats, center_feat, dim=-1)

        for i, nid in enumerate(node_list):
            if sims[i] >= similarity_threshold:
                filtered_pairs.append((nid, eid))
                connected_nodes.add(nid)
                if eid not in filtered_hypergraph:
                    filtered_hypergraph[eid] = []
                filtered_hypergraph[eid].append(nid)

    all_nodes = set(range(num_nodes))
    missing_nodes = all_nodes - connected_nodes

    print(f"[Semantic Filter] Filtered connections retained: {len(filtered_pairs)}")
    print(f"[Semantic Filter] Nodes with no remaining edge: {len(missing_nodes)}")
    print(f"⚠️ Note: Self-loops should already exist in preprocessed hypergraph.")

    if len(filtered_pairs) > 0:
        f_row = torch.tensor([p[0] for p in filtered_pairs], dtype=torch.long, device=device)
        f_col = torch.tensor([p[1] for p in filtered_pairs], dtype=torch.long, device=device)
        filtered_hyperedge_index = torch.stack([f_row, f_col], dim=0)
    else:
        filtered_hyperedge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    return filtered_hyperedge_index, filtered_hypergraph

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HiTeC unsupervised training.')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'citeseer2', 'history', 'photo','computers','fitness'])
    parser.add_argument('--device', type=str, default='auto',
                        help="GPU id (0), 'cuda:0', 'cpu', or 'auto' (default).")
    parser.add_argument('--config', type=str, default=str(CONFIG_PATH), help='Path to config.yaml')

    parser.add_argument('--plm_type', type=str, default='bert', choices = ['bert', 'roberta', 'distilbert', 'deberta'])
    parser.add_argument('--num_neighbors', type=int, default=10, help='length of node sequence')
    parser.add_argument('--train_textencoder',  action='store_true')
    parser.add_argument('--pooling', type=str, default='cls', choices = ['cls', 'mean'])
    parser.add_argument('--use_lora',  action='store_true')
    parser.add_argument('--encode_emb',  action='store_true')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--pro_dim',  type=int, default=4096, help='output dim of projector')
    parser.add_argument('--save_model_path', type=str, default='checkpoints/textencoder')
    parser.add_argument('--save_emb_path', type=str, default=str(EMB_DIR))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--plm_epochs', type=int, default=5)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    # hitec para
    parser.add_argument('--model_type', type=str, default='hitec_ngs', choices=['hitec_n', 'hitec_ng', 'hitec_ngs'])
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--num_splits', type=int, default=20, help='Linear-probe splits (paper: 20).')
    parser.add_argument('--epochs', type=int, default=None, help='Override Stage-2 epochs from config.yaml.')
    parser.add_argument('--smoke', action='store_true',
                        help='Fast install check: 1 seed, 1 split, 2 epochs.')
    parser.add_argument('--s_walk', type=int, default=5)
    parser.add_argument('--sub_rate', type=float, default=0.3)
    parser.add_argument('--similarity_threshold', type=float, default=0.9)

    parser.set_defaults(train_textencoder=False) 
    parser.set_defaults(use_lora=True)
    parser.set_defaults(encode_emb=False)
    

    args = parser.parse_args()
    print(args)
    config_path = args.config if osp.isabs(args.config) else str(ROOT / args.config)
    if not osp.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = yaml.safe_load(open(config_path))
    if args.dataset not in cfg:
        raise KeyError(
            f"Dataset '{args.dataset}' has no block in {config_path}. "
            f"Available: {sorted(k for k in cfg if not str(k).startswith('_'))}"
        )
    params = cfg[args.dataset]
    print(params) 
    params['sub_rate'] = args.sub_rate
    params['s_walk'] = args.s_walk
    if args.epochs is not None:
        params['epochs'] = args.epochs
    if args.smoke:
        args.num_seeds = 1
        args.num_splits = 1
        params['epochs'] = min(int(params.get('epochs', 2)), 2)
        print(f"[smoke] 1 seed, 1 split, {params['epochs']} epochs")

    from HiTeC.train_hitec import train_hitec
    from HiTeC.utils import (
        get_semantic_score,
        drop_incidence_with_semantic_score,
        sample_important_subhypergraph_swalk,
        select_important_nodes,
    )
    from data.loader import DatasetLoader
    from HiTeC.models import HyperEncoder, HiTeC
    from evaluation.evaluation import node_classification_eval, edge_prediction_eval

    # read dataset
    device = resolve_device(args.device)
    print(f"[device] {device}")
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
        from TextEncoder.models import TextEncoder
        from TextEncoder.train_textencoder import (
            ContrastiveHardTextDataset,
            train_textencoder_hard_neg,
            save_text_encoder,
            load_text_encoder,
        )
        from augment.augment import get_all_one_hop_neighbors, build_negative_texts, build_all_prompts

        save_model_path = args.save_model_path
        if not osp.isabs(save_model_path):
            save_model_path = str(ROOT / save_model_path)
        save_emb_path = args.save_emb_path
        if not osp.isabs(save_emb_path):
            save_emb_path = str(ROOT / save_emb_path)

        if args.train_textencoder:
            textencoder = TextEncoder(model_name=args.plm_type,  use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout = args.lora_dropout, pooling = args.pooling).to(device)
            textencoder.print_num_parameters()

            augment_texts = get_all_one_hop_neighbors(nodes2edge=node2edges, edge2nodes=edge2nodes,  raw_text_list = raw_texts, text_embs = data.features, max_neighbors=5) # [L , ~10]
            print(f"len of raw texts:{len(raw_texts)}, aug_texts:{len(augment_texts)}")
            hard_neg_texts = build_negative_texts(data=data, k=3, num_negatives=3)
            dataset = ContrastiveHardTextDataset(raw_texts, augment_texts, hard_neg_texts, textencoder, max_length=args.max_length, device = device)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) 

            textencoder_loss = train_textencoder_hard_neg(textencoder, dataloader, epochs=args.plm_epochs, lr=2e-5, temperature=0.5) 

            encoder_path = osp.join(save_model_path , args.dataset)
            save_text_encoder(textencoder, encoder_path)
            print(f"[✓] Saved text encoder to {encoder_path}")

        else:
            # load from checkpoint
            textencoder = TextEncoder(model_name=args.plm_type, use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout = args.lora_dropout, pooling = args.pooling).to(device)
            textencoder.print_num_parameters()
            encoder_path = osp.join(save_model_path , args.dataset)
            textencoder = load_text_encoder(textencoder,  encoder_path, device=device)
        
        emb_dir = osp.join(save_emb_path, args.dataset)
        p_path = osp.join(emb_dir, f'augmented_emb.pt')
        r_path = osp.join(emb_dir, 'raw_emb.pt')
       
        prompts = build_all_prompts(data, args.dataset, node2edges, edge2nodes,args.num_neighbors)
        print(f"len of prompts:{len(prompts)}")

        prompts_emb = encode_and_save(prompts, text_encoder=textencoder, save_path=p_path,max_length=args.max_length, device=device).to(device)
        raw_emb = encode_and_save(raw_texts, text_encoder=textencoder, save_path=r_path, max_length=args.max_length, device=device).to(device) 
        
        data.prompts_emb = prompts_emb
        data.raw_emb = raw_emb
        print("[✓] Stage-1 embeddings written. Re-run without --encode_emb to train Stage-2.")
        sys.exit(0)
    else:
        save_emb_path = args.save_emb_path
        if not osp.isabs(save_emb_path):
            save_emb_path = str(ROOT / save_emb_path)
        emb_dir = osp.join(save_emb_path, args.dataset)
        p_path = osp.join(emb_dir, f'augmented_emb.pt')
        r_path = osp.join(emb_dir, 'raw_emb.pt')
        if not osp.exists(p_path) or not osp.exists(r_path):
            raise FileNotFoundError(
                missing_data_hint(args.dataset, kind="emb")
                + f"\nLooked for:\n  {p_path}\n  {r_path}"
            )
        
        prompts_emb = torch_load(p_path).to(device) 
        data.prompts_emb = prompts_emb
        raw_emb = torch_load(r_path).to(device) 
        data.raw_emb = raw_emb
        print(f'load augmented_emb from {p_path}, and raw_emb from {r_path}.')
        
        data.prompts_score = get_semantic_score(data.hypergraph, data.prompts_emb, data.num_edges)
        data.raw_score = get_semantic_score(data.hypergraph, data.raw_emb, data.num_edges)
        print('semantic-score load')

        node_accs = []
        edge_accs = []
        for seed in range(args.num_seeds):
            fix_seed(seed)
            
            hyperedge_index1 = drop_incidence_with_semantic_score(hyperedge_index, data.prompts_score, params['drop_incidence_rate'],0.5)
            hyperedge_index2 = drop_incidence_with_semantic_score(hyperedge_index, data.raw_score, params['drop_incidence_rate'],0.5)
            
            imp_n1 = select_important_nodes(hyperedge_index1, prompts_emb, params['sub_rate'])
            imp_n2 = select_important_nodes(hyperedge_index2, raw_emb, params['sub_rate'])
    
            imp_set1 = set(imp_n1.tolist())
            imp_set2 = set(imp_n2.tolist())
            shared_nodes = sorted(list(imp_set1 & imp_set2))

            subgraph1 = sample_important_subhypergraph_swalk(hyperedge_index=hyperedge_index1, important_nodes= shared_nodes, x=prompts_emb,walk_len=params['n_walk'],restart_prob=0.3,s=params['s_walk'])
            subgraph2 = sample_important_subhypergraph_swalk(hyperedge_index=hyperedge_index2, important_nodes= shared_nodes, x=raw_emb,walk_len=params['n_walk'],restart_prob=0.3,s=params['s_walk'])

            
            encoder = HyperEncoder(data.prompts_emb.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'])
            model = HiTeC(encoder, params['proj_dim']).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            print("[•] Training hnnencoder started.")
            start_time = time.time()  
            # epoch_times = []
            for epoch in tqdm(range(1, params['epochs'] + 1)):
                t0 = time.time()
                encoder_loss = train_hitec(data, model, optimizer, args.model_type, params, hyperedge_index1, hyperedge_index2,subgraph1,subgraph2, num_negs=None)
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
            
            node_acc = node_classification_eval(encoder=model, data=data, num_splits=args.num_splits)
            edge_acc = edge_prediction_eval(encoder=model, data=data, dataset=args.dataset, device=device, num_splits=args.num_splits)
            
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
        print(f'On task:[linear_node], [Final] dataset: {args.dataset}, test_acc: {node_accs_mean[2]:.2f}+-{node_accs_std[2]:.2f}')
        
        edge_accs = np.array(edge_accs).reshape(-1, 3)
        edge_accs_mean = list(np.mean(edge_accs, axis=0))
        edge_accs_std = list(np.std(edge_accs, axis=0))
        print(f'On task:[linear_edge], [Final] dataset: {args.dataset}, test_acc: {edge_accs_mean[2]:.2f}+-{edge_accs_std[2]:.2f}')

        