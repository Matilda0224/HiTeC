# TAHG datasets

All datasets use the same on-disk layout as `cora/` and `citeseer/`.

Download extras from: https://drive.google.com/drive/folders/1tkNOf2ehJoUxvPRTxwKPGdVdiA5MXsqC?usp=sharing

| File | Contents |
|---|---|
| `features.pt` | node features |
| `hypergraph_dict.pt` | hyperedge id → list of node ids |
| `labels.pt` | node labels |
| `texts.pt` | list of node texts |
| `splits/{0..19}.pt` | `train_mask`, `val_mask`, `test_mask` |
| `edge_bucket_cns.pt` | hyperedge-prediction buckets |

Stage-1 embeddings (same as Cora): `emb/<name>/raw_emb.pt` and `emb/<name>/augmented_emb.pt`.

Cora and CiteSeer are in git. History / Photo / Computers / Fitness should be dropped into `tahg_datasets/<name>/` after you download them. Then:

```bash
python train.py --dataset history --device auto
```
