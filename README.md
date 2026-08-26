# HiTeC: Hierarchical Contrastive Learning on Text-Attributed Hypergraph with Semantic-Aware Augmentation


**Author: Mengting Pan, Fan Li, Chen Chen, Xiaoyang Wang, Wenjie Zhang**

**Paper: [https://arxiv.org/abs/2508.03104](https://arxiv.org/abs/2508.03104)**

Published as a main conference paper at EMNLP 2026 🎉

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Datasets

We provide the original texts and hypergraph structures of the TAHGs, as well as the edge splits for hyperedge prediction. Cora and CiteSeer are in `tahg_datasets/`; History, Photo, Computers, and Fitness are on **[Google Drive](https://drive.google.com/drive/folders/1tkNOf2ehJoUxvPRTxwKPGdVdiA5MXsqC?usp=sharing)**.


## Run


**1. Quick reproduction:** — no `--encode_emb`. Loads shipped `emb/<dataset>/raw_emb.pt` and `augmented_emb.pt`.

```bash
python train.py --dataset cora --device 0
```

**2. Text encoder pretraining:** — `--encode_emb --train_textencoder`. Trains the text encoder, writes embeddings to `emb/<dataset>/`.

```bash
python train.py --dataset cora --encode_emb --train_textencoder --device 0
```

**3. Hypergraph encoder pretraining:** — no `--encode_emb`, with embeddings you already have under `emb/<dataset>/`.

```bash
python train.py --dataset cora --device 0
```

## Citation

```bibtex
@inproceedings{pan2026hitec,
  title={HiTeC: Hierarchical Contrastive Learning on Text-Attributed Hypergraph with Semantic-Aware Augmentation},
  author={Pan, Mengting and Li, Fan and Chen, Chen and Wang, Xiaoyang and Zhang, Wenjie},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026}
}
```

