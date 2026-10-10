# OcuGraphRAG

Guideline-Grounded Graph Retrieval-Augmented Generation and Parameter-Efficient Alignment for Ophthalmic Decision Support.

Published in BMC Medical Informatics and Decision Making.

## Structure

```
OcuGraphRAG/
├── configs/default.yaml
├── data/eyesGraph/
├── ocugraphrag/
│   ├── schema.py
│   ├── canonicalize.py
│   ├── graph.py
│   ├── query.py
│   ├── dense.py
│   ├── paths.py
│   ├── rerank.py
│   ├── prompt.py
│   ├── generate.py
│   ├── peft_train.py
│   ├── pipeline.py
│   └── evaluate.py
└── scripts/
    ├── build_index.py
    ├── train_reranker.py
    ├── train_peft.py
    ├── infer.py
    └── evaluate.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

1. Build chunk index: `python scripts/build_index.py`
2. Train reranker: `python scripts/train_reranker.py`
3. Dual-stage PEFT: `python scripts/train_peft.py`
4. Inference: `python scripts/infer.py`
5. Evaluation: `python scripts/evaluate.py`

## Configuration

All hyperparameters are in `configs/default.yaml`:
- Base LLM: Qwen2.5-7B-Instruct
- Embedding: BGE-M3
- Chunk: 512 tokens / 64 overlap
- LoRA: rank 32, alpha 64, dropout 0.05
- Fusion weights: alpha=0.35, beta=0.30, gamma=0.20, delta=0.15

## Citation

```bibtex
@article{zhou2026ocugraphrag,
  title={OcuGraphRAG: Guideline-Grounded Graph Retrieval-Augmented Generation and Parameter-Efficient Alignment for Ophthalmic Decision Support},
  author={Zhou, Peng and Zhao, Ming-Wei and Li, Xiao-Xin},
  journal={BMC Medical Informatics and Decision Making},
  year={2026}
}
```
