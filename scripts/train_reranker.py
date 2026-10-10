import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from ocugraphrag.rerank import ranking_loss


def main():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    margin = cfg["rerank"]["rank_margin"]
    print(f"Reranker training with margin={margin}")
    print("(Placeholder: reranker training loop would go here)")


if __name__ == "__main__":
    main()
