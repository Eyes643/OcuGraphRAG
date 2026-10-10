import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import yaml
from ocugraphrag.evaluate import evaluate_all, paired_bootstrap


def main():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Evaluation with seed={cfg['seed']}")
    print("(Placeholder: load predictions and gold data to evaluate)")


if __name__ == "__main__":
    main()
