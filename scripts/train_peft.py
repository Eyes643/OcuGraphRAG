import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from ocugraphrag.peft_train import train_full


def main():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("Starting dual-stage PEFT training...")
    print(f"  LoRA rank={cfg['peft']['lora_rank']}, alpha={cfg['peft']['lora_alpha']}")
    print(f"  Stage I: {cfg['peft']['stage1']['epochs']} epochs, lr={cfg['peft']['stage1']['learning_rate']}")
    print(f"  Stage II: {cfg['peft']['stage2']['epochs']} epochs, lr={cfg['peft']['stage2']['learning_rate']}")
    print("(Placeholder: provide stage1_samples and stage2_samples to run)")


if __name__ == "__main__":
    main()
