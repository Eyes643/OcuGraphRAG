import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from ocugraphrag.pipeline import OcuGraphRAGPipeline


def main():
    pipeline = OcuGraphRAGPipeline("configs/default.yaml")
    pipeline.load_graph("data/eyesGraph/combined.json")
    print("Graph loaded. Provide chunks and adapter to run inference.")
    print("Example: pipeline.infer('What is the treatment for blepharitis?')")


if __name__ == "__main__":
    main()
