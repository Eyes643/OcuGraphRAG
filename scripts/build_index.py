import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from ocugraphrag.canonicalize import load_tokenizer, canonicalize_document
from ocugraphrag.dense import DenseRetriever
from ocugraphrag.graph import load_graph


def main():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tokenizer = load_tokenizer(cfg["model"]["base"])
    guideline_dir = Path("data/guidelines")
    all_chunks = []
    for doc_path in sorted(guideline_dir.glob("*.txt")):
        chunks = canonicalize_document(
            doc_path, doc_path.stem, tokenizer,
            cfg["chunking"]["length"], cfg["chunking"]["overlap"],
        )
        all_chunks.extend(chunks)

    retriever = DenseRetriever(cfg["embedding"]["model"])
    retriever.build_index(all_chunks)

    G = load_graph("data/eyesGraph/combined.json")
    print(f"Chunks: {len(all_chunks)}, Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()
