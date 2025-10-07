import argparse
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Number of random vectors to generate")
    parser.add_argument("--out", type=str, default="embeddings.jsonl", help="Output file")
    args = parser.parse_args()

    # Generate n random embeddings, each of dimension 3, values in (-1, 1)
    embeddings = np.random.uniform(low=-1.0, high=1.0, size=(args.n, 3))

    with open(args.out, "w", buffering=1) as f:
        for embedding in embeddings:
            rec = {"embedding": embedding.tolist()}
            f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()