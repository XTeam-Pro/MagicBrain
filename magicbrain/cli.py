import argparse
import sys
import os
from .brain import TextBrain
from .io import save_model, load_model
from .tasks.text_task import build_vocab, train_loop
from .tasks.self_repair import benchmark_self_repair
from .sampling import sample

DEFAULT_TEXT = """
From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
""" * 50

def main():
    parser = argparse.ArgumentParser(description="MagicBrain CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # TRAIN
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--genome", type=str, default="30121033102301230112332100123")
    train_parser.add_argument("--text", type=str, help="Path to text file. If not provided, uses default Shakespeare snippet.")
    train_parser.add_argument("--steps", type=int, default=10000)
    train_parser.add_argument("--out", type=str, default="model.npz", help="Output model path")
    train_parser.add_argument("--load", type=str, help="Resume from existing model path")

    # SAMPLE
    sample_parser = subparsers.add_parser("sample", help="Sample text from model")
    sample_parser.add_argument("--model", type=str, required=True)
    sample_parser.add_argument("--seed", type=str, default="To be")
    sample_parser.add_argument("--n", type=int, default=500)
    sample_parser.add_argument("--temp", type=float, default=0.75)

    # REPAIR
    repair_parser = subparsers.add_parser("repair", help="Run self-repair benchmark")
    repair_parser.add_argument("--genome", type=str, default="30121033102301230112332100123")
    repair_parser.add_argument("--text", type=str, help="Path to text file")
    repair_parser.add_argument("--damage", type=float, default=0.2)

    args = parser.parse_args()

    if args.command == "train":
        text = DEFAULT_TEXT
        if args.text:
            with open(args.text, "r", encoding="utf-8") as f:
                text = f.read()
        
        # Normalize
        text = " ".join(text.split())

        if args.load:
            print(f"Loading model from {args.load}...")
            brain, stoi, itos = load_model(args.load)
        else:
            print(f"Creating new brain with genome: {args.genome}")
            stoi, itos = build_vocab(text)
            brain = TextBrain(args.genome, len(stoi))

        print("Starting training...")
        train_loop(brain, text, stoi, steps=args.steps)
        
        print(f"Saving model to {args.out}...")
        save_model(brain, stoi, itos, args.out)

    elif args.command == "sample":
        print(f"Loading model from {args.model}...")
        brain, stoi, itos = load_model(args.model)
        
        print(f"Sampling with seed='{args.seed}'...")
        out = sample(brain, stoi, itos, args.seed, n=args.n, temperature=args.temp)
        print("-" * 60)
        print(out)
        print("-" * 60)

    elif args.command == "repair":
        text = DEFAULT_TEXT
        if args.text:
            with open(args.text, "r", encoding="utf-8") as f:
                text = f.read()
        text = " ".join(text.split())
        
        stoi, itos = build_vocab(text)
        brain = TextBrain(args.genome, len(stoi))
        
        # Pre-train
        print("Pre-training...")
        train_loop(brain, text, stoi, steps=10000, print_every=2000)
        
        benchmark_self_repair(brain, text, stoi, itos, damage_frac=args.damage)

if __name__ == "__main__":
    main()
