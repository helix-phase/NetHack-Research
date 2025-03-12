#!/usr/bin/env python3
import os
import sys
import argparse
from src.models.bco import train, TrainConfig
from multiprocessing import set_start_method
from katakomba.utils.roles import Role, Race, Alignment


def parse_character(character_str):
    """Parse character string into role, race, alignment objects"""
    parts = character_str.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid character format: {character_str}. Expected format: 'role-race-alignment'")
    return Role(parts[0]), Race(parts[1]), Alignment(parts[2])


def main():
    parser = argparse.ArgumentParser(description="Inverse BC Training for NetHack")
    parser.add_argument("--character", type=str, default="ran-orc-cha",
                        help="Character type in format 'role-race-alignment'")
    parser.add_argument("--data_path", type=str,
                        default="/code/NetHack-Research/data/processed/hdf5_data/",
                        help="Path to HDF5 data files")
    parser.add_argument("--data_mode", type=str, default="compressed",
                        choices=["compressed", "in_memory", "memmap"],
                        help="Data loading mode")
    parser.add_argument("--checkpoints_path", type=str, default="./checkpoints",
                        help="Path to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--seq_len", type=int, default=8,
                        help="Sequence length for training")
    parser.add_argument("--update_steps", type=int, default=25000,
                        help="Number of update steps")
    parser.add_argument("--eval_episodes", type=int, default=25,
                        help="Number of episodes to evaluate on")
    parser.add_argument("--eval_every", type=int, default=5000,
                        help="Evaluation interval (steps)")
    parser.add_argument("--train_inverse", action="store_true",
                        help="Train the inverse model along with BC")
    parser.add_argument("--inverse_weight", type=float, default=1.0,
                        help="Weight for inverse model loss")
    parser.add_argument("--use_diff_vector", action="store_true",
                        help="Use difference vector for inverse model")
    parser.add_argument("--inverse_model_path", type=str, default=None,
                        help="Path to pretrained inverse model (optional)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="NetHack",
                        help="WandB project name")
    parser.add_argument("--wandb_group", type=str, default="inverse_bc",
                        help="WandB group name")

    args = parser.parse_args()

    # Parse the character string
    role, race, align = parse_character(args.character)

    # Create a config with the parsed arguments
    config = TrainConfig(
        character=args.character,
        data_path=args.data_path,
        data_mode=args.data_mode,
        project=args.wandb_project,
        group=args.wandb_group,
        name=f"inverse_bc-{args.character}",
        checkpoints_path=args.checkpoints_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        update_steps=args.update_steps,
        eval_episodes=args.eval_episodes,
        eval_every=args.eval_every,
        use_inverse_model=True,  # Always use inverse model for state-only data
        train_inverse_model=args.train_inverse,
        inverse_model_weight=args.inverse_weight,
        use_difference_vector=args.use_diff_vector,
        inverse_model_path=args.inverse_model_path,
        train_seed=args.seed,
        eval_seed=args.seed + 10,
    )

    # Set role, race, align explicitly since they're needed in the code
    config.role = role
    config.race = race
    config.align = align

    print(f"Starting training with character: {args.character}")
    print(f"Inverse model training: {'Enabled' if args.train_inverse else 'Disabled'}")
    print(f"Checkpoints will be saved to: {args.checkpoints_path}")

    try:
        # Run training
        actor, inverse_model = train(config)

        print("\nTraining complete!")
        if config.checkpoints_path:
            print(f"Final models saved to: {config.checkpoints_path}")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    set_start_method("spawn")
    sys.exit(main())