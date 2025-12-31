"""
Extract training data by difficulty level (easy/medium/hard).

Difficulty is determined by:
- label_confidence: How certain the label is
- score_diff: Absolute difference between score_a and score_b

Easy pairs: High confidence, large score difference (clearly distinguishable)
Medium pairs: Moderate confidence/difference
Hard pairs: Low confidence, small score difference (ambiguous)

Usage:
    # Extract only easy pairs (default)
    python extract_easy_pairs.py --difficulty easy

    # Extract mixed dataset with proportions
    python extract_easy_pairs.py --difficulty mixed --easy-ratio 0.5 --medium-ratio 0.3 --hard-ratio 0.2

    # Custom thresholds
    python extract_easy_pairs.py --difficulty easy --min-confidence 0.95 --min-score-diff 0.2

    # Specify output directory
    python extract_easy_pairs.py --difficulty medium --output-dir combined_fast_medium
"""
import argparse
import json
import random
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class DifficultyThresholds:
    """Thresholds for classifying pair difficulty."""
    # Easy: confidence >= easy_conf AND score_diff >= easy_diff
    easy_conf: float = 0.9
    easy_diff: float = 0.1

    # Hard: confidence < hard_conf OR score_diff < hard_diff
    hard_conf: float = 0.6
    hard_diff: float = 0.05

    # Medium: everything in between


def load_pairs(filepath: Path) -> List[Dict]:
    """Load pairs from a jsonl file."""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return pairs


def save_pairs(pairs: List[Dict], filepath: Path) -> None:
    """Save pairs to a jsonl file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')


def classify_difficulty(pair: Dict, thresholds: DifficultyThresholds) -> str:
    """
    Classify a pair as easy, medium, or hard.

    Returns:
        'easy', 'medium', or 'hard'
    """
    confidence = pair.get('label_confidence', 0)
    score_diff = abs(pair['score_a'] - pair['score_b'])

    # Easy: high confidence AND large score difference
    if confidence >= thresholds.easy_conf and score_diff >= thresholds.easy_diff:
        return 'easy'

    # Hard: low confidence OR small score difference
    if confidence < thresholds.hard_conf or score_diff < thresholds.hard_diff:
        return 'hard'

    # Medium: everything else
    return 'medium'


def split_by_difficulty(
    pairs: List[Dict],
    thresholds: DifficultyThresholds
) -> Dict[str, List[Dict]]:
    """Split pairs into easy, medium, hard categories."""
    result = {'easy': [], 'medium': [], 'hard': []}

    for pair in pairs:
        difficulty = classify_difficulty(pair, thresholds)
        result[difficulty].append(pair)

    return result


def sample_by_proportion(
    difficulty_splits: Dict[str, List[Dict]],
    ratios: Dict[str, float],
    total_target: Optional[int] = None,
    seed: int = 42
) -> List[Dict]:
    """
    Sample from each difficulty category according to ratios.

    Args:
        difficulty_splits: Dict with 'easy', 'medium', 'hard' lists
        ratios: Dict with ratios for each difficulty (should sum to 1.0)
        total_target: Target total count (if None, use max available)
        seed: Random seed

    Returns:
        Combined list of sampled pairs
    """
    random.seed(seed)

    # Normalize ratios
    total_ratio = sum(ratios.values())
    ratios = {k: v / total_ratio for k, v in ratios.items()}

    # Calculate available counts
    available = {k: len(v) for k, v in difficulty_splits.items()}

    # If no target specified, calculate max possible while maintaining ratios
    if total_target is None:
        # Find limiting factor
        max_possible = float('inf')
        for diff, ratio in ratios.items():
            if ratio > 0:
                max_for_diff = available[diff] / ratio
                max_possible = min(max_possible, max_for_diff)
        total_target = int(max_possible)

    # Calculate counts for each difficulty
    counts = {}
    for diff, ratio in ratios.items():
        target = int(total_target * ratio)
        counts[diff] = min(target, available[diff])

    # Sample from each category
    sampled = []
    for diff, count in counts.items():
        if count > 0 and difficulty_splits[diff]:
            samples = random.sample(difficulty_splits[diff], min(count, len(difficulty_splits[diff])))
            sampled.extend(samples)

    # Shuffle the combined result
    random.shuffle(sampled)

    return sampled


def print_stats(pairs: List[Dict], label: str = "") -> None:
    """Print statistics for a set of pairs."""
    if not pairs:
        print(f"  {label}: 0 pairs")
        return

    confidences = [p['label_confidence'] for p in pairs]
    score_diffs = [abs(p['score_a'] - p['score_b']) for p in pairs]

    print(f"  {label}:")
    print(f"    Count: {len(pairs):,}")
    print(f"    Avg confidence: {sum(confidences)/len(confidences):.4f}")
    print(f"    Avg score diff: {sum(score_diffs)/len(score_diffs):.4f}")

    # Label distribution
    label_1 = sum(1 for p in pairs if p['label'] == 1)
    label_0 = len(pairs) - label_1
    print(f"    Labels: 0={label_0} ({label_0/len(pairs)*100:.1f}%), 1={label_1} ({label_1/len(pairs)*100:.1f}%)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract training data by difficulty level",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Easy pairs only (high confidence, large score difference)
  python extract_easy_pairs.py --difficulty easy

  # Medium difficulty pairs
  python extract_easy_pairs.py --difficulty medium

  # Hard pairs only (low confidence, small score difference)
  python extract_easy_pairs.py --difficulty hard

  # Mixed dataset with custom proportions
  python extract_easy_pairs.py --difficulty mixed --easy-ratio 0.5 --medium-ratio 0.3 --hard-ratio 0.2

  # Custom thresholds for what counts as "easy"
  python extract_easy_pairs.py --difficulty easy --min-confidence 0.95 --min-score-diff 0.15
"""
    )

    # Difficulty selection
    parser.add_argument(
        '--difficulty', '-d',
        type=str,
        choices=['easy', 'medium', 'hard', 'mixed', 'all'],
        default='easy',
        help="Difficulty level to extract: "
             "'easy' (high confidence), "
             "'medium' (moderate), "
             "'hard' (low confidence), "
             "'mixed' (custom proportions), "
             "'all' (keep all, just classify)"
    )

    # Proportion control for mixed mode
    parser.add_argument(
        '--easy-ratio',
        type=float,
        default=0.5,
        help="Proportion of easy pairs in mixed mode (default: 0.5)"
    )
    parser.add_argument(
        '--medium-ratio',
        type=float,
        default=0.3,
        help="Proportion of medium pairs in mixed mode (default: 0.3)"
    )
    parser.add_argument(
        '--hard-ratio',
        type=float,
        default=0.2,
        help="Proportion of hard pairs in mixed mode (default: 0.2)"
    )

    # Threshold customization
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.9,
        help="Minimum confidence for 'easy' pairs (default: 0.9)"
    )
    parser.add_argument(
        '--min-score-diff',
        type=float,
        default=0.1,
        help="Minimum score difference for 'easy' pairs (default: 0.1)"
    )
    parser.add_argument(
        '--hard-confidence',
        type=float,
        default=0.6,
        help="Maximum confidence for 'hard' pairs (default: 0.6)"
    )
    parser.add_argument(
        '--hard-score-diff',
        type=float,
        default=0.05,
        help="Maximum score difference for 'hard' pairs (default: 0.05)"
    )

    # I/O paths
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default='combined_fast',
        help="Input directory containing train/val/test_pairs.jsonl (default: combined_fast)"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help="Output directory (default: {input_dir}_{difficulty})"
    )

    # Other options
    parser.add_argument(
        '--max-pairs',
        type=int,
        default=None,
        help="Maximum number of pairs per split (optional)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help="Only analyze and print statistics, don't save files"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set up paths - resolve relative to current working directory
    input_dir = Path(args.input_dir).resolve()

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_{args.difficulty}"

    # Create thresholds
    thresholds = DifficultyThresholds(
        easy_conf=args.min_confidence,
        easy_diff=args.min_score_diff,
        hard_conf=args.hard_confidence,
        hard_diff=args.hard_score_diff
    )

    print("=" * 60)
    print("EXTRACT TRAINING DATA BY DIFFICULTY")
    print("=" * 60)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"\nDifficulty mode: {args.difficulty}")
    print(f"\nThresholds:")
    print(f"  Easy:   confidence >= {thresholds.easy_conf}, score_diff >= {thresholds.easy_diff}")
    print(f"  Hard:   confidence < {thresholds.hard_conf} OR score_diff < {thresholds.hard_diff}")
    print(f"  Medium: everything else")

    if args.difficulty == 'mixed':
        total = args.easy_ratio + args.medium_ratio + args.hard_ratio
        print(f"\nMixed proportions (normalized):")
        print(f"  Easy:   {args.easy_ratio/total:.1%}")
        print(f"  Medium: {args.medium_ratio/total:.1%}")
        print(f"  Hard:   {args.hard_ratio/total:.1%}")

    print()

    # Create output directory
    if not args.analyze_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'val', 'test']:
        input_file = input_dir / f'{split}_pairs.jsonl'
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping")
            continue

        print(f"\n{'='*40}")
        print(f"Processing {split}...")
        print('='*40)

        # Load pairs
        pairs = load_pairs(input_file)
        print(f"Loaded {len(pairs):,} pairs")

        # Split by difficulty
        splits = split_by_difficulty(pairs, thresholds)

        print(f"\nDifficulty distribution:")
        for diff in ['easy', 'medium', 'hard']:
            count = len(splits[diff])
            pct = count / len(pairs) * 100
            print(f"  {diff.capitalize():8s}: {count:,} ({pct:.1f}%)")

        # Select pairs based on mode
        if args.difficulty == 'easy':
            selected = splits['easy']
        elif args.difficulty == 'medium':
            selected = splits['medium']
        elif args.difficulty == 'hard':
            selected = splits['hard']
        elif args.difficulty == 'mixed':
            ratios = {
                'easy': args.easy_ratio,
                'medium': args.medium_ratio,
                'hard': args.hard_ratio
            }
            selected = sample_by_proportion(
                splits, ratios,
                total_target=args.max_pairs,
                seed=args.seed
            )
        else:  # 'all'
            selected = pairs

        # Apply max_pairs limit
        if args.max_pairs and len(selected) > args.max_pairs:
            random.seed(args.seed)
            random.shuffle(selected)
            selected = selected[:args.max_pairs]

        # Shuffle
        random.seed(args.seed + hash(split))
        random.shuffle(selected)

        print(f"\nSelected: {len(selected):,} pairs ({len(selected)/len(pairs)*100:.1f}%)")
        print_stats(selected, "Statistics")

        # Save
        if not args.analyze_only:
            output_file = output_dir / f'{split}_pairs.jsonl'
            save_pairs(selected, output_file)
            print(f"Saved to {output_file}")

    # Copy config and metadata
    if not args.analyze_only:
        print(f"\n{'='*40}")
        print("Copying auxiliary files...")
        print('='*40)

        for file in ['config.json', 'metadata.json']:
            src = input_dir / file
            dst = output_dir / file
            if src.exists():
                shutil.copy(src, dst)
                print(f"  Copied {file}")

        # Handle floor_plans directory
        floor_plans_src = input_dir / 'floor_plans'
        floor_plans_dst = output_dir / 'floor_plans'
        if floor_plans_src.exists() and not floor_plans_dst.exists():
            # Try to create a junction/symlink on Windows
            try:
                import subprocess
                subprocess.run(
                    ['cmd', '/c', 'mklink', '/J', str(floor_plans_dst), str(floor_plans_src)],
                    check=True, capture_output=True
                )
                print(f"  Created junction for floor_plans")
            except Exception:
                print(f"  Note: Link floor_plans manually or copy from {floor_plans_src}")

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    if not args.analyze_only:
        print(f"\nCreated {args.difficulty} dataset at: {output_dir}")

        # Print final counts
        for split in ['train', 'val', 'test']:
            output_file = output_dir / f'{split}_pairs.jsonl'
            if output_file.exists():
                count = sum(1 for _ in open(output_file))
                print(f"  {split}: {count:,} pairs")

    print("\nDone!")


if __name__ == '__main__':
    main()
