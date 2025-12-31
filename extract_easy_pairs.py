"""
Extract easy (more distinguishable) pairs from the training data.
Easy pairs are defined as those with high label_confidence and/or large score difference.
"""
import json
import os
import random
from pathlib import Path

def load_pairs(filepath):
    """Load pairs from a jsonl file."""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return pairs

def filter_easy_pairs(pairs, min_confidence=0.9, min_score_diff=None):
    """
    Filter pairs to keep only easy ones.

    Args:
        pairs: List of pair dictionaries
        min_confidence: Minimum label_confidence threshold (default: 0.9)
        min_score_diff: Minimum absolute score difference (optional)

    Returns:
        List of easy pairs
    """
    easy_pairs = []
    for pair in pairs:
        confidence = pair.get('label_confidence', 0)
        score_diff = abs(pair['score_a'] - pair['score_b'])

        # Check confidence threshold
        if confidence < min_confidence:
            continue

        # Check score difference threshold if specified
        if min_score_diff is not None and score_diff < min_score_diff:
            continue

        easy_pairs.append(pair)

    return easy_pairs

def save_pairs(pairs, filepath):
    """Save pairs to a jsonl file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

def main():
    input_dir = Path('c:/dev/Fire_Simulator/combined_fast')
    output_dir = Path('c:/dev/Fire_Simulator/combined_fast_easy')

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Configuration for "easy" pairs
    MIN_CONFIDENCE = 0.9  # High confidence threshold
    MIN_SCORE_DIFF = 0.1  # Minimum score difference for clear distinction

    print(f"Extracting easy pairs with:")
    print(f"  - min_confidence >= {MIN_CONFIDENCE}")
    print(f"  - min_score_diff >= {MIN_SCORE_DIFF}")
    print()

    # Process each split
    for split in ['train', 'val', 'test']:
        input_file = input_dir / f'{split}_pairs.jsonl'
        output_file = output_dir / f'{split}_pairs.jsonl'

        print(f"Processing {split}...")

        # Load pairs
        pairs = load_pairs(input_file)
        original_count = len(pairs)

        # Filter easy pairs
        easy_pairs = filter_easy_pairs(
            pairs,
            min_confidence=MIN_CONFIDENCE,
            min_score_diff=MIN_SCORE_DIFF
        )
        easy_count = len(easy_pairs)

        # Shuffle to maintain randomness
        random.seed(42)
        random.shuffle(easy_pairs)

        # Save
        save_pairs(easy_pairs, output_file)

        print(f"  Original: {original_count:,} pairs")
        print(f"  Easy:     {easy_count:,} pairs ({easy_count/original_count*100:.1f}%)")
        print()

    # Copy config and metadata
    for file in ['config.json', 'metadata.json']:
        src = input_dir / file
        dst = output_dir / file
        if src.exists():
            import shutil
            shutil.copy(src, dst)
            print(f"Copied {file}")

    # Create symlink or copy floor_plans directory
    floor_plans_src = input_dir / 'floor_plans'
    floor_plans_dst = output_dir / 'floor_plans'
    if floor_plans_src.exists() and not floor_plans_dst.exists():
        # On Windows, just note that floor_plans should be shared
        print(f"\nNote: floor_plans directory is at {floor_plans_src}")
        print("You can create a symlink or copy it to the easy directory if needed.")

    # Print summary statistics
    print("\n" + "="*50)
    print("Summary Statistics for Easy Dataset:")
    print("="*50)

    for split in ['train', 'val', 'test']:
        output_file = output_dir / f'{split}_pairs.jsonl'
        pairs = load_pairs(output_file)

        if pairs:
            confidences = [p['label_confidence'] for p in pairs]
            score_diffs = [abs(p['score_a'] - p['score_b']) for p in pairs]

            print(f"\n{split.upper()}:")
            print(f"  Count: {len(pairs):,}")
            print(f"  Avg confidence: {sum(confidences)/len(confidences):.4f}")
            print(f"  Avg score diff: {sum(score_diffs)/len(score_diffs):.4f}")

            # Label distribution
            label_1 = sum(1 for p in pairs if p['label'] == 1)
            label_0 = len(pairs) - label_1
            print(f"  Label distribution: 0={label_0} ({label_0/len(pairs)*100:.1f}%), 1={label_1} ({label_1/len(pairs)*100:.1f}%)")

if __name__ == '__main__':
    main()
