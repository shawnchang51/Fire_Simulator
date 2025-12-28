"""
重新分配训练数据到 train/val/test

从 raw_pairs.jsonl 读取所有 pairs，然后用不同的参数重新分配
"""

import json
from data_validator import create_dataset_splits, DataValidator
from pair_constructor import PairWriter, PairwiseLabel


def resplit_from_raw_pairs(
    raw_pairs_file: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    从 raw_pairs.jsonl 重新分配数据集

    Args:
        raw_pairs_file: raw_pairs.jsonl 文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例 (默认 0.7)
        val_ratio: 验证集比例 (默认 0.15)
        seed: 随机种子
    """

    # 读取所有 pairs
    print(f"Reading pairs from {raw_pairs_file}...")
    pairs = []
    with open(raw_pairs_file, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} pairs")

    # 重新分配
    print(f"\nSplitting with ratios: train={train_ratio:.0%}, val={val_ratio:.0%}, test={1-train_ratio-val_ratio:.0%}")
    print(f"Random seed: {seed}")

    train_pairs, val_pairs, test_pairs = create_dataset_splits(
        pairs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )

    print(f"\nSplit results:")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Val:   {len(val_pairs)} pairs")
    print(f"  Test:  {len(test_pairs)} pairs")
    print(f"  Total: {len(train_pairs) + len(val_pairs) + len(test_pairs)} pairs")
    print(f"  Filtered: {len(pairs) - len(train_pairs) - len(val_pairs) - len(test_pairs)} pairs (cross-split)")

    # 验证
    print("\nValidating splits...")
    validator = DataValidator()
    report = validator.validate_dataset(train_pairs, val_pairs, test_pairs)
    report.print_summary()

    # 保存
    print(f"\nSaving to {output_dir}...")
    writer = PairWriter(output_dir)
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in train_pairs], 'train_pairs.jsonl')
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in val_pairs], 'val_pairs.jsonl')
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in test_pairs], 'test_pairs.jsonl')

    print("Done!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='重新分配数据集')
    parser.add_argument('--raw-pairs', type=str, required=True,
                        help='raw_pairs.jsonl 文件路径')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例 (默认 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='验证集比例 (默认 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    resplit_from_raw_pairs(
        raw_pairs_file=args.raw_pairs,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
