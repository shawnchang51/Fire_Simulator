"""
从 raw_pairs.jsonl 过滤特定的 floor plans 然后重新分配

支持：
- 只使用前 N 个 plans
- 指定特定的 plan IDs
- 排除某些 plans
"""

import json
import argparse
from data_validator import create_dataset_splits, DataValidator
from pair_constructor import PairWriter, PairwiseLabel


def filter_and_resplit(
    raw_pairs_file: str,
    output_dir: str,
    max_plans: int = None,
    plan_ids: list = None,
    exclude_plans: list = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    过滤 floor plans 然后重新分配数据集

    Args:
        raw_pairs_file: raw_pairs.jsonl 文件路径
        output_dir: 输出目录
        max_plans: 只使用前 N 个 floor plans（按 ID 排序）
        plan_ids: 明确指定要使用的 plan IDs 列表
        exclude_plans: 要排除的 plan IDs 列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    """

    # 读取所有 pairs
    print(f"Reading pairs from {raw_pairs_file}...")
    all_pairs = []
    with open(raw_pairs_file, 'r') as f:
        for line in f:
            all_pairs.append(json.loads(line))

    print(f"Loaded {len(all_pairs)} pairs")

    # 收集所有 floor plan IDs
    all_plan_ids = set()
    for pair in all_pairs:
        all_plan_ids.add(pair.get('floor_plan_id_a'))
        all_plan_ids.add(pair.get('floor_plan_id_b'))

    print(f"Found {len(all_plan_ids)} unique floor plans (IDs: {min(all_plan_ids)} to {max(all_plan_ids)})")

    # 确定要使用的 plan IDs
    if plan_ids is not None:
        selected_plans = set(plan_ids)
        print(f"\nUsing specified {len(selected_plans)} plans")
    elif max_plans is not None:
        # 使用前 N 个 plans（按 ID 排序）
        sorted_plans = sorted(all_plan_ids)
        selected_plans = set(sorted_plans[:max_plans])
        print(f"\nUsing first {len(selected_plans)} plans (IDs: {min(selected_plans)} to {max(selected_plans)})")
    else:
        selected_plans = all_plan_ids
        print(f"\nUsing all {len(selected_plans)} plans")

    # 排除指定的 plans
    if exclude_plans:
        selected_plans -= set(exclude_plans)
        print(f"Excluded {len(exclude_plans)} plans, remaining: {len(selected_plans)} plans")

    # 过滤 pairs：只保留两端都在 selected_plans 中的 pairs
    filtered_pairs = []
    for pair in all_pairs:
        plan_a = pair.get('floor_plan_id_a')
        plan_b = pair.get('floor_plan_id_b')

        if plan_a in selected_plans and plan_b in selected_plans:
            filtered_pairs.append(pair)

    print(f"\nFiltered pairs: {len(all_pairs)} -> {len(filtered_pairs)}")

    if len(filtered_pairs) == 0:
        print("ERROR: No pairs remaining after filtering!")
        return

    # 统计 pair types
    pair_types = {}
    for pair in filtered_pairs:
        pt = pair.get('pair_type', 'unknown')
        pair_types[pt] = pair_types.get(pt, 0) + 1

    print(f"Pair type distribution:")
    for pt, count in sorted(pair_types.items()):
        print(f"  {pt}: {count} ({count/len(filtered_pairs)*100:.1f}%)")

    # 重新分配
    print(f"\nSplitting with ratios: train={train_ratio:.0%}, val={val_ratio:.0%}, test={1-train_ratio-val_ratio:.0%}")
    print(f"Random seed: {seed}")

    train_pairs, val_pairs, test_pairs = create_dataset_splits(
        filtered_pairs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )

    print(f"\nSplit results:")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Val:   {len(val_pairs)} pairs")
    print(f"  Test:  {len(test_pairs)} pairs")
    print(f"  Total: {len(train_pairs) + len(val_pairs) + len(test_pairs)} pairs")
    print(f"  Filtered: {len(filtered_pairs) - len(train_pairs) - len(val_pairs) - len(test_pairs)} pairs (cross-split)")

    # 验证
    print("\nValidating splits...")
    validator = DataValidator()
    report = validator.validate_dataset(train_pairs, val_pairs, test_pairs)
    report.print_summary()

    # 保存
    print(f"\nSaving to {output_dir}...")
    import os
    os.makedirs(output_dir, exist_ok=True)

    writer = PairWriter(output_dir)
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in train_pairs], 'train_pairs.jsonl')
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in val_pairs], 'val_pairs.jsonl')
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in test_pairs], 'test_pairs.jsonl')

    # 保存过滤后的 raw pairs（可选）
    with open(os.path.join(output_dir, 'filtered_raw_pairs.jsonl'), 'w') as f:
        for pair in filtered_pairs:
            f.write(json.dumps(pair) + '\n')

    print("Done!")
    print(f"\nUsed {len(selected_plans)} floor plans")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='过滤 floor plans 并重新分配数据集')
    parser.add_argument('--raw-pairs', type=str, required=True,
                        help='raw_pairs.jsonl 文件路径')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录')

    # 过滤选项（互斥）
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('--max-plans', type=int,
                              help='只使用前 N 个 floor plans')
    filter_group.add_argument('--plan-ids', type=int, nargs='+',
                              help='明确指定要使用的 plan IDs')

    parser.add_argument('--exclude-plans', type=int, nargs='+',
                        help='要排除的 plan IDs')

    # 分配选项
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例 (默认 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='验证集比例 (默认 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    filter_and_resplit(
        raw_pairs_file=args.raw_pairs,
        output_dir=args.output_dir,
        max_plans=args.max_plans,
        plan_ids=args.plan_ids,
        exclude_plans=args.exclude_plans,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
