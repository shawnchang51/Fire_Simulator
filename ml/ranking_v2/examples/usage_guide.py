"""
Ranking V2.1 使用指南 - 完整範例

本指南展示如何使用所有新增的增強功能。

使用方式:
    # 使用新建模型 (隨機權重)
    python -m ml.ranking_v2.examples.usage_guide

    # 使用訓練好的模型
    python -m ml.ranking_v2.examples.usage_guide --checkpoint path/to/model.pt

    # 使用訓練好的模型 + 真實資料
    python -m ml.ranking_v2.examples.usage_guide --checkpoint model.pt --data data.npz

    # 指定語言和輸出目錄
    python -m ml.ranking_v2.examples.usage_guide --checkpoint model.pt --language en --output-dir reports/
"""

import argparse
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# =============================================================================
# 使用者配置區 - 可直接修改這裡的值
# =============================================================================
class UserConfig:
    """
    在這裡指定你要使用的模型和資料路徑
    也可以透過命令列參數覆蓋這些設定
    """
    # 模型 checkpoint 路徑 (設為 None 則使用隨機初始化的新模型)
    CHECKPOINT_PATH: Optional[str] = None
    # 例如: CHECKPOINT_PATH = "checkpoints/ranking_v2_best.pt"
    # 例如: CHECKPOINT_PATH = r"C:\dev\Fire_Simulator\checkpoints\best_model.pt"

    # 資料路徑 (設為 None 則使用隨機生成的假資料)
    DATA_PATH: Optional[str] = None
    # 例如: DATA_PATH = "data/floor_plan_001.npz"

    # 報告語言: "zh" (中文) 或 "en" (英文)
    LANGUAGE: str = "zh"

    # 輸出目錄 (用於儲存報告、熱力圖等)
    OUTPUT_DIR: str = "outputs"

    # 要執行的示範 (None = 全部執行, 或指定數字列表如 [1, 4, 5])
    RUN_DEMOS: Optional[list] = None
    # 例如: RUN_DEMOS = [1, 4]  # 只執行示範 1 和 4


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='Ranking V2.1 功能示範指南',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-c', '--checkpoint',
        type=str,
        default=None,
        help='訓練好的模型 checkpoint 路徑 (.pt 檔案)'
    )
    parser.add_argument(
        '-d', '--data',
        type=str,
        default=None,
        help='輸入資料路徑 (.npz 檔案，包含 grid 和 scenario)'
    )
    parser.add_argument(
        '-l', '--language',
        type=str,
        choices=['zh', 'en'],
        default='zh',
        help='報告語言 (預設: zh)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='outputs',
        help='輸出目錄 (預設: outputs)'
    )
    parser.add_argument(
        '--demos',
        type=str,
        default=None,
        help='要執行的示範編號，以逗號分隔 (例如: 1,4,5)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用的示範'
    )
    return parser.parse_args()


# 合併配置
def get_config():
    """取得最終配置 (命令列參數優先於 UserConfig)"""
    args = parse_args()

    config = {
        'checkpoint': args.checkpoint or UserConfig.CHECKPOINT_PATH,
        'data': args.data or UserConfig.DATA_PATH,
        'language': args.language or UserConfig.LANGUAGE,
        'output_dir': args.output_dir or UserConfig.OUTPUT_DIR,
        'demos': None,
        'list_demos': args.list,
    }

    # 解析 demos
    if args.demos:
        config['demos'] = [int(x.strip()) for x in args.demos.split(',')]
    elif UserConfig.RUN_DEMOS:
        config['demos'] = UserConfig.RUN_DEMOS

    return config


# =============================================================================
# 全域設定
# =============================================================================
CONFIG = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 建立輸出目錄
Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

print(f"使用裝置: {device}")
print(f"Checkpoint: {CONFIG['checkpoint'] or '(使用隨機初始化)'}")
print(f"資料來源: {CONFIG['data'] or '(使用隨機生成)'}")
print(f"語言: {CONFIG['language']}")
print(f"輸出目錄: {CONFIG['output_dir']}")


# =============================================================================
# 1. 基礎設定 - 載入模型和資料
# =============================================================================

def load_model_from_checkpoint(checkpoint_path: str):
    """
    從 checkpoint 載入訓練好的模型

    Args:
        checkpoint_path: checkpoint 檔案路徑

    Returns:
        Tuple of (model, config)
    """
    from ml.ranking_v2 import RankingV2Config, CrossAttentionRanker

    print(f"\n載入 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 取得 config
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"  從 checkpoint 載入 config")
    elif 'hparams' in checkpoint:
        config = RankingV2Config(**checkpoint['hparams'])
        print(f"  從 hparams 重建 config")
    else:
        print(f"  警告: checkpoint 中沒有 config，使用預設值")
        config = RankingV2Config()

    # 建立模型
    model = CrossAttentionRanker(config)

    # 載入權重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"  模型參數量: {model.count_parameters():,}")
    print(f"  模型載入成功!")

    return model, config


def load_data_from_file(data_path: str):
    """
    從檔案載入真實資料

    Args:
        data_path: .npz 檔案路徑

    Returns:
        資料字典
    """
    print(f"\n載入資料: {data_path}")

    data = np.load(data_path)

    # 解析 grid
    if 'grid' in data:
        grid = torch.from_numpy(data['grid']).float()
    elif 'floor_plan' in data:
        grid = torch.from_numpy(data['floor_plan']).float()
    elif 'grid_a' in data:
        # 已經是成對資料
        return {
            'grid_a': torch.from_numpy(data['grid_a']).float(),
            'scenario_a': torch.from_numpy(data['scenario_a']).float(),
            'grid_b': torch.from_numpy(data['grid_b']).float(),
            'scenario_b': torch.from_numpy(data['scenario_b']).float(),
            'labels': torch.from_numpy(data.get('labels', np.array([1]))).long(),
        }
    else:
        raise ValueError(f"找不到 'grid' 或 'floor_plan' 在 {data_path}")

    # 確保是 4D (batch, channels, H, W)
    if grid.dim() == 3:
        grid = grid.unsqueeze(0)

    # 解析 scenario
    if 'scenario' in data:
        scenario = torch.from_numpy(data['scenario']).float()
    else:
        scenario = torch.tensor([[100.0, 1.0, 0.5, 30.0]])  # 預設值

    if scenario.dim() == 1:
        scenario = scenario.unsqueeze(0)

    print(f"  Grid shape: {tuple(grid.shape)}")
    print(f"  Scenario: agents={scenario[0,0]:.0f}, fires={scenario[0,1]:.0f}")

    # 建立成對資料 (複製一份作為 B，實際應用時應該用不同配置)
    return {
        'grid_a': grid,
        'scenario_a': scenario,
        'grid_b': grid.clone(),  # 用相同資料作為示範
        'scenario_b': scenario.clone(),
        'labels': torch.tensor([1]),
    }


def create_demo_data(batch_size=4):
    """創建演示用的假資料"""
    H, W = 96, 128

    # 建築平面圖 (5 通道: 牆壁, 可通行區, 門, 出口, 有效區)
    grid_a = torch.rand(batch_size, 5, H, W)
    grid_b = torch.rand(batch_size, 5, H, W)

    # 場景參數 (人數, 火源數, 火勢擴散率, 發現延遲)
    scenario_a = torch.rand(batch_size, 4)
    scenario_b = torch.rand(batch_size, 4)

    # 標籤 (1 = A 優於 B)
    labels = torch.randint(0, 2, (batch_size,))

    return {
        'grid_a': grid_a,
        'scenario_a': scenario_a,
        'grid_b': grid_b,
        'scenario_b': scenario_b,
        'labels': labels,
    }


def get_demo_data(batch_size=4):
    """
    取得示範資料 - 優先使用真實資料，否則用假資料

    Returns:
        資料字典
    """
    if CONFIG['data']:
        return load_data_from_file(CONFIG['data'])
    else:
        return create_demo_data(batch_size)


def demo_1_basic_model():
    """
    示範 1: 基礎模型使用

    可以從 checkpoint 載入訓練好的模型，或創建新模型
    """
    print("\n" + "="*60)
    print("示範 1: 基礎 Ranking V2 模型")
    print("="*60)

    from ml.ranking_v2 import RankingV2Config, CrossAttentionRanker, get_full_config

    # 根據配置決定載入方式
    if CONFIG['checkpoint']:
        # 從 checkpoint 載入訓練好的模型
        model, config = load_model_from_checkpoint(CONFIG['checkpoint'])
    else:
        # 創建新模型 (隨機初始化)
        print("\n建立新模型 (隨機權重)...")

        # 方法 1: 使用預設配置
        # config = get_full_config()

        # 方法 2: 自訂配置
        config = RankingV2Config(
            latent_dim=64,
            use_cross_attention=True,
            attention_heads=4,
            mining_strategy="curriculum",
            auxiliary_tasks=["survival_rate", "steps"],
        )

        model = CrossAttentionRanker(config).to(device)
        print(f"  模型參數量: {model.count_parameters():,}")

    # 準備資料
    data = get_demo_data()

    # 前向傳播
    with torch.no_grad():
        outputs = model(
            data['grid_a'].to(device),
            data['scenario_a'].to(device),
            data['grid_b'].to(device),
            data['scenario_b'].to(device),
        )

    print(f"\n預測結果:")
    print(f"  分數差 (logit): {outputs['logit']}")
    print(f"  配置 A 分數: {outputs['score_a']}")
    print(f"  配置 B 分數: {outputs['score_b']}")

    # 單一配置評分
    single_score = model.score_single(
        data['grid_a'].to(device),
        data['scenario_a'].to(device),
    )
    print(f"  單一配置評分: {single_score}")

    return model, config


# =============================================================================
# 2. 不確定性量化
# =============================================================================
def demo_2_uncertainty(model):
    """
    示範 2: 不確定性量化 - 知道模型「多有信心」
    """
    print("\n" + "="*60)
    print("示範 2: 不確定性量化")
    print("="*60)

    from ml.ranking_v2 import (
        MCDropoutWrapper,
        DeepEnsemble,
        create_uncertainty_model,
    )

    data = create_demo_data(batch_size=2)

    # 方法 1: MC Dropout (最簡單)
    print("\n--- MC Dropout ---")
    mc_model = MCDropoutWrapper(model, n_samples=20)

    uncertainty = mc_model.predict_with_uncertainty(
        data['grid_a'].to(device),
        data['scenario_a'].to(device),
        data['grid_b'].to(device),
        data['scenario_b'].to(device),
    )

    print(f"平均預測: {uncertainty.mean_logit}")
    print(f"預測標準差: {uncertainty.std_logit}")
    print(f"預測熵: {uncertainty.entropy}")
    print(f"信心度: {uncertainty.confidence}")
    print(f"認知不確定性 (模型不確定): {uncertainty.epistemic}")
    print(f"隨機不確定性 (資料噪音): {uncertainty.aleatoric}")

    # 實際應用: 當信心度低時，觸發真實模擬
    for i, conf in enumerate(uncertainty.confidence):
        if conf < 0.7:
            print(f"  ⚠️ 樣本 {i}: 信心度低 ({conf:.2%})，建議執行真實模擬驗證")
        else:
            print(f"  ✓ 樣本 {i}: 信心度高 ({conf:.2%})，可信賴預測結果")

    return mc_model


# =============================================================================
# 3. 配置生成與優化
# =============================================================================
def demo_3_config_generation(model):
    """
    示範 3: 自動生成最優配置
    """
    print("\n" + "="*60)
    print("示範 3: 配置生成與優化")
    print("="*60)

    from ml.ranking_v2 import (
        Configuration,
        ConfigurationScorer,
        EvolutionaryOptimizer,
        MCTSOptimizer,
    )

    # 準備基礎樓層平面 (不含門/出口的基本結構)
    H, W = 96, 128
    base_grid = torch.rand(3, H, W)  # [牆壁, 可通行區, 有效區]
    scenario = torch.rand(4)  # 場景參數

    # 有效的門/出口位置 (假設這些位置可以放置門或出口)
    valid_positions = [(y, x) for y in range(10, 80, 10) for x in range(10, 120, 10)]

    # 創建評分器
    scorer = ConfigurationScorer(model, base_grid.to(device), scenario.to(device), device)

    # 方法 1: 演化算法優化
    print("\n--- 演化算法優化 ---")
    evo_optimizer = EvolutionaryOptimizer(
        scorer=scorer,
        valid_positions=valid_positions,
        n_doors=3,
        n_exits=2,
        population_size=20,
        n_generations=10,  # 實際使用時建議 50-100
    )

    result = evo_optimizer.optimize(floor_plan_id="demo_floor")

    print(f"最佳配置分數: {result.best_config.score:.4f}")
    print(f"門位置: {result.best_config.door_positions}")
    print(f"出口位置: {result.best_config.exit_positions}")
    print(f"總評估次數: {result.total_evaluations}")
    print(f"是否收斂: {result.converged}")

    # 顯示優化過程
    print("\n優化歷程:")
    for i, h in enumerate(result.optimization_history[:5]):
        print(f"  世代 {h['generation']}: 最佳={h['best_score']:.4f}, 平均={h['mean_score']:.4f}")

    return result


# =============================================================================
# 4. 可解釋性報告
# =============================================================================
def demo_4_explainability(model, config):
    """
    示範 4: 生成可解釋的推薦報告
    """
    print("\n" + "="*60)
    print("示範 4: 可解釋性報告")
    print("="*60)

    from ml.ranking_v2 import (
        ExplanationPipeline,
        FeatureAttributor,
    )

    data = create_demo_data(batch_size=1)

    # 創建解釋管道
    explainer = ExplanationPipeline(
        model=model,
        config=config,
        device=device,
        language="zh",  # 支援 "en" 或 "zh"
    )

    # 生成單一配置的解釋報告
    print("\n--- 單一配置分析 ---")
    valid_positions = [(y, x) for y in range(10, 80, 10) for x in range(10, 120, 10)]

    report = explainer.explain_single(
        grid=data['grid_a'][0].to(device),
        scenario=data['scenario_a'][0].to(device),
        valid_positions=valid_positions,
    )

    print(f"評分: {report.score:.4f}")
    print(f"信心度: {report.confidence:.2%}")
    print(f"\n摘要:\n{report.summary}")

    # 查看改進建議
    if report.recommendations:
        print("\n改進建議:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"  {i}. {rec.reasoning}")
            print(f"     預期改善: {rec.expected_improvement:.4f}")

    # 匯出報告
    print("\n--- 匯出報告 ---")

    # JSON 格式
    json_report = explainer.export_report(report, format="json")
    print("JSON 報告 (前 200 字):")
    print(json_report[:200] + "...")

    # Markdown 格式
    md_report = explainer.export_report(report, format="markdown")
    print("\nMarkdown 報告 (前 300 字):")
    print(md_report[:300] + "...")

    return report


# =============================================================================
# 5. 主動學習
# =============================================================================
def demo_5_active_learning(model, config):
    """
    示範 5: 主動學習 - 智慧選擇最有價值的樣本
    """
    print("\n" + "="*60)
    print("示範 5: 主動學習")
    print("="*60)

    from ml.ranking_v2 import (
        ActiveLearningLoop,
        UncertaintySampling,
        DiversitySampling,
        BatchModeSampler,
        create_acquisition_function,
    )

    # 模擬未標註資料池
    unlabeled_data = [create_demo_data(batch_size=1) for _ in range(50)]
    # 轉換格式
    unlabeled_data = [
        {
            'grid_a': d['grid_a'][0],
            'scenario_a': d['scenario_a'][0],
            'grid_b': d['grid_b'][0],
            'scenario_b': d['scenario_b'][0],
        }
        for d in unlabeled_data
    ]

    # 創建主動學習迴圈
    acquisition_fn = create_acquisition_function(
        strategy='uncertainty',
        n_mc_samples=10,
        uncertainty_type='entropy',
    )

    al_loop = ActiveLearningLoop(
        model=model,
        acquisition_fn=acquisition_fn,
        config=config,
        query_batch_size=5,
        initial_labeled_size=10,
        max_queries=30,
        device=device,
    )

    # 初始化
    all_data = unlabeled_data
    al_loop.initialize(all_data)

    print(f"初始已標註: {len(al_loop.state.labeled_indices)}")
    print(f"未標註: {len(al_loop.state.unlabeled_indices)}")

    # 選擇下一批要標註的樣本
    query_result = al_loop.query(al_loop.get_unlabeled_data(all_data))

    print(f"\n選擇了 {len(query_result.indices)} 個樣本進行標註")
    print(f"選擇的索引: {query_result.indices}")
    print(f"不確定性分數: {query_result.scores}")

    # 在實際應用中，這些樣本會被送去執行模擬以獲取真實標籤
    print("\n⚡ 這些樣本會被優先送去執行火災模擬，因為模型對它們最不確定")

    return al_loop


# =============================================================================
# 6. 多目標排序
# =============================================================================
def demo_6_multi_objective(config):
    """
    示範 6: 多目標排序 - 同時考慮多個優化目標
    """
    print("\n" + "="*60)
    print("示範 6: 多目標排序")
    print("="*60)

    from ml.ranking_v2 import (
        MultiObjectiveRanker,
        ObjectiveConfig,
        ObjectiveType,
        ParetoOptimizer,
        create_default_objectives,
    )

    # 定義多個目標
    objectives = [
        ObjectiveConfig(
            name="survival_rate",
            type=ObjectiveType.SURVIVAL_RATE,
            weight=1.0,
            minimize=False,  # 越高越好
            importance="critical",
        ),
        ObjectiveConfig(
            name="evacuation_time",
            type=ObjectiveType.EVACUATION_TIME,
            weight=0.8,
            minimize=True,  # 越低越好
            importance="high",
        ),
        ObjectiveConfig(
            name="modification_cost",
            type=ObjectiveType.MODIFICATION_COST,
            weight=0.5,
            minimize=True,  # 越低越好
            importance="medium",
        ),
    ]

    # 創建多目標排序模型
    mo_ranker = MultiObjectiveRanker(
        config=config,
        objectives=objectives,
        aggregation="weighted_sum",  # 或 "chebyshev", "hypervolume"
    ).to(device)

    data = create_demo_data(batch_size=2)

    # 預測多個目標
    with torch.no_grad():
        outputs = mo_ranker(
            data['grid_a'].to(device),
            data['scenario_a'].to(device),
        )

    print("各目標預測值:")
    for name, values in outputs['objectives'].items():
        print(f"  {name}: {values}")
    print(f"\n聚合分數: {outputs['aggregated_score']}")

    # Pareto 優化
    print("\n--- Pareto 最優解分析 ---")
    pareto_opt = ParetoOptimizer(objectives)

    # 模擬多個解
    solutions = [
        {"survival_rate": 0.95, "evacuation_time": 100, "modification_cost": 50000},
        {"survival_rate": 0.90, "evacuation_time": 80, "modification_cost": 30000},
        {"survival_rate": 0.85, "evacuation_time": 60, "modification_cost": 20000},
        {"survival_rate": 0.92, "evacuation_time": 90, "modification_cost": 40000},
    ]

    # 找出 Pareto 前沿
    pareto_indices = pareto_opt.find_pareto_front(solutions)
    print(f"Pareto 最優解索引: {pareto_indices}")
    print("Pareto 最優解:")
    for idx in pareto_indices:
        print(f"  解 {idx}: {solutions[idx]}")

    return mo_ranker


# =============================================================================
# 7. 遷移學習
# =============================================================================
def demo_7_transfer_learning(model, config):
    """
    示範 7: 跨建築遷移學習
    """
    print("\n" + "="*60)
    print("示範 7: 遷移學習")
    print("="*60)

    from ml.ranking_v2 import (
        TransferLearningPipeline,
        FeatureExtractor,
        MAML,
    )

    # 方法 1: 特徵提取 (凍結編碼器，只微調評分頭)
    print("\n--- 特徵提取方法 ---")

    from copy import deepcopy
    source_model = deepcopy(model)

    extractor = FeatureExtractor(
        source_model,
        freeze_encoder=True,  # 凍結 CNN 編碼器
        freeze_attention=False,  # 微調注意力層
    )

    trainable_params = extractor.get_trainable_params()
    print(f"可訓練參數數量: {sum(p.numel() for p in trainable_params):,}")
    print(f"總參數數量: {model.count_parameters():,}")
    print("凍結了大部分參數，只需要少量新建築資料即可微調")

    # 方法 2: MAML (元學習)
    print("\n--- MAML 元學習方法 ---")
    maml = MAML(model, inner_lr=0.01, n_inner_steps=5)
    print("MAML 可以用 5-10 個樣本快速適應新建築類型")

    return extractor


# =============================================================================
# 8. 持續學習
# =============================================================================
def demo_8_continual_learning(model, config):
    """
    示範 8: 持續學習 - 線上更新不遺忘
    """
    print("\n" + "="*60)
    print("示範 8: 持續學習")
    print("="*60)

    from ml.ranking_v2 import (
        ContinualLearner,
        ExperienceReplayBuffer,
        EWC,
        create_continual_learner,
    )

    # 創建持續學習器
    learner = create_continual_learner(
        model=model,
        config=config,
        strategy="replay_ewc",  # 結合經驗回放和 EWC
        replay_buffer_size=5000,
        ewc_lambda=100.0,
        replay_ratio=0.3,
        device=device,
    )

    print(f"學習策略: replay_ewc")
    print(f"經驗回放緩衝區大小: 5000")
    print(f"EWC 正則化強度: 100.0")

    # 查看狀態
    state = learner.get_state()
    print(f"\n當前狀態:")
    print(f"  已學習任務: {state.tasks_seen}")
    print(f"  總樣本數: {state.total_samples_seen}")

    # 模擬線上學習
    print("\n線上學習流程:")
    print("  1. 收到新建築資料 → 加入經驗回放緩衝區")
    print("  2. 訓練時混合新舊資料 → 防止遺忘")
    print("  3. EWC 保護重要參數 → 保持舊知識")

    return learner


# =============================================================================
# 9. GNN 編碼器
# =============================================================================
def demo_9_gnn_encoder(config):
    """
    示範 9: 圖神經網路編碼器
    """
    print("\n" + "="*60)
    print("示範 9: GNN 編碼器")
    print("="*60)

    from ml.ranking_v2 import (
        GNNEncoder,
        HybridEncoder,
        FloorPlanEncoder,
        create_gnn_encoder,
    )

    # 創建 GNN 編碼器
    gnn_encoder = create_gnn_encoder(
        config_or_latent_dim=64,
        gnn_type='gatv2',  # 或 'sage', 'gin'
        num_layers=3,
        heads=4,
    ).to(device)

    print(f"GNN 類型: GATv2")
    print(f"層數: 3")
    print(f"注意力頭數: 4")

    # 測試
    data = create_demo_data(batch_size=2)

    # 注意: GNN 編碼器會將柵格轉換為圖
    print("\n處理流程:")
    print("  1. 柵格圖 → 識別區域 (房間, 走廊, 門, 出口)")
    print("  2. 建立圖結構 → 節點=區域, 邊=連通性")
    print("  3. GNN 傳遞訊息 → 學習拓撲特徵")
    print("  4. 圖池化 → 輸出固定維度向量")

    # 混合編碼器 (CNN + GNN)
    print("\n--- 混合編碼器 (CNN + GNN) ---")
    cnn_encoder = FloorPlanEncoder(config).to(device)

    hybrid = HybridEncoder(
        cnn_encoder=cnn_encoder,
        fusion='attention',  # 或 'concat', 'add'
        latent_dim=64,
    ).to(device)

    print("混合編碼器結合 CNN 的局部紋理特徵和 GNN 的全域拓撲特徵")

    return gnn_encoder


# =============================================================================
# 10. 對比學習預訓練
# =============================================================================
def demo_10_contrastive_learning(config):
    """
    示範 10: 對比學習預訓練
    """
    print("\n" + "="*60)
    print("示範 10: 對比學習預訓練")
    print("="*60)

    from ml.ranking_v2 import (
        SimCLRModel,
        MoCoModel,
        ContrastivePretrainer,
        create_contrastive_model,
        transfer_pretrained_encoder,
        CrossAttentionRanker,
    )

    # 創建 SimCLR 模型
    simclr = create_contrastive_model(
        config=config,
        method='simclr',
        projection_dim=128,
        temperature=0.5,
    ).to(device)

    print("SimCLR 自監督學習:")
    print("  1. 同一樓層圖 → 兩種不同增強 → 正樣本對")
    print("  2. 不同樓層圖 → 負樣本對")
    print("  3. 對比損失 → 學習有意義的表示")

    # 模擬預訓練
    data = create_demo_data(batch_size=4)

    with torch.no_grad():
        loss = simclr(data['grid_a'].to(device))
    print(f"\n對比損失: {loss.item():.4f}")

    # 預訓練後遷移
    print("\n遷移到排序任務:")
    pretrained_encoder = simclr.encoder
    target_model = CrossAttentionRanker(config).to(device)

    target_model = transfer_pretrained_encoder(
        pretrained_encoder=pretrained_encoder,
        target_model=target_model,
        freeze_encoder=True,  # 可選: 凍結編碼器
    )

    print("  ✓ 已將預訓練編碼器遷移到排序模型")

    return simclr


# =============================================================================
# 主程式
# =============================================================================
DEMO_REGISTRY = {
    1: ("基礎模型", demo_1_basic_model),
    2: ("不確定性量化", demo_2_uncertainty),
    3: ("配置生成優化", demo_3_config_generation),
    4: ("可解釋性報告", demo_4_explainability),
    5: ("主動學習", demo_5_active_learning),
    6: ("多目標排序", demo_6_multi_objective),
    7: ("遷移學習", demo_7_transfer_learning),
    8: ("持續學習", demo_8_continual_learning),
    9: ("GNN 編碼器", demo_9_gnn_encoder),
    10: ("對比學習預訓練", demo_10_contrastive_learning),
}


def list_demos():
    """列出所有可用的示範"""
    print("\n可用的示範:")
    print("-" * 40)
    for num, (name, _) in DEMO_REGISTRY.items():
        print(f"  {num:2d}. {name}")
    print("-" * 40)
    print("\n使用方式:")
    print("  # 執行特定示範")
    print("  python -m ml.ranking_v2.examples.usage_guide --demos 1,4")
    print()
    print("  # 使用訓練好的模型")
    print("  python -m ml.ranking_v2.examples.usage_guide -c model.pt --demos 4")
    print()


def main():
    """執行示範"""
    # 如果只是列出示範
    if CONFIG['list_demos']:
        list_demos()
        return

    print("="*60)
    print("Ranking V2.1 完整使用指南")
    print("="*60)

    # 決定要執行哪些示範
    demos_to_run = CONFIG['demos'] or list(DEMO_REGISTRY.keys())

    print(f"\n將執行的示範: {demos_to_run}")

    # 示範 1 是必須的（載入模型）
    if 1 not in demos_to_run:
        demos_to_run = [1] + demos_to_run

    model = None
    config = None
    results = {}

    for demo_num in demos_to_run:
        if demo_num not in DEMO_REGISTRY:
            print(f"\n警告: 示範 {demo_num} 不存在，跳過")
            continue

        name, func = DEMO_REGISTRY[demo_num]

        try:
            # 根據示範編號決定參數
            if demo_num == 1:
                model, config = func()
                results[demo_num] = (model, config)
            elif demo_num == 2:
                results[demo_num] = func(model)
            elif demo_num == 3:
                results[demo_num] = func(model)
            elif demo_num == 4:
                results[demo_num] = func(model, config)
            elif demo_num == 5:
                results[demo_num] = func(model, config)
            elif demo_num == 6:
                results[demo_num] = func(config)
            elif demo_num == 7:
                results[demo_num] = func(model, config)
            elif demo_num == 8:
                results[demo_num] = func(model, config)
            elif demo_num == 9:
                results[demo_num] = func(config)
            elif demo_num == 10:
                results[demo_num] = func(config)

        except Exception as e:
            print(f"\n錯誤: 示範 {demo_num} ({name}) 執行失敗: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("示範執行完成！")
    print("="*60)

    print("\n快速參考:")
    print("  - 不確定性: MCDropoutWrapper, DeepEnsemble")
    print("  - 配置優化: EvolutionaryOptimizer, MCTSOptimizer")
    print("  - 解釋性: ExplanationPipeline")
    print("  - 主動學習: ActiveLearningLoop")
    print("  - 多目標: MultiObjectiveRanker")
    print("  - 遷移: TransferLearningPipeline, MAML")
    print("  - 持續學習: ContinualLearner, EWC")
    print("  - GNN: GNNEncoder, HybridEncoder")
    print("  - 預訓練: SimCLRModel, MoCoModel")

    return results


if __name__ == "__main__":
    main()
