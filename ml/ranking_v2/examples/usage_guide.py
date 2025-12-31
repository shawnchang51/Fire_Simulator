"""
Ranking V2.1 使用指南 - 完整範例

本指南展示如何使用所有新增的增強功能。
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用裝置: {device}")


# =============================================================================
# 1. 基礎設定 - 創建模型和假資料
# =============================================================================
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


def demo_1_basic_model():
    """
    示範 1: 基礎模型使用
    """
    print("\n" + "="*60)
    print("示範 1: 基礎 Ranking V2 模型")
    print("="*60)

    from ml.ranking_v2 import RankingV2Config, CrossAttentionRanker, get_full_config

    # 方法 1: 使用預設配置
    config = get_full_config()

    # 方法 2: 自訂配置
    config = RankingV2Config(
        latent_dim=64,
        use_cross_attention=True,
        attention_heads=4,
        mining_strategy="curriculum",
        auxiliary_tasks=["survival_rate", "steps"],
    )

    # 創建模型
    model = CrossAttentionRanker(config).to(device)
    print(f"模型參數量: {model.count_parameters():,}")

    # 準備資料
    data = create_demo_data()

    # 前向傳播
    with torch.no_grad():
        outputs = model(
            data['grid_a'].to(device),
            data['scenario_a'].to(device),
            data['grid_b'].to(device),
            data['scenario_b'].to(device),
        )

    print(f"預測分數差 (logit): {outputs['logit']}")
    print(f"配置 A 分數: {outputs['score_a']}")
    print(f"配置 B 分數: {outputs['score_b']}")

    # 單一配置評分
    single_score = model.score_single(
        data['grid_a'].to(device),
        data['scenario_a'].to(device),
    )
    print(f"單一配置評分: {single_score}")

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
def main():
    """執行所有示範"""
    print("="*60)
    print("Ranking V2.1 完整使用指南")
    print("="*60)

    # 1. 基礎模型
    model, config = demo_1_basic_model()

    # 2. 不確定性量化
    mc_model = demo_2_uncertainty(model)

    # 3. 配置生成
    gen_result = demo_3_config_generation(model)

    # 4. 可解釋性
    report = demo_4_explainability(model, config)

    # 5. 主動學習
    al_loop = demo_5_active_learning(model, config)

    # 6. 多目標排序
    mo_ranker = demo_6_multi_objective(config)

    # 7. 遷移學習
    extractor = demo_7_transfer_learning(model, config)

    # 8. 持續學習
    learner = demo_8_continual_learning(model, config)

    # 9. GNN 編碼器
    gnn = demo_9_gnn_encoder(config)

    # 10. 對比學習
    simclr = demo_10_contrastive_learning(config)

    print("\n" + "="*60)
    print("所有示範完成！")
    print("="*60)

    print("\n📚 快速參考:")
    print("  - 不確定性: MCDropoutWrapper, DeepEnsemble")
    print("  - 配置優化: EvolutionaryOptimizer, MCTSOptimizer")
    print("  - 解釋性: ExplanationPipeline")
    print("  - 主動學習: ActiveLearningLoop")
    print("  - 多目標: MultiObjectiveRanker")
    print("  - 遷移: TransferLearningPipeline, MAML")
    print("  - 持續學習: ContinualLearner, EWC")
    print("  - GNN: GNNEncoder, HybridEncoder")
    print("  - 預訓練: SimCLRModel, MoCoModel")


if __name__ == "__main__":
    main()
