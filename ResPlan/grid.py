# 整理後的完整程式 — 可直接貼上執行
# 確保 resplan_utils.py 可被找到（如果在不同工作目錄，請改 DATA_PATH 或把路徑加入 sys.path）
import sys
import os
import pickle
import random

# 若 resplan_utils.py 放在 /mnt/data，就把它加入 path；否則視情況修改
sys.path.append('/mnt/data')

# 套件
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

# 嘗試載入 resplan_utils（如果找不到，會顯示友善錯誤）
try:
    from resplan_utils import (
        CATEGORY_COLORS,
        normalize_keys, get_plan_width,
        get_geometries, centroid,
        geometry_to_mask,
        augment_geom,
        buffer_shrink_expand, buffer_expand_shrink,
        plot_plan,
        plan_to_graph, plot_plan_and_graph,
        get_structural_plan, structural_plan_to_multilinestring,
        shrink_short_side, multilinestring_to_grid
    )
except Exception as e:
    raise ImportError(
        "無法匯入 resplan_utils。請確認 resplan_utils.py 在 python path 中，"
        "或把它放在 /mnt/data，或修改 sys.path.append(...) 的路徑。\n原始錯誤: " + str(e)
    )

plt.rcParams['figure.dpi'] = 110

# ---------------------------
# 使用者參數（可修改）
# ---------------------------
DATA_PATH = 'ResPlan.pkl'   # ResPlan 檔案路徑（相對或絕對）
CELL_SIZE_M = 0.3           # 網格 cell size（公尺）
FILTER_BRANCH_ENDPOINTS = True
MAX_BRANCH_LENGTH = 2.0
RNG_SEED = None             # 若要固定隨機結果，設定為 int（例如 42），否則設 None 以隨機挑選

# ---------------------------
# 讀檔與前處理
# ---------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"找不到資料檔案：{DATA_PATH}。請確認路徑正確。")

with open(DATA_PATH, 'rb') as f:
    plans = pickle.load(f)

if not isinstance(plans, (list, tuple)) or len(plans) == 0:
    raise ValueError("載入的 ResPlan 資料格式不正確或為空。期待一個含多個 plan 的 list。")

print(f"已載入 {len(plans)} 個 plans。")

# 正規化 key（安全處理）
for p in plans:
    try:
        normalize_keys(p)
    except Exception:
        # 如果某些 plan 在 normalize 時出錯，不要終止整個流程，僅警告
        print("警告：normalize_keys 在某 plan 上失敗（略過該 step）")

# ---------------------------
# 選一個 sample plan
# ---------------------------
if RNG_SEED is not None:
    random.seed(RNG_SEED)
sample_idx = random.randrange(len(plans))
sample_plan = plans[sample_idx]
print(f"選取範例 Plan #{sample_idx}")

# ---------------------------
# 產生 MultiLineString 與 Raster grid
# ---------------------------
# 1) 抽取 structural elements（wall/door/window/front_door）
structural = get_structural_plan(sample_plan)

# 2) polygon -> MultiLineString（中心線或向量化）
mls = structural_plan_to_multilinestring(sample_plan)
n_lines = len(mls.geoms) if hasattr(mls, 'geoms') else 1
print(f"MultiLineString 含 {n_lines} 條線段")

# 3) MultiLineString -> grid（rasterize）
grid = multilinestring_to_grid(
    mls,
    sample_plan,
    cell_size_m=CELL_SIZE_M,
    filter_branch_endpoints=FILTER_BRANCH_ENDPOINTS,
    max_branch_length=MAX_BRANCH_LENGTH
)

# ---------------------------
# 繪圖：1x3 比較圖
# ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (A) 原始 structural plan（多邊形）
ax0 = axes[0]
plot_plan(
    structural,
    categories=["wall", "door", "window", "front_door"],
    title=f'Original Plan #{sample_idx}',
    ax=ax0,
    legend=False
)
ax0.invert_yaxis()
ax0.set_aspect('equal')

# (B) MultiLineString（向量）
ax1 = axes[1]
# 使用 geopandas 畫出多線（如果 mls 不是 GeoSeries，包一層）
gpd.GeoSeries([mls]).plot(ax=ax1, color='blue', linewidth=0.6)
ax1.invert_yaxis()
ax1.set_aspect('equal')
ax1.set_axis_off()
ax1.set_title('MultiLineString (vectorized)')

# (C) Rasterized grid（以 -2 作為 structural 的標記）
ax2 = axes[2]
display_grid = np.where(grid == -2, 1, 0)
ax2.imshow(display_grid, cmap='Greys', interpolation='nearest', origin='lower')
ax2.set_title(f'Grid (cell={CELL_SIZE_M} m)\nShape: {grid.shape}')
ax2.set_aspect('equal')
ax2.axis('off')

plt.suptitle('Structural Plan → MultiLineString → Grid (side-by-side)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# ---------------------------
# Grid 統計
# ---------------------------
total_cells = grid.size
structural_cells = int(np.sum(grid == -2))
passable_cells = int(np.sum(grid == 0))
coverage = structural_cells / total_cells * 100.0

print("\nGrid statistics:")
print(f"  cell_size = {CELL_SIZE_M} m")
print(f"  Shape: {grid.shape}")
print(f"  Structural cells (-2): {structural_cells}")
print(f"  Passable cells (0): {passable_cells}")
print(f"  Coverage: {coverage:.1f}%")

# 額外：若想把 grid 轉成圖像檔（PNG），可解除下列註解並指定檔名：
# out_png = f'grid_plan_{sample_idx}_cell{CELL_SIZE_M:.2f}m.png'
# plt.imsave(out_png, display_grid, cmap='Greys', origin='lower')
# print(f"已儲存網格影像到 {out_png}")
