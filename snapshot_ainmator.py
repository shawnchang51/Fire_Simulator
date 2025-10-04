import time
from typing import List, Optional, Iterable, Union

import numpy as np
import matplotlib.pyplot as plt

# try optional import for interpolation
try:
    from scipy.interpolate import RegularGridInterpolator
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

class RealtimeGridAnimator:
    """
    即時 Grid 動畫工具。
    使用方式：
        anim = RealtimeGridAnimator(initial_grid, interp=True, interp_shape=(200,200), interval=0.05)
        for each step:
            anim.update(new_grid)
        # 或使用 run(generator)
    參數:
      - initial_grid: List[List[float]] or np.ndarray（用於初始化畫面）
      - interp: 是否對每個 input grid 做空間插值（需要 scipy）
      - interp_shape: 插值後的畫素大小 (rows, cols)
      - cmap: colormap 名稱
      - contour_levels: 等高線層數，若為 0 則不畫等高線
      - interval: 每次更新後的 pause 秒數（run(generator) 或 update 內部會使用）
      - vmin, vmax: colorbar 範圍（None 表示自動）
    """
    def __init__(
        self,
        initial_grid: Union[List[List[float]], np.ndarray],
        interp: bool = False,
        interp_shape: tuple = (200, 200),
        cmap: str = "viridis",
        contour_levels: int = 8,
        interval: float = 0.05,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        figsize: tuple = (6, 5),
    ):
        self.interp = interp and _HAS_SCIPY
        if interp and not _HAS_SCIPY:
            print("Warning: scipy not available — 將使用不插值的即時顯示。若要插值請安裝 scipy.")
        self.interp_shape = tuple(interp_shape)
        self.cmap = cmap
        self.contour_levels = contour_levels
        self.interval = interval
        self.vmin = vmin
        self.vmax = vmax

        # internal state
        self.paused = False
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._contours = None

        # initialize with given grid
        z0 = np.array(initial_grid)
        self._rows, self._cols = z0.shape

        # setup interpolation grid if needed
        if self.interp:
            self._setup_interp_coords()

            Z_init_disp = self._interp_array(z0)
            self._disp_shape = Z_init_disp.shape
        else:
            Z_init_disp = z0
            self._disp_shape = Z_init_disp.shape

        # image and colorbar
        self.im = self.ax.imshow(
            Z_init_disp,
            origin="lower",
            aspect="auto",
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            extent=(0, self._cols - 1, 0, self._rows - 1),
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax)
        self.cbar.set_label("Value")

        # initial contours
        if self.contour_levels and self.contour_levels > 0:
            self._contours = self.ax.contour(
                Z_init_disp,
                levels=self.contour_levels,
                colors="black",
                linewidths=0.8,
                origin="lower",
                extent=(0, self._cols - 1, 0, self._rows - 1),
            )

        # connect key events for pause/resume
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.ion()
        plt.show()

    def _setup_interp_coords(self):
        # source coords (row, col)
        self._src_y = np.arange(self._rows)
        self._src_x = np.arange(self._cols)
        # target coords (for display)
        yi = np.linspace(0, self._rows - 1, self.interp_shape[0])
        xi = np.linspace(0, self._cols - 1, self.interp_shape[1])
        self._interp_XI, self._interp_YI = np.meshgrid(xi, yi)

    def _interp_array(self, arr: np.ndarray) -> np.ndarray:
        """對 arr 做 RegularGridInterpolator 2D 插值並回傳新 array"""
        if not self.interp:
            return arr
        # build interpolator each call (fast enough for moderate grids); faster: cache if shape constant
        interp = RegularGridInterpolator((self._src_y, self._src_x), arr, method="linear", bounds_error=False, fill_value=np.nan)
        pts = np.stack([self._interp_YI.ravel(), self._interp_XI.ravel()], axis=-1)
        Zi = interp(pts).reshape(self.interp_shape)
        # 若 nan 出現（out-of-bounds），可以用 nearest 或填 0，這裡用 nan->0
        Zi = np.nan_to_num(Zi, nan=0.0)
        return Zi

    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
            state = "paused" if self.paused else "resumed"
            print(f"[RealtimeGridAnimator] {state}")

    def update(self, grid: Union[List[List[float]], np.ndarray]):
        """即時更新畫面。呼叫者每次模擬完成一個 step 就呼叫此方法並傳入新的 grid。"""
        if not plt.fignum_exists(self.fig.number):
            # figure 已關閉 -> 停止
            raise RuntimeError("Figure closed")

        if self.paused:
            # 當 paused 時不更新，但仍讓 GUI 有反應
            plt.pause(0.001)
            return

        Z = np.array(grid)
        # if grid shape changed, update source coords & extent
        if Z.shape != (self._rows, self._cols):
            self._rows, self._cols = Z.shape
            if self.interp:
                self._setup_interp_coords()
            self.im.set_extent((0, self._cols - 1, 0, self._rows - 1))

        # maybe interpolate
        if self.interp:
            Z_disp = self._interp_array(Z)
        else:
            Z_disp = Z

        # update color scale dynamically if vmin/vmax not set
        if self.vmin is None or self.vmax is None:
            self.im.set_clim(np.nanmin(Z_disp), np.nanmax(Z_disp))

        # set image data
        self.im.set_data(Z_disp)

        # update contours: remove old then draw new
        if self.contour_levels and self.contour_levels > 0:
            if self._contours is not None:
                try:
                    for coll in self._contours.collections:
                        coll.remove()
                except Exception:
                    pass
            self._contours = self.ax.contour(
                Z_disp,
                levels=self.contour_levels,
                colors="black",
                linewidths=0.8,
                origin="lower",
                extent=(0, self._cols - 1, 0, self._rows - 1),
            )

        # force redraw
        self.fig.canvas.draw_idle()
        # allow GUI event loop to run
        plt.pause(self.interval)

    def run(self, generator: Iterable[Union[List[List[float]], np.ndarray]]):
        """
        便利方法：給一個產生器（或可迭代物件）就會自動在畫面上依序顯示每個 grid。
        支援在任何時候關閉視窗或按空白鍵暫停/恢復。
        """
        try:
            for grid in generator:
                if not plt.fignum_exists(self.fig.number):
                    break
                while self.paused:
                    # 當 paused 時每 0.05 秒檢查一次
                    plt.pause(0.05)
                    if not plt.fignum_exists(self.fig.number):
                        break
                self.update(grid)
        except RuntimeError:
            # figure closed
            pass
