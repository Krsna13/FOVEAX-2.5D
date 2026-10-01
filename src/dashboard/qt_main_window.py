import math
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QGroupBox, QListWidget, QProgressBar,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QFrame, QSizePolicy, QHeaderView, QSpinBox, QDialog, QFileDialog, QLineEdit, QDoubleSpinBox, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush
from src.dashboard.dashboard_state import FrameState
from src.dashboard.win32_embed import resize_child
from src.dashboard.track_labels import (
    track_distance_m,
    tracked_object_row_text,
    zone_object_counts,
)

# Real resolution-zone spec (src/07_adaptive_2point5d_map.py SPEC_ZONES_100M),
# quoted directly for the System Details drawer -- not re-derived or guessed.
RESOLUTION_ZONES = [
    ("Near", "0-10 m", "5 cm"),
    ("Middle", "10-30 m", "20 cm"),
    ("Far", "30-100 m", "50 cm"),
]

# Canonical traversability convention (src/06_terrain_traversability.py).
TRAVERSABILITY_SAFE_MIN = 0.70
TRAVERSABILITY_CAUTION_MIN = 0.40

# Number of consecutive frames a new Approaching/Receding/Crossing
# classification must persist before the DISPLAYED direction label
# actually changes (Bug 3 fix). Confirmed on real data: a real VEHICLE
# track's radial velocity oscillating within ~0.1 m/s of the existing
# +-0.2 m/s deadband boundary flipped the label 3 times in 20 frames
# without this. This does not change the underlying radial-velocity
# computation or the deadband itself -- only how quickly a label change
# is allowed to reach the screen.
DIRECTION_HYSTERESIS_FRAMES = 3

# Dashboard color tokens, matched to foveax_rough_dashboard.html's CSS
# custom properties (kept identical so the restyled Qt UI reads as the same
# visual system as the reference/web dashboards).
COLOR_BG = "#060B14"
COLOR_PANEL = "#0B1420"
COLOR_PANEL_ALT = "#0E1B2B"
COLOR_BORDER = "#16324A"
COLOR_TEXT = "#E2E8F0"
COLOR_TEXT_DIM = "#64748B"
COLOR_SAFE = "#34D399"
COLOR_CAUTION = "#FBBF24"
COLOR_CRITICAL = "#F43F5E"
COLOR_CYAN = "#22D3EE"
COLOR_DYNAMIC = "#A855F7"
COLOR_STATIC = "#FB923C"

# Shared typography scale, applied consistently across every widget added
# incrementally in separate tasks (confidence rings, hazard table,
# overhead legend, centerline profile, accuracy/efficiency/ego-motion
# cards) so they read as one coherent design rather than separately
# styled pieces:
#   NOTE      -- small dim explanatory/caveat text under a panel title.
#   BODY      -- default readable text (matches the base 11px in STYLESHEET).
#   EMPHASIS  -- a single inline number/word that should stand out a little
#                within its group (e.g. "Overall" accuracy).
#   HEADLINE  -- a card's one big real-time stat (e.g. road width).
#   HERO      -- the single most prominent status word on the whole
#                dashboard (ego terrain status) -- intentionally the
#                largest, there is exactly one of these per screen.
FONT_SIZE_NOTE = "9px"
FONT_SIZE_BODY = "11px"
FONT_SIZE_EMPHASIS = "12px"
FONT_SIZE_HEADLINE = "16px"
FONT_SIZE_HERO = "18px"

# Shared style string for small dim explanatory/caveat labels -- applied
# identically everywhere this kind of text appears (previously varied:
# some had this exact rule, some had no font-size at all).
NOTE_STYLE = f"color: {COLOR_TEXT_DIM}; font-size: {FONT_SIZE_NOTE};"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    font-family: 'Segoe UI', Inter, sans-serif;
    font-size: 11px;
}}
QGroupBox {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    margin-top: 14px;
    font-weight: 600;
    color: {COLOR_CYAN};
    padding: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}
QLabel {{ color: {COLOR_TEXT}; }}
QPushButton {{
    background-color: {COLOR_PANEL_ALT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    color: {COLOR_TEXT};
}}
QPushButton:hover {{ border: 1px solid {COLOR_CYAN}; }}
QPushButton:checked {{
    background-color: {COLOR_CYAN};
    color: #041019;
    font-weight: 600;
}}
QListWidget, QTableWidget, QComboBox {{
    background-color: {COLOR_PANEL_ALT};
    border: 1px solid {COLOR_BORDER};
    color: {COLOR_TEXT};
    gridline-color: {COLOR_BORDER};
}}
QHeaderView::section {{
    background-color: {COLOR_PANEL};
    color: {COLOR_TEXT_DIM};
    border: 1px solid {COLOR_BORDER};
    padding: 3px;
}}
QTabWidget::pane {{ border: 1px solid {COLOR_BORDER}; background: {COLOR_PANEL}; }}
QTabBar::tab {{
    background: {COLOR_PANEL_ALT};
    color: {COLOR_TEXT_DIM};
    padding: 6px 14px;
    border: 1px solid {COLOR_BORDER};
}}
QTabBar::tab:selected {{ color: {COLOR_CYAN}; background: {COLOR_PANEL}; font-weight: 600; }}
QProgressBar {{
    background-color: {COLOR_PANEL_ALT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 3px;
    text-align: center;
    color: {COLOR_TEXT};
}}
QProgressBar::chunk {{ background-color: {COLOR_CYAN}; }}
"""


class Open3DEmbedWidget(QWidget):
    """Placeholder panel that hosts the reparented Open3D native window.

    This widget itself draws nothing -- it exists only to (a) provide a
    real native HWND for src/13_realtime_dashboard.py to reparent the
    Open3D child process's window into, and (b) forward Qt resize events
    to that reparented window via win32gui.MoveWindow, since a foreign
    HWND does not automatically follow its new Qt "parent" widget's size.

    `child_hwnd` is set externally once the reparenting handshake in
    DashboardApplication succeeds; until then (or if it never succeeds,
    see the fallback path there) this is just an empty black panel and the
    Open3D window shows separately, exactly as before this feature existed.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.child_hwnd = None
        # Forces Qt to back this widget with a real native window (HWND) up
        # front, rather than the "alien widget" optimization Qt normally
        # uses for plain child widgets on Windows -- SetParent needs a real
        # HWND to attach to.
        self.setAttribute(Qt.WA_NativeWindow, True)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(300, 300)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.child_hwnd:
            resize_child(self.child_hwnd, self.width(), self.height())

    # Box colors from open3d_viewer.py: blue for Dynamic, purple for Static.
    _LABEL_COLORS = {True: "#0080ff", False: "#9933cc"}

    def set_track_labels(self, labels) -> None:
        """Place one class label above each tracked box.

        `labels` are TrackLabel records projected by the Open3D process from
        the same TrackState list its boxes were drawn from. Each label is its
        own native window: the reparented OpenGL window is a native child
        HWND, and non-native Qt widgets cannot paint over it.

        Called every frame, so each widget remembers what it last showed and
        only the properties that actually changed are touched (restyling,
        resizing and re-raising a native window are the expensive calls).
        """
        if not hasattr(self, "_label_widgets"):
            self._label_widgets = {}
            self._label_state = {}
            self._label_pool = []  # hidden, reusable label windows

        seen = set()
        for lab in labels:
            seen.add(lab.track_id)
            widget = self._label_widgets.get(lab.track_id)
            if widget is None:
                # Creating a native window is far costlier than reusing one, and
                # tracks come and go constantly as the vehicle moves.
                if self._label_pool:
                    widget = self._label_pool.pop()
                else:
                    widget = QLabel(self)
                    widget.setAttribute(Qt.WA_NativeWindow, True)
                    widget.setAttribute(Qt.WA_TransparentForMouseEvents, True)
                    widget.setAlignment(Qt.AlignCenter)
                self._label_widgets[lab.track_id] = widget
                self._label_state[lab.track_id] = {
                    "dynamic": None, "text": None, "pos": None, "shown": False, "size": None}
            st = self._label_state[lab.track_id]

            if not lab.visible:
                if st["shown"]:
                    widget.hide()
                    st["shown"] = False
                continue

            if st["dynamic"] != lab.dynamic:
                color = self._LABEL_COLORS[lab.dynamic]
                widget.setStyleSheet(
                    f"background-color: rgba(10,10,20,215); color: white; "
                    f"border: 1px solid {color}; border-radius: 3px; "
                    f"padding: 1px 4px; font-size: 10px; font-weight: bold;"
                )
                st["dynamic"] = lab.dynamic
                st["text"] = None  # restyling can change the size; force a resize
                st["size"] = None
            if st["text"] != lab.text:
                widget.setText(lab.text)
                # The distance in the text changes nearly every frame; resizing a
                # native window each time is costly, so the width is rounded up to
                # 8 px and the window is resized only when that rounded size changes.
                hint = widget.sizeHint()
                size = (-(-hint.width() // 8) * 8, hint.height())
                if st["size"] != size:
                    widget.resize(*size)
                    st["size"] = size
                    st["pos"] = None
                st["text"] = lab.text

            # The Open3D framebuffer can differ in size from this panel mid-resize.
            sx = self.width() / max(1, lab.view_width)
            sy = self.height() / max(1, lab.view_height)
            x = int(round(lab.u * sx - widget.width() / 2))
            y = int(round(lab.v * sy - widget.height() - 4))
            if st["pos"] != (x, y):
                widget.move(x, y)
                st["pos"] = (x, y)
            if not st["shown"]:
                widget.show()
                widget.raise_()
                st["shown"] = True

        for track_id in list(self._label_widgets):
            if track_id not in seen:
                gone = self._label_widgets.pop(track_id)
                gone.hide()
                self._label_pool.append(gone)
                self._label_state.pop(track_id, None)

    def visible_label_texts(self) -> dict[int, str]:
        """track_id -> text for labels currently shown."""
        return {
            tid: w.text()
            for tid, w in getattr(self, "_label_widgets", {}).items()
            if w.isVisible()
        }


class ElevationProfileWidget(QWidget):
    """Real elevation profile (z_max along the ego-forward centerline) with
    real hazard markers where traversability < 0.40 at the same column.

    No synthetic curve, no scripted pothole positions -- both the line and
    the markers are drawn directly from FrameState.elevation_profile /
    elevation_hazards_m, which src/dashboard/data_streamer.py computes from
    the real local grid each frame.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.profile: list[tuple[float, float]] = []
        self.hazards_m: list[float] = []

    def set_data(self, profile, hazards_m):
        self.profile = profile
        self.hazards_m = hazards_m
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COLOR_PANEL_ALT))
        w, h = self.width(), self.height()

        if not self.profile:
            painter.setPen(QColor(COLOR_TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignCenter, "No elevation data")
            return

        ys = [p[0] for p in self.profile]
        zs = [p[1] for p in self.profile]
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)
        y_span = max(y_max - y_min, 1e-6)
        z_span = max(z_max - z_min, 1e-6)

        def to_px(y_val, z_val):
            px = (y_val - y_min) / y_span * (w - 10) + 5
            py = h - 15 - (z_val - z_min) / z_span * (h - 25)
            return px, py

        painter.setPen(QPen(QColor(COLOR_CYAN), 1))
        prev = None
        for y_val, z_val in self.profile:
            px, py = to_px(y_val, z_val)
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(px), int(py))
            prev = (px, py)

        painter.setPen(QPen(QColor(COLOR_CRITICAL), 2))
        painter.setBrush(QColor(COLOR_CRITICAL))
        for hz_y in self.hazards_m:
            closest = min(self.profile, key=lambda p: abs(p[0] - hz_y))
            px, py = to_px(closest[0], closest[1])
            painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)

        painter.setPen(QColor(COLOR_TEXT_DIM))
        painter.drawText(5, h - 3, f"{y_min:.0f}m")
        painter.drawText(w - 40, h - 3, f"{y_max:.0f}m")


# A bin needs at least this many real points backing it to be drawn at
# full confidence; sparser real bins are still drawn (never hidden), just
# with a lighter/dashed visual treatment so they aren't presented with
# the same confidence as well-observed bins.
LOW_CONFIDENCE_POINT_COUNT = 5


_LUT8_CACHE: dict[str, np.ndarray] = {}


def _colormap_lut8(name: str) -> np.ndarray:
    """(256, 4) uint8 table equal, entry for entry, to `(cmap(x) * 255).astype(uint8)`
    for the float x that matplotlib maps to that entry. Indexing it with
    `min(int(x * 256), 255)` reproduces the colormap call without its overhead."""
    lut = _LUT8_CACHE.get(name)
    if lut is None:
        cmap = mpl.colormaps[name]
        lut = (cmap((np.arange(256) + 0.5) / 256.0) * 255).astype(np.uint8)
        _LUT8_CACHE[name] = lut
    return lut


def grid_to_rgba8(grid: np.ndarray, map_type: str):
    """RGBA8 image (h, w, 4) of one 2D map layer, or None when the grid has no
    valid cell. Pure numpy/cv2 (no Qt) so it can be tested pixel for pixel."""
    valid_mask = ~np.isnan(grid)
    if not np.any(valid_mask):
        return None

    # Normalize for colormap
    g_min, g_max = np.nanmin(grid), np.nanmax(grid)
    if g_max > g_min:
        norm_grid = (grid - g_min) / (g_max - g_min)
    else:
        norm_grid = np.zeros_like(grid)

    norm_grid[~valid_mask] = 0.0  # background

    # Apply colormap through a 256-entry table (identical pixels to calling the
    # matplotlib colormap on the float grid, minus its per-call overhead)
    cmap_name = {"elevation": "turbo", "traversability": "RdYlGn",  # green = high = safe
                 "roi": "magma", "uncertainty": "cividis"}.get(map_type, "viridis")
    rgba_8 = _colormap_lut8(cmap_name)[np.minimum((norm_grid * 256).astype(np.int32), 255)]
    rgba_8[~valid_mask, 3] = 0  # transparent for nan

    # Add hatch pattern for traversability lethal cells (Accessibility - Task R)
    if map_type == "traversability":
        try:
            import cv2
            # Blocked/lethal is traversability score < 0.40, matching canonical convention
            # (06_terrain_traversability.py, grid_overlay.py): 1.0 = safe, 0.0 = blocked
            lethal_mask = (grid < 0.40) & valid_mask
            if np.any(lethal_mask):
                no_hatch = rgba_8.copy()
                for i in range(0, rgba_8.shape[0] + rgba_8.shape[1], 5):
                    cv2.line(rgba_8, (0, i), (i, 0), (0, 0, 0, 255), 1)
                # keep the hatch only where the cell is lethal
                rgba_8[~lethal_mask] = no_hatch[~lethal_mask]
        except ImportError:
            pass
    return rgba_8


def overhead_to_rgba8(trav_grid, overhead_grid):
    """RGBA8 image of the OVERHEAD layer (see FoveaXDashboardWindow
    ._render_overhead_to_label), or None when there is no overhead grid."""
    from src.perception.overhead_detection import (
        EMPTY, GROUND, SOLID_OBSTACLE, OVERHEAD_OBSTACLE, INDETERMINATE,
    )

    if overhead_grid is None:
        return None

    h, w = overhead_grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    ground_mask = overhead_grid == GROUND
    if trav_grid is not None:
        valid_trav = ~np.isnan(trav_grid)
        norm = np.zeros_like(trav_grid, dtype=np.float32)
        norm[valid_trav] = np.clip(trav_grid[valid_trav], 0.0, 1.0)
        ground_rgba = _colormap_lut8("RdYlGn")[np.minimum((norm * 256).astype(np.int32), 255)]
        fillable_ground = ground_mask & valid_trav
        rgba[fillable_ground] = ground_rgba[fillable_ground]

    solid_mask = overhead_grid == SOLID_OBSTACLE
    rgba[solid_mask] = [251, 146, 60, 255]  # COLOR_STATIC

    overhead_mask = overhead_grid == OVERHEAD_OBSTACLE
    rgba[overhead_mask] = [34, 211, 238, 255]  # COLOR_CYAN base fill, hatched below

    indet_mask = overhead_grid == INDETERMINATE
    rgba[indet_mask] = [100, 100, 100, 255]

    rgba[overhead_grid == EMPTY] = [0, 0, 0, 0]

    # Cross-hatch pattern over overhead_obstacle cells only, using the same
    # diagonal-line technique as the lethal-traversability hatch.
    if np.any(overhead_mask):
        try:
            import cv2
            hatched = rgba.copy()
            hatch_color = (10, 10, 10, 255)
            spacing = 4
            for i in range(0, h + w, spacing):
                cv2.line(hatched, (0, i), (i, 0), hatch_color, 1)
            rgba = np.where(overhead_mask[:, :, None], hatched, rgba)
        except ImportError:
            pass
    return np.ascontiguousarray(rgba)


def _nice_ticks(lo: float, hi: float, target: int = 7) -> list[float]:
    """Round tick positions (1, 2, 2.5, 5 x 10^k) covering [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(1, target)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = mag * 10
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            step = m * mag
            break
    first = math.ceil(lo / step - 1e-9) * step
    ticks = []
    t = first
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


class CenterlineProfileChart(QWidget):
    """Real forward-corridor cross-section: real height above / depth
    below a real local ground baseline, from
    src/perception/centerline_profile.py, with real hazard/tracked-object
    markers at their real distances. A gap in the real data (no points in
    a bin) is drawn as a gap in the chart -- never interpolated across.

    Drawn directly with QPainter (about a millisecond per frame). The earlier
    matplotlib version rebuilt the whole figure and rasterised its text every
    frame, which cost tens of milliseconds and dominated the GUI frame time.
    Every frame redraws from exactly the data it is given.
    """

    _MARGIN_L, _MARGIN_R, _MARGIN_T, _MARGIN_B = 64, 12, 8, 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._bins: list[dict] = []
        self._markers: list[dict] = []
        self._bars: list[tuple] = []
        self._xlim = (0.0, 1.0)
        self._ylim = (-1.0, 1.0)

    def set_data(self, profile_bins: list[dict], markers: list[dict]):
        self._bins = list(profile_bins)
        self._markers = list(markers)
        self._prepare()
        self.update()

    def _prepare(self):
        # One entry per drawn bar: (distance, value, low_confidence, is_height).
        # Bins with no points are not drawn (never interpolated across).
        bars = []
        for b in self._bins:
            n = b["point_count"]
            if n == 0:
                continue
            low = n < LOW_CONFIDENCE_POINT_COUNT
            h_above = b["height_above_baseline_m"]
            if h_above is not None and h_above > 0.0:
                bars.append((b["distance_m"], h_above, low, True))
            d_below = b["depth_below_baseline_m"]
            if d_below is not None:
                bars.append((b["distance_m"], d_below, low, False))
        self._bars = bars

        xs = [d for d, _, _, _ in bars] + [m["distance_m"] for m in self._markers]
        if xs:
            lo, hi = min(xs) - 0.45, max(xs) + 0.45
            pad = 0.05 * (hi - lo)
            self._xlim = (lo - pad, hi + pad)
        else:
            self._xlim = (0.0, 1.0)
        vals = [0.0] + [v for _, v, _, _ in bars]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        # bars sit on the zero baseline, so no padding is added on that side
        self._ylim = (lo - (0.05 * span if lo < 0.0 else 0.0), hi + (0.05 * span if hi > 0.0 else 0.0))
        if self._ylim[1] <= self._ylim[0]:
            self._ylim = (self._ylim[0], self._ylim[0] + 1.0)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor(COLOR_PANEL))
        w, h = self.width(), self.height()
        L, R, T, B = self._MARGIN_L, self._MARGIN_R, self._MARGIN_T, self._MARGIN_B
        pw, ph = w - L - R, h - T - B
        if pw < 20 or ph < 20:
            return
        plot = QRectF(L, T, pw, ph)
        p.fillRect(plot, QColor(COLOR_PANEL_ALT))

        dim = QColor(COLOR_TEXT_DIM)
        small = QFont(self.font())
        small.setPixelSize(10)
        p.setFont(small)

        if not self._bins:
            p.setPen(dim)
            p.drawText(plot, Qt.AlignCenter, "No real corridor data for this frame")
            p.setPen(QPen(QColor(COLOR_BORDER), 1))
            p.drawRect(plot)
            return

        x0, x1 = self._xlim
        y0, y1 = self._ylim

        def px(x):
            return L + (x - x0) / (x1 - x0) * pw

        def py(y):
            return T + (1.0 - (y - y0) / (y1 - y0)) * ph

        # ticks
        p.setPen(dim)
        fm = p.fontMetrics()
        for t in _nice_ticks(x0, x1, 8):
            x = px(t)
            p.drawLine(QPointF(x, T + ph), QPointF(x, T + ph + 3))
            label = f"{t:g}"
            p.drawText(QPointF(x - fm.horizontalAdvance(label) / 2, T + ph + 14), label)
        for t in _nice_ticks(y0, y1, 5):
            y = py(t)
            p.drawLine(QPointF(L - 3, y), QPointF(L, y))
            label = f"{t:g}"
            p.drawText(QPointF(L - 6 - fm.horizontalAdvance(label), y + 3.5), label)
        p.drawText(QRectF(L, h - 16, pw, 14), Qt.AlignCenter, "Distance ahead (m)")
        p.save()
        p.translate(11, T + ph / 2)
        p.rotate(-90)
        p.drawText(QRectF(-ph / 2, -6, ph, 14), Qt.AlignCenter, "Height rel. baseline (m)")
        p.restore()

        # clip to the plot so nothing can spill over the axes
        p.setClipRect(plot)

        # zero baseline
        p.setPen(QPen(dim, 1))
        p.drawLine(QPointF(L, py(0.0)), QPointF(L + pw, py(0.0)))

        # bars (width 0.9 m, centred on each bin), hatched when low confidence
        edge = QPen(QColor(COLOR_BORDER), 0.5)
        for d, v, low, is_height in self._bars:
            base = QColor(COLOR_STATIC if is_height else COLOR_CRITICAL)
            fill = QColor(base)
            fill.setAlphaF(0.35 if low else 0.85)
            rect = QRectF(QPointF(px(d - 0.45), py(max(v, 0.0))), QPointF(px(d + 0.45), py(min(v, 0.0))))
            p.setPen(edge)
            p.setBrush(fill)
            p.drawRect(rect)
            if low:
                p.setBrush(QBrush(base, Qt.BDiagPattern))
                p.drawRect(rect)

        # real hazard / tracked-object markers at their real distance
        cyan = QColor(COLOR_CYAN)
        cyan.setAlphaF(0.8)
        dash = QPen(cyan, 1.0, Qt.DashLine)
        tiny = QFont(small)
        tiny.setPixelSize(9)
        for m in self._markers:
            d = m["distance_m"]
            label = m.get("label", m.get("severity", m.get("class_name", "?")))
            x = px(d)
            p.setPen(dash)
            p.drawLine(QPointF(x, T), QPointF(x, T + ph))
            p.setPen(QColor(COLOR_CYAN))
            p.setFont(tiny)
            p.drawText(QRectF(x + 2, T + 1, 120, 24), Qt.AlignLeft | Qt.AlignTop, f"{label}\n{d:.1f}m")
        p.setFont(small)
        p.setClipping(False)

        # frame + legend (upper right)
        p.setPen(QPen(QColor(COLOR_BORDER), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(plot)
        entries = [
            (QColor(COLOR_STATIC), False, "Height above ground (bump)"),
            (QColor(COLOR_CRITICAL), False, "Depth below ground (dip)"),
            (QColor(COLOR_TEXT_DIM), True, "Low confidence (<5 pts)"),
        ]
        tiny_fm = p.fontMetrics()
        lw = max(tiny_fm.horizontalAdvance(t) for _, _, t in entries) + 30
        lh = 14 * len(entries) + 6
        box = QRectF(L + pw - lw - 6, T + 6, lw, lh)
        p.setPen(QPen(QColor(COLOR_BORDER), 1))
        p.setBrush(QColor(COLOR_PANEL))
        p.drawRect(box)
        for i, (c, hatched, text) in enumerate(entries):
            y = box.top() + 4 + 14 * i
            sw = QRectF(box.left() + 6, y + 1, 12, 9)
            fill = QColor(c)
            if hatched:
                fill.setAlphaF(0.35)
            p.setPen(Qt.NoPen)
            p.setBrush(fill)
            p.drawRect(sw)
            if hatched:
                p.setBrush(QBrush(c, Qt.BDiagPattern))
                p.drawRect(sw)
            p.setPen(dim)
            p.drawText(QPointF(box.left() + 24, y + 9), text)


class AlertCard(QFrame):
    """Real nearby object alert based on actual tracked positions and velocities."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background-color: {COLOR_PANEL}; border: 1px solid {COLOR_CRITICAL}; border-radius: 6px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        self.lbl_title = QLabel("")
        self.lbl_title.setStyleSheet(f"color: {COLOR_TEXT}; font-weight: 700; font-size: {FONT_SIZE_EMPHASIS};")

        self.lbl_body = QLabel("")
        self.lbl_body.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-weight: 600; font-size: {FONT_SIZE_BODY};")
        self.lbl_body.setWordWrap(True)

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_body)
        # No longer setFixedWidth(320): that was sized for the old
        # fixed-position floating overlay. Now laid out inside the left
        # panel column (see _build_left_panel), it should size to that
        # column's real width instead of forcing a fixed width that may be
        # wider or narrower than the panel.
        self.setMinimumHeight(60)

    def set_data(self, track, dist: float, speed: float, direction: str, proximity: str):
        self.lbl_title.setText(f"{track.class_name} #{track.track_id}")
        conf_str = f"{track.confidence*100:.1f}%" if track.confidence > 0.0 else "N/A"
        text = (
            f"Distance: {dist:.1f} m\n"
            f"Speed: {speed:.1f} m/s\n"
            f"Direction: {direction}\n"
            f"{proximity}\n"
            f"Confidence: {conf_str}"
        )
        self.lbl_body.setText(text)


class FoveaXDashboardWindow(QMainWindow):
    # Emitted when a left-panel control requests a real Open3D camera
    # action ("zoom_in", "zoom_out", "rotate", "reset", "view_top",
    # "view_perspective", "view_side"). Connected externally (in
    # src/13_realtime_dashboard.py, which owns the o3d command queue) to
    # actually forward the command to the Open3D process.
    camera_command_requested = pyqtSignal(str)

    # Playback transport controls. Connected externally (in
    # src/13_realtime_dashboard.py, which owns the real frame-by-frame
    # replay loop and per-frame cache) -- this window only emits requests,
    # it never advances frames itself.
    play_pause_requested = pyqtSignal()
    step_forward_requested = pyqtSignal()
    step_backward_requested = pyqtSignal()
    rate_changed = pyqtSignal(float)
    restart_requested = pyqtSignal()
    
    # Environment switching
    environment_requested = pyqtSignal(str)
    
    # Load dataset requested: format, data_root, sequence, kwargs(ground_noise_m, min_points, etc)
    load_dataset_requested = pyqtSignal(str, str, str, dict)

    def __init__(self, embed_3d: bool = False):
        super().__init__()
        self.setWindowTitle("FOVEAX Phase 9 - 2D & Metrics Dashboard")
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)

        # Default to the OVERHEAD layer so real overhead obstacles (and the
        # always-visible FORWARD CORRIDOR CROSS-SECTION panel below) make
        # real terrain diversity evident without an extra click.
        self.current_map_type = "overhead"
        self.embed_3d = embed_3d
        self.embed_widget = None
        self._map_type_buttons = {}
        # Direction-label hysteresis state (Bug 3 fix): track_id ->
        # {"committed": str, "candidate": str, "count": int}. See
        # _stabilize_direction_label() for why this exists -- it never
        # changes the real radial-velocity computation, only how many
        # consecutive frames a new label must persist before the
        # DISPLAYED label actually changes.
        self._direction_label_state: dict[int, dict] = {}

        # OPT-5: Per-layer hash of the last rendered grid data.  If the hash
        # matches the incoming grid, the QImage render pass is skipped entirely
        # -- colormap + uint8 cast + QPixmap.scaled() are non-trivial even for
        # a small 80x80 grid.
        self._last_grid_hash: dict[str, int] = {}

        # OPT-9: Fingerprint of the last hazard cluster list.  Table rebuild
        # (setRowCount + setItem for every cell) is skipped when identical.
        self._last_hazard_key: tuple | None = None

        # OPT-8: Reusable AlertCard pool -- widgets are updated in-place
        # rather than deleted + recreated on every frame.
        self._alert_cards: list = []

        self._init_ui()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer = QVBoxLayout(central_widget)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        self.threat_toast = self._build_threat_toast(central_widget)
        self.threat_toast.hide()

        # Built before _build_body() (which builds the left panel) so
        # _build_left_panel can lay this out directly in its own empty
        # space below the combined ADAPTIVE RESOLUTION ZONES box, instead
        # of floating it as a fixed-position overlay on top of the 3D view.
        self.alert_container = QWidget()
        self.alert_layout = QVBoxLayout(self.alert_container)
        self.alert_layout.setContentsMargins(0, 0, 0, 0)
        self.alert_layout.setSpacing(6)
        self.alert_container.hide()

        outer.addWidget(self._build_header())
        outer.addWidget(self._build_metrics_bar())
        outer.addLayout(self._build_body(), stretch=1)
        outer.addWidget(self._build_centerline_panel())
        outer.addWidget(self._build_footer())

        self.drawer = self._build_system_drawer(central_widget)
        self.drawer.hide()

    def _build_header(self):
        header = QFrame()
        header.setFixedHeight(46)
        header.setStyleSheet(f"background-color: {COLOR_PANEL}; border-bottom: 1px solid {COLOR_BORDER};")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 4, 12, 4)

        title = QLabel("FOVEAX 2.5D LiDAR PERCEPTION")
        title.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 14px; font-weight: 700;")
        layout.addWidget(title)
        
        self.env1_btn = QPushButton("ENVIRONMENT 1 (Seq 00001)")
        self.env1_btn.clicked.connect(lambda: self.environment_requested.emit("00001"))
        layout.addWidget(self.env1_btn)
        
        self.env2_btn = QPushButton("ENVIRONMENT 2 (Seq 00003)")
        self.env2_btn.clicked.connect(lambda: self.environment_requested.emit("00003"))
        layout.addWidget(self.env2_btn)
        self.set_active_environment("00001")
        
        self.load_dataset_btn = QPushButton("LOAD DATASET")
        self.load_dataset_btn.setStyleSheet(f"background-color: {COLOR_STATIC}; color: #000; font-weight: 700; border: none; padding: 4px 12px; border-radius: 4px;")
        self.load_dataset_btn.clicked.connect(self._open_load_dataset_dialog)
        layout.addWidget(self.load_dataset_btn)

        layout.addStretch(1)

        # --- Playback transport controls -----------------------------
        # This dashboard replays a real, finite, recorded sequence of
        # LiDAR frames (not a live sensor feed) -- these controls step
        # through the real recorded frames, they never fabricate motion
        # between them (see step/rate wiring in 13_realtime_dashboard.py).
        self.step_back_btn = QPushButton("<< STEP")
        self.step_back_btn.clicked.connect(self.step_backward_requested.emit)
        layout.addWidget(self.step_back_btn)

        self.play_pause_btn = QPushButton("PAUSE")
        self.play_pause_btn.clicked.connect(self.play_pause_requested.emit)
        layout.addWidget(self.play_pause_btn)
        self._apply_play_pause_style(playing=True)

        self.step_fwd_btn = QPushButton("STEP >>")
        self.step_fwd_btn.clicked.connect(self.step_forward_requested.emit)
        layout.addWidget(self.step_fwd_btn)

        self.restart_btn = QPushButton("RESTART")
        self.restart_btn.setStyleSheet(f"QPushButton {{ border: 1px solid {COLOR_CYAN}; }}")
        self.restart_btn.clicked.connect(self.restart_requested.emit)
        layout.addWidget(self.restart_btn)

        rate_label = QLabel("Hz:")
        rate_label.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        layout.addWidget(rate_label)
        self.rate_spinbox = QSpinBox()
        self.rate_spinbox.setRange(1, 30)
        self.rate_spinbox.setValue(10)
        self.rate_spinbox.valueChanged.connect(lambda v: self.rate_changed.emit(float(v)))
        layout.addWidget(self.rate_spinbox)

        self.lbl_frame_counter = QLabel("Frame -- / --")
        self.lbl_frame_counter.setStyleSheet(f"color: {COLOR_TEXT}; font-weight: 600; padding: 0 8px;")
        layout.addWidget(self.lbl_frame_counter)

        layout.addStretch(1)

        self.lbl_fps_badge = QLabel("-- FPS")
        self.lbl_fps_badge.setStyleSheet(f"color: {COLOR_SAFE}; font-weight: 600;")
        layout.addWidget(self.lbl_fps_badge)

        drawer_btn = QPushButton("SYSTEM DETAILS")
        drawer_btn.clicked.connect(self._toggle_drawer)
        layout.addWidget(drawer_btn)
        return header

    def set_active_environment(self, seq: str):
        active_style = f"background-color: {COLOR_CYAN}; color: {COLOR_PANEL}; font-weight: 700; border: none; padding: 4px 8px; border-radius: 4px;"
        inactive_style = f"background-color: transparent; color: {COLOR_TEXT_DIM}; font-weight: 600; border: 1px solid {COLOR_BORDER}; padding: 4px 8px; border-radius: 4px;"
        self.env1_btn.setStyleSheet(active_style if seq == "00001" else inactive_style)
        self.env2_btn.setStyleSheet(active_style if seq == "00003" else inactive_style)
        
    def _open_load_dataset_dialog(self):
        dialog = LoadDatasetDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            fmt, path, seq, kwargs = dialog.get_result()
            self.load_dataset_requested.emit(fmt, path, seq, kwargs)


    def set_frame_counter(self, current: int, total: int):
        self.lbl_frame_counter.setText(f"Frame {current} / {total}")

    def set_play_pause_state(self, playing: bool):
        self._apply_play_pause_style(playing)

    def _apply_play_pause_style(self, playing: bool):
        # Explicit, self-contained inline style per state -- deliberately
        # NOT driven by the QPushButton:checked QSS pseudo-class, which
        # was found to sometimes not repaint after a programmatic
        # setChecked() call (a real, confirmed Qt style-repolish quirk),
        # leaving the button text low-contrast even though the logical
        # play/pause state itself was always correct. Setting a full
        # inline stylesheet directly guarantees readable text in both
        # states regardless of that repaint timing.
        self.play_pause_btn.setText("PAUSE" if playing else "PLAY")
        accent = COLOR_SAFE if playing else COLOR_CAUTION
        self.play_pause_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_PANEL_ALT};
                border: 1px solid {accent};
                border-radius: 4px;
                padding: 5px 8px;
                color: {COLOR_TEXT};
                font-weight: 600;
            }}
            QPushButton:hover {{ border: 1px solid {COLOR_CYAN}; }}
        """)

    def set_rate_hz(self, hz: float):
        self.rate_spinbox.blockSignals(True)
        self.rate_spinbox.setValue(int(round(hz)))
        self.rate_spinbox.blockSignals(False)

    def _populate_efficiency_stat(self, zones=None):
        """Compute (once, or when the zone config changes) the real Phase B
        memory-reduction stat via
        src/07_adaptive_2point5d_map.py::compute_memory_reduction_metrics()
        -- reused directly, never re-derived here. A static architectural
        property of the given zone config, not a per-frame measurement.
        """
        import importlib
        mod = importlib.import_module("src.07_adaptive_2point5d_map")
        metrics = mod.compute_memory_reduction_metrics(zones=zones)
        self.lbl_efficiency_cells.setText(
            f"Cell reduction vs. uniform 3D voxels: {metrics['cell_savings_vs_3d_pct']:.2f}%"
        )
        self.lbl_efficiency_bytes.setText(
            f"Byte savings vs. uniform 3D voxels: {metrics['byte_savings_vs_3d_pct']:.2f}%"
        )

    def set_accuracy_metrics(self, has_ground_truth: bool, predictor_mode: str, accuracy_metrics: dict | None):
        if predictor_mode == "ground_truth":
            self.lbl_accuracy_note.setText(
                "predictor=ground_truth: comparing real ground truth against "
                "itself (trivially ~100%). Use --predictor salsanext (or "
                "mock) to compare an independent real prediction."
            )
        else:
            self.lbl_accuracy_note.setText(f"predictor={predictor_mode}, compared against real ground truth.")

        if not has_ground_truth or accuracy_metrics is None:
            na_text = "N/A - no ground truth for this frame"
            self.lbl_accuracy_near.setText(f"Near (0-10m): {na_text}")
            self.lbl_accuracy_mid.setText(f"Mid (10-30m): {na_text}")
            self.lbl_accuracy_far.setText(f"Far (30-100m): {na_text}")
            self.lbl_accuracy_overall.setText(f"Overall (0-100m): {na_text}")
            return

        def fmt(bucket_name, result):
            if result is None:
                return f"{bucket_name}: N/A - no ground truth points in this bucket"
            return f"{bucket_name}: {result['accuracy']*100:.2f}% (n={result['n_points']:,})"

        self.lbl_accuracy_near.setText(fmt("Near (0-10m)", accuracy_metrics.get("near")))
        self.lbl_accuracy_mid.setText(fmt("Mid (10-30m)", accuracy_metrics.get("mid")))
        self.lbl_accuracy_far.setText(fmt("Far (30-100m)", accuracy_metrics.get("far")))
        self.lbl_accuracy_overall.setText(fmt("Overall (0-100m)", accuracy_metrics.get("overall")))

    def set_ego_motion_estimate(self, displacement_m: float | None, computed_at_frame: int | None = None,
                                every_n: int | None = None):
        """Show the ICP displacement estimate. It is recomputed only every
        `every_n`-th frame (it costs tens of ms), so the label says which frame
        it was computed at instead of implying it is current."""
        cadence = f"; recomputed every {every_n} frames" if every_n and every_n > 1 else ""
        if displacement_m is None:
            self.lbl_ego_motion.setText(f"Est. relative displacement: N/A (no estimate yet, or it failed{cadence})")
        elif computed_at_frame is None:
            self.lbl_ego_motion.setText(f"Est. relative displacement: {displacement_m:.3f} m (since previous frame)")
        else:
            self.lbl_ego_motion.setText(
                f"Est. relative displacement: {displacement_m:.3f} m "
                f"(frame {computed_at_frame + 1} vs the one before{cadence})")

    def _build_metrics_bar(self):
        bar = QFrame()
        bar.setFixedHeight(56)
        bar.setStyleSheet(f"background-color: {COLOR_PANEL}; border: 1px solid {COLOR_BORDER}; border-radius: 6px;")
        layout = QHBoxLayout(bar)

        def make_card(label_text):
            box = QVBoxLayout()
            val = QLabel("--")
            val.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 15px; font-weight: 700;")
            cap = QLabel(label_text)
            cap.setStyleSheet(NOTE_STYLE)
            box.addWidget(val)
            box.addWidget(cap)
            layout.addLayout(box)
            return val

        self.lbl_m_fps = make_card("FPS")
        self.lbl_m_latency = make_card("LATENCY")
        self.lbl_m_points = make_card("POINTS/FRAME")
        self.lbl_m_cpu = make_card("CPU")
        self.lbl_m_ram = make_card("RAM")
        self.lbl_m_gpu = make_card("GPU")
        self.lbl_m_vram = make_card("VRAM")
        layout.addStretch(1)
        return bar

    def _build_body(self):
        body = QHBoxLayout()
        body.addWidget(self._build_left_panel(), stretch=2)
        body.addWidget(self._build_center_panel(), stretch=5 if self.embed_3d else 3)
        body.addWidget(self._build_right_panel(), stretch=2)
        return body

    def _build_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        cam_group = QGroupBox("VIEW / CAMERA (live Open3D)")
        cam_layout = QGridLayout()
        view_top = QPushButton("TOP")
        view_persp = QPushButton("PERSPECTIVE")
        view_side = QPushButton("SIDE")
        for i, (btn, cmd) in enumerate([
            (view_top, "view_top"), (view_persp, "view_perspective"), (view_side, "view_side")
        ]):
            btn.clicked.connect(lambda _, c=cmd: self.camera_command_requested.emit(c))
            cam_layout.addWidget(btn, 0, i)

        zoom_in = QPushButton("ZOOM +")
        zoom_out = QPushButton("ZOOM -")
        rotate_btn = QPushButton("ROTATE")
        reset_btn = QPushButton("RESET")
        for i, (btn, cmd) in enumerate([
            (zoom_in, "zoom_in"), (zoom_out, "zoom_out"),
            (rotate_btn, "rotate"), (reset_btn, "reset")
        ]):
            btn.clicked.connect(lambda _, c=cmd: self.camera_command_requested.emit(c))
            cam_layout.addWidget(btn, 1, i)
        cam_group.setLayout(cam_layout)
        layout.addWidget(cam_group)

        layers_group = QGroupBox("2D MAP LAYER")
        layers_layout = QVBoxLayout()
        for map_type in ["elevation", "traversability", "roi", "uncertainty", "overhead"]:
            btn = QPushButton(map_type.upper())
            btn.setCheckable(True)
            btn.setChecked(map_type == self.current_map_type)
            btn.clicked.connect(lambda _, mt=map_type: self._on_map_changed(mt))
            layers_layout.addWidget(btn)
            self._map_type_buttons[map_type] = btn
        layers_group.setLayout(layers_layout)
        layout.addWidget(layers_group)

        # Combined spec + real per-zone tracked-object count, one row per
        # zone -- was two separate boxes (ADAPTIVE RESOLUTION ZONES (spec)
        # and OBJECTS PER ZONE (live)) repeating the same Near/Middle/Far
        # row structure; merged since they describe the same three zones.
        # The live count comes from src/dashboard/track_labels.py::
        # zone_object_counts, refreshed each frame in update_ui from
        # state.tracks -- 0 objects in a zone is shown as "0", not hidden.
        res_group = QGroupBox("ADAPTIVE RESOLUTION ZONES (spec + live objects)")
        res_layout = QVBoxLayout()
        self._zone_labels: dict[str, QLabel] = {}
        for name, rng, res in RESOLUTION_ZONES:
            lbl = QLabel(f"{name}: {rng} -> {res} | 0 objects")
            res_layout.addWidget(lbl)
            self._zone_labels[name] = lbl
        res_group.setLayout(res_layout)
        layout.addWidget(res_group)

        # Real nearest-dynamic-object threat toast (built in _init_ui,
        # populated in update_ui): previously a fixed-position overlay
        # floating on top of the 3D view at screen coords (250, 115).
        # Laid out here instead, between the zone box and the approaching-
        # object alert cards below -- hidden (no reserved space) when no
        # dynamic object is currently tracked.
        layout.addWidget(self.threat_toast)

        # Real nearby-object alert cards (built in _init_ui, populated by
        # _update_alerts): previously a fixed-position overlay floating on
        # top of the 3D view at screen coords (620, 115), which also meant
        # fighting the embedded view's native child windows for z-order.
        # Laid out here instead, in what was empty space below the zone
        # panel -- hidden (no minimum height reserved) when there are no
        # real alerts.
        layout.addWidget(self.alert_container)

        layout.addStretch(1)
        return panel

    def _build_center_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        if self.embed_3d:
            self.embed_widget = Open3DEmbedWidget()
            layout.addWidget(self.embed_widget, stretch=3)

        self.map_label = QLabel("Waiting for data...")
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setMinimumSize(400, 300)
        self.map_label.setStyleSheet(f"background-color: black; color: {COLOR_TEXT_DIM}; border: 1px solid {COLOR_BORDER};")
        layout.addWidget(self.map_label, stretch=2)

        ego_relative_note = QLabel(
            "Ego-relative view: recentered on the vehicle each frame -- not an "
            "absolute-position world trajectory (this pipeline has no odometry)."
        )
        ego_relative_note.setStyleSheet(NOTE_STYLE + " font-style: italic;")
        ego_relative_note.setAlignment(Qt.AlignCenter)
        layout.addWidget(ego_relative_note)

        self.overhead_legend = QLabel(
            f"<span style='color:{COLOR_STATIC};'>&#9632;</span> Solid obstacle &nbsp;&nbsp;"
            f"<span style='color:{COLOR_CYAN};'>&#9645;</span> Overhead clearance (hatched, passable below) &nbsp;&nbsp;"
            f"<span style='color:#646464;'>&#9632;</span> Indeterminate (too few points)"
        )
        self.overhead_legend.setStyleSheet(NOTE_STYLE + " padding: 2px;")
        self.overhead_legend.hide()
        layout.addWidget(self.overhead_legend)
        return panel

    def _build_right_panel(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_status_tab(), "STATUS")
        self.tabs.addTab(self._build_terrain_tab(), "TERRAIN")
        self.tabs.addTab(self._build_system_tab(), "SYSTEM")
        return self.tabs

    def _build_status_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        status_group = QGroupBox("EGO TERRAIN STATUS")
        status_layout = QVBoxLayout()
        self.lbl_terrain_status = QLabel("UNKNOWN")
        self.lbl_terrain_status.setStyleSheet(f"font-size: {FONT_SIZE_HERO}; font-weight: 700; color: {COLOR_TEXT_DIM};")
        self.lbl_terrain_confidence = QLabel("Confidence: --")
        status_layout.addWidget(self.lbl_terrain_status)
        status_layout.addWidget(self.lbl_terrain_confidence)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        self.lbl_coord_warning = QLabel("")
        self.lbl_coord_warning.setStyleSheet(f"color: {COLOR_CAUTION}; font-weight: bold;")
        self.lbl_coord_warning.setWordWrap(True)
        self.lbl_coord_warning.hide()
        layout.addWidget(self.lbl_coord_warning)

        tracks_group = QGroupBox("TRACKED OBJECTS")
        tracks_layout = QVBoxLayout()
        self.list_tracks = QListWidget()
        tracks_layout.addWidget(self.list_tracks)
        tracks_group.setLayout(tracks_layout)
        layout.addWidget(tracks_group, stretch=1)

        threat_group = QGroupBox("NEAREST DYNAMIC OBJECT")
        threat_layout = QVBoxLayout()
        self.lbl_threat = QLabel("No dynamic objects tracked.")
        self.lbl_threat.setWordWrap(True)
        threat_layout.addWidget(self.lbl_threat)
        threat_group.setLayout(threat_layout)
        layout.addWidget(threat_group)

        return tab

    def _build_terrain_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        width_group = QGroupBox("ROAD WIDTH (Safe corridor, >=0.70)")
        width_layout = QVBoxLayout()
        self.lbl_road_width = QLabel("N/A")
        self.lbl_road_width.setStyleSheet(f"font-size: {FONT_SIZE_HEADLINE}; font-weight: 700; color: {COLOR_SAFE};")
        width_layout.addWidget(self.lbl_road_width)
        width_group.setLayout(width_layout)
        layout.addWidget(width_group)

        elev_group = QGroupBox("ELEVATION PROFILE (centerline, real hazards)")
        elev_layout = QVBoxLayout()
        self.elevation_widget = ElevationProfileWidget()
        elev_layout.addWidget(self.elevation_widget)
        elev_group.setLayout(elev_layout)
        layout.addWidget(elev_group)

        unc_group = QGroupBox("MAP UNCERTAINTY (Phase 5 formula)")
        unc_layout = QVBoxLayout()
        self.lbl_uncertainty = QLabel("Mean uncertainty: N/A")
        self.lbl_occupied = QLabel("Occupied grid fraction: N/A")
        unc_layout.addWidget(self.lbl_uncertainty)
        unc_layout.addWidget(self.lbl_occupied)
        unc_group.setLayout(unc_layout)
        layout.addWidget(unc_group)

        hazard_group = QGroupBox("HAZARD CLUSTERS (connected components)")
        hazard_layout = QVBoxLayout()
        self.hazard_table = QTableWidget(0, 6)
        self.hazard_table.setHorizontalHeaderLabels(
            ["Dist (m)", "Min Trav.", "Severity", "Conf.", "Kind", "Depth/Height (m)"]
        )
        self.hazard_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hazard_table.verticalHeader().setVisible(False)
        hazard_layout.addWidget(self.hazard_table)
        hazard_group.setLayout(hazard_layout)
        layout.addWidget(hazard_group, stretch=1)

        return tab

    def _build_system_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        stats_group = QGroupBox("HARDWARE")
        stats_layout = QVBoxLayout()
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_latency = QLabel("Latency: -- ms")
        self.lbl_cpu = QLabel("CPU: -- %")
        self.prog_cpu = QProgressBar()
        self.lbl_ram = QLabel("RAM: -- %")
        self.prog_ram = QProgressBar()
        self.lbl_gpu = QLabel("GPU: -- %")
        self.prog_gpu = QProgressBar()
        self.lbl_vram = QLabel("VRAM: -- %")
        self.prog_vram = QProgressBar()
        for w in [self.lbl_fps, self.lbl_latency, self.lbl_cpu, self.prog_cpu,
                  self.lbl_ram, self.prog_ram, self.lbl_gpu, self.prog_gpu,
                  self.lbl_vram, self.prog_vram]:
            stats_layout.addWidget(w)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        session_group = QGroupBox("SESSION")
        session_layout = QVBoxLayout()
        self.lbl_frame_idx = QLabel("Frame: --")
        session_layout.addWidget(self.lbl_frame_idx)
        session_group.setLayout(session_layout)
        layout.addWidget(session_group)

        accuracy_group = QGroupBox("ACCURACY (current frame, vs. real ground truth)")
        accuracy_layout = QVBoxLayout()
        self.lbl_accuracy_note = QLabel("")
        self.lbl_accuracy_note.setWordWrap(True)
        self.lbl_accuracy_note.setStyleSheet(NOTE_STYLE)
        accuracy_layout.addWidget(self.lbl_accuracy_note)
        self.lbl_accuracy_near = QLabel("Near (0-10m): --")
        self.lbl_accuracy_mid = QLabel("Mid (10-30m): --")
        self.lbl_accuracy_far = QLabel("Far (30-100m): --")
        self.lbl_accuracy_overall = QLabel("Overall (0-100m): --")
        self.lbl_accuracy_overall.setStyleSheet(f"font-size: {FONT_SIZE_HEADLINE}; font-weight: 700; color: {COLOR_SAFE};")
        for w in [self.lbl_accuracy_near, self.lbl_accuracy_mid,
                  self.lbl_accuracy_far, self.lbl_accuracy_overall]:
            accuracy_layout.addWidget(w)
        accuracy_group.setLayout(accuracy_layout)
        layout.addWidget(accuracy_group)

        efficiency_group = QGroupBox("EFFICIENCY (Phase B, static grid design property)")
        efficiency_layout = QVBoxLayout()
        efficiency_note = QLabel(
            "Fixed property of the current Near/Middle/Far resolution-zone "
            "config (src/07_adaptive_2point5d_map.py), not a live per-frame "
            "measurement -- recomputed only if that zone config changes."
        )
        efficiency_note.setWordWrap(True)
        efficiency_note.setStyleSheet(NOTE_STYLE)
        efficiency_layout.addWidget(efficiency_note)
        self.lbl_efficiency_cells = QLabel("Cell reduction vs. uniform 3D voxels: --")
        self.lbl_efficiency_bytes = QLabel("Byte savings vs. uniform 3D voxels: --")
        efficiency_layout.addWidget(self.lbl_efficiency_cells)
        efficiency_layout.addWidget(self.lbl_efficiency_bytes)
        efficiency_group.setLayout(efficiency_layout)
        layout.addWidget(efficiency_group)
        self._populate_efficiency_stat()

        motion_group = QGroupBox("EGO MOTION (informational only)")
        motion_layout = QVBoxLayout()
        motion_note = QLabel(
            "This pipeline has no odometry. The value below is a coarse, "
            "real point-cloud-registration (ICP) estimate of relative "
            "displacement between consecutive frames -- informational "
            "only. It does NOT reposition any view; every 2D/3D panel "
            "remains ego-relative (recentered each frame)."
        )
        motion_note.setWordWrap(True)
        motion_note.setStyleSheet(NOTE_STYLE)
        motion_layout.addWidget(motion_note)
        self.lbl_ego_motion = QLabel("Est. relative displacement: N/A")
        motion_layout.addWidget(self.lbl_ego_motion)
        motion_group.setLayout(motion_layout)
        layout.addWidget(motion_group)

        layout.addStretch(1)
        return tab

    def _build_centerline_panel(self):
        group = QGroupBox("FORWARD CORRIDOR CROSS-SECTION (real elevation, real hazards)")
        group.setFixedHeight(190)
        layout = QVBoxLayout(group)
        self.centerline_chart = CenterlineProfileChart()
        layout.addWidget(self.centerline_chart)
        return group

    def _build_footer(self):
        footer = QFrame()
        footer.setFixedHeight(22)
        footer.setStyleSheet(f"background-color: {COLOR_PANEL}; border-top: 1px solid {COLOR_BORDER};")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(10, 0, 10, 0)
        self.lbl_footer = QLabel("FOVEAX Phase 9 -- waiting for first frame")
        self.lbl_footer.setStyleSheet(NOTE_STYLE)
        layout.addWidget(self.lbl_footer)
        layout.addStretch(1)
        self.lbl_dataset = QLabel("")
        self.lbl_dataset.setStyleSheet(NOTE_STYLE)
        layout.addWidget(self.lbl_dataset)
        return footer

    def set_dataset_banner(self, text, sample: bool = False):
        """Always-visible statement of which data is playing (sample data is highlighted)."""
        self.lbl_dataset.setText(text or "")
        self.lbl_dataset.setStyleSheet(
            f"font-size: {FONT_SIZE_BODY}; font-weight: 700; color: {COLOR_CAUTION};" if sample else NOTE_STYLE)

    def _build_threat_toast(self, parent):
        toast = QFrame(parent)
        toast.setStyleSheet(
            f"background-color: {COLOR_PANEL}; border: 1px solid {COLOR_CRITICAL}; border-radius: 6px;"
        )
        layout = QVBoxLayout(toast)
        layout.setContentsMargins(10, 8, 10, 8)
        self.lbl_toast_text = QLabel("")
        self.lbl_toast_text.setStyleSheet(f"color: {COLOR_TEXT}; font-weight: 600; font-size: {FONT_SIZE_BODY};")
        # Wrapping (rather than a fixed single-line box) is what fixed the
        # garbled/overlapping toast text: without it, a long message (real
        # class name + track id + distance + speed) simply overflowed the
        # frame's last-known size instead of reflowing to fit.
        self.lbl_toast_text.setWordWrap(True)
        layout.addWidget(self.lbl_toast_text)
        toast.setMinimumHeight(50)
        # No longer a fixed-position overlay floating on top of the 3D view
        # (previously .move(250, 115) + WA_NativeWindow + bring_to_absolute_top
        # to fight the embedded view's native child windows for z-order --
        # see the matching alert_container history). Laid out directly in
        # the left panel now (see _build_left_panel), between the
        # ADAPTIVE RESOLUTION ZONES box and the alert cards, so it no
        # longer overlaps any native child window at all.
        return toast

    def _build_system_drawer(self, parent):
        drawer = QFrame(parent)
        drawer.setStyleSheet(
            f"background-color: {COLOR_PANEL}; border: 1px solid {COLOR_BORDER}; border-radius: 6px;"
        )
        layout = QVBoxLayout(drawer)

        title = QLabel("SYSTEM DETAILS")
        title.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 13px; font-weight: 700;")
        layout.addWidget(title)

        layout.addWidget(QLabel(
            "Mode: Adaptive resolution (fixed Near/Middle/Far zone config,\n"
            "not a live-switchable runtime mode in this pipeline)."
        ))

        pipeline = QLabel(
            "PIPELINE:\n"
            "LiDAR point cloud\n"
            "  -> Object detection & tracking (src/perception, src/tracking)\n"
            "  -> 2.5D elevation / traversability / ROI / uncertainty grids\n"
            "     (src/dashboard/data_streamer.py, canonical Safe/Caution/\n"
            "     Blocked convention from src/06_terrain_traversability.py)\n"
            "  -> Open3D 3D view + Qt 2D dashboard (this window)"
        )
        pipeline.setWordWrap(True)
        layout.addWidget(pipeline)

        tags = QLabel("Tech stack: PyQt5, Open3D, NumPy, SciPy, PyTorch, pywin32 (win32gui)")
        tags.setWordWrap(True)
        tags.setStyleSheet(NOTE_STYLE)
        layout.addWidget(tags)

        close_btn = QPushButton("CLOSE")
        close_btn.clicked.connect(self._toggle_drawer)
        layout.addWidget(close_btn)

        drawer.setFixedWidth(320)
        return drawer

    def _toggle_drawer(self):
        if self.drawer.isVisible():
            self.drawer.hide()
        else:
            self.drawer.move(self.width() - self.drawer.width() - 20, 60)
            self.drawer.resize(self.drawer.width(), self.height() - 100)
            self.drawer.raise_()
            self.drawer.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.drawer.isVisible():
            self.drawer.move(self.width() - self.drawer.width() - 20, 60)
            self.drawer.resize(self.drawer.width(), self.height() - 100)
            
    def _stabilize_direction_label(self, track_id: int, raw_direction: str) -> str:
        """Hysteresis for the DISPLAYED Approaching/Receding/Crossing
        label (Bug 3 fix). Never touches the real radial-velocity value
        or the +-0.2 m/s deadband itself -- a new classification must
        repeat for DIRECTION_HYSTERESIS_FRAMES consecutive real frames
        before it's allowed to replace the currently displayed label,
        the same "require consistency before changing what's shown"
        pattern already used for the Static/Dynamic flag's min_hits_for_dynamic.
        """
        state = self._direction_label_state.get(track_id)
        if state is None:
            self._direction_label_state[track_id] = {
                "committed": raw_direction, "candidate": raw_direction, "count": 1,
            }
            return raw_direction

        if raw_direction == state["committed"]:
            state["candidate"] = raw_direction
            state["count"] = 0
            return state["committed"]

        if raw_direction == state["candidate"]:
            state["count"] += 1
        else:
            state["candidate"] = raw_direction
            state["count"] = 1

        if state["count"] >= DIRECTION_HYSTERESIS_FRAMES:
            state["committed"] = raw_direction
            state["count"] = 0

        return state["committed"]

    def _update_alerts(self, alerts):
        """Update the real approaching object alert cards.

        OPT-8: Reuses the _alert_cards pool instead of calling deleteLater()
        + creating new AlertCard widgets on every frame.  Each card is just
        updated in-place (show/hide + set_data) which avoids a Qt layout
        recalculation triggered by widget deletion.
        """
        n_alerts = len(alerts)
        n_cards  = len(self._alert_cards)

        # Grow the pool if needed (up to 3 alert cards are ever shown).
        for _ in range(n_alerts - n_cards):
            card = AlertCard(self.alert_container)
            self.alert_layout.addWidget(card)
            self._alert_cards.append(card)

        # Update visible cards in-place; hide surplus cards.
        for i, card in enumerate(self._alert_cards):
            if i < n_alerts:
                dist, track, speed, direction, prox = alerts[i]
                card.set_data(track, dist, speed, direction, prox)
                card.show()
            else:
                card.hide()

        self.alert_container.setVisible(bool(alerts))

    # ------------------------------------------------------------------
    # Data wiring
    # ------------------------------------------------------------------
    def _on_map_changed(self, map_type: str):
        self.current_map_type = map_type
        for mt, btn in self._map_type_buttons.items():
            btn.setChecked(mt == map_type)
        self.overhead_legend.setVisible(map_type == "overhead")

    # -- write-only-on-change helpers: Qt repaints/relayouts on every set call
    @staticmethod
    def _set_text(widget, text):
        if widget.text() != text:
            widget.setText(text)

    @staticmethod
    def _set_value(widget, value):
        if widget.value() != value:
            widget.setValue(value)

    @staticmethod
    def _set_style(widget, style):
        if widget.styleSheet() != style:
            widget.setStyleSheet(style)

    @staticmethod
    def _sync_list(list_widget, rows):
        """Make a QListWidget show exactly `rows`, touching only changed items."""
        n = list_widget.count()
        for i, text in enumerate(rows[:n]):
            item = list_widget.item(i)
            if item.text() != text:
                item.setText(text)
        for text in rows[n:]:
            list_widget.addItem(text)
        while list_widget.count() > len(rows):
            list_widget.takeItem(list_widget.count() - 1)

    def update_ui(self, state: FrameState):
        """Update the dashboard UI from the FrameState."""
        # 1. Update Hardware Metrics
        m = state.metrics
        target_str = f"Target: {m.target_fps:.0f} Hz" if m.target_fps > 0 else "Target: --"
        self._set_text(self.lbl_fps, f"Achieved: {m.achieved_fps:.1f} FPS ({target_str})")
        self._set_text(self.lbl_latency, f"Latency: {m.latency_ms:.1f} ms")
        self._set_text(self.lbl_fps_badge, f"{m.achieved_fps:.1f} FPS ({target_str})")
        self._set_text(self.lbl_m_fps, f"{m.achieved_fps:.1f}")
        self._set_text(self.lbl_m_latency, f"{m.latency_ms:.0f} ms")
        self._set_text(self.lbl_m_points, f"{len(state.points):,}")
        self._set_text(self.lbl_m_cpu, f"{m.cpu_percent:.0f}%")
        self._set_text(self.lbl_m_ram, f"{m.ram_percent:.0f}%")

        if state.coordinate_warnings:
            self._set_text(self.lbl_coord_warning, "\n".join(state.coordinate_warnings))
            self.lbl_coord_warning.show()
        else:
            if self.lbl_coord_warning.text():
                self.lbl_coord_warning.clear()
            if self.lbl_coord_warning.isVisible():
                self.lbl_coord_warning.hide()

        self._set_text(self.lbl_cpu, f"CPU: {m.cpu_percent:.1f}%")
        self._set_value(self.prog_cpu, int(m.cpu_percent))

        self._set_text(self.lbl_ram, f"RAM: {m.ram_percent:.1f}% ({m.ram_used_gb:.1f}/{m.ram_total_gb:.1f} GB)")
        self._set_value(self.prog_ram, int(m.ram_percent))

        if m.gpu_available:
            self._set_text(self.lbl_gpu, f"GPU: {m.gpu_percent:.1f}%")
            self._set_value(self.prog_gpu, int(m.gpu_percent))
            self._set_text(self.lbl_vram, f"VRAM: {m.vram_percent:.1f}% ({m.vram_used_gb:.1f}/{m.vram_total_gb:.1f} GB)")
            self._set_value(self.prog_vram, int(m.vram_percent))
            self._set_text(self.lbl_m_gpu, f"{m.gpu_percent:.0f}%")
            self._set_text(self.lbl_m_vram, f"{m.vram_percent:.0f}%")
        else:
            self._set_text(self.lbl_gpu, "GPU: N/A")
            self._set_value(self.prog_gpu, 0)
            self._set_text(self.lbl_vram, "VRAM: N/A")
            self._set_value(self.prog_vram, 0)
            self._set_text(self.lbl_m_gpu, "N/A")
            self._set_text(self.lbl_m_vram, "N/A")

        # 1b. Real per-zone object counts (same tracks as the list below),
        # written into the combined spec+live row for each zone.
        counts = zone_object_counts(state.tracks)
        for name, rng, res in RESOLUTION_ZONES:
            self._set_text(self._zone_labels[name], 
                f"{name}: {rng} -> {res} | {counts[name]} objects"
            )

        # 2. Update Tracked Objects List + nearest dynamic threat
        rows = [tracked_object_row_text(t) for t in state.tracks]
        self._sync_list(self.list_tracks, rows)
        nearest_dynamic = None
        nearest_dist = None
        for t in state.tracks:
            if t.dynamic:
                dist = track_distance_m(t)
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_dynamic = t

        if nearest_dynamic is not None:
            speed = float(np.linalg.norm(nearest_dynamic.state[3:6]))
            self._set_text(self.lbl_threat, 
                f"ID {nearest_dynamic.track_id} ({nearest_dynamic.class_name})\n"
                f"Distance: {nearest_dist:.1f} m | Speed: {speed:.1f} m/s"
            )
            self._set_text(self.lbl_toast_text, 
                f"DYNAMIC OBJECT: {nearest_dynamic.class_name} #{nearest_dynamic.track_id} "
                f"at {nearest_dist:.1f} m, {speed:.1f} m/s"
            )
            self.threat_toast.show()
        else:
            self._set_text(self.lbl_threat, "No dynamic objects tracked.")
            self.threat_toast.hide()

        # Update real alert cards for approaching objects
        alerts = []
        seen_track_ids = set()
        for t in state.tracks:
            dist = float(np.hypot(t.state[0], t.state[1]))
            if dist < 15.0 and t.dynamic:
                seen_track_ids.add(t.track_id)
                speed = float(np.linalg.norm(t.state[3:6]))
                x, y = float(t.state[0]), float(t.state[1])
                vx, vy = float(t.state[3]), float(t.state[4])

                # relative radial velocity = (x*vx + y*vy)/dist
                # Negative means the object is moving towards the ego (0,0)
                # -- this real computation is unchanged by the Bug 3 fix
                # below; only the DISPLAYED label is stabilized.
                radial_v = (x*vx + y*vy) / dist if dist > 0.1 else 0.0

                # Classify direction using real vector math
                if radial_v < -0.2:
                    raw_direction = "Approaching"
                elif radial_v > 0.2:
                    raw_direction = "Receding"
                else:
                    raw_direction = "Crossing"

                direction = self._stabilize_direction_label(t.track_id, raw_direction)

                if direction == "Approaching":
                    if dist < 10.0:
                        prox = "Proximity: Close (<10m)"
                    elif dist < 25.0:
                        prox = "Proximity: Near (10-25m)"
                    else:
                        prox = "Proximity: Far (>25m)"
                    alerts.append((dist, t, speed, direction, prox))
                    
        alerts.sort(key=lambda x: x[0])
        self._update_alerts(alerts[:3])

        # Drop hysteresis state for tracks no longer in the qualifying
        # (dist<15m, dynamic) set -- avoids unbounded growth over a long
        # playback and lets a track that leaves/re-enters range start
        # fresh rather than carrying stale candidate-label state.
        stale_ids = set(self._direction_label_state) - seen_track_ids
        for tid in stale_ids:
            del self._direction_label_state[tid]

        # 3. Update 2D Map Image
        if self.current_map_type == "overhead":
            self._render_overhead_to_label(
                state.grid_maps.get("traversability"), state.grid_maps.get("overhead")
            )
        elif self.current_map_type in state.grid_maps:
            grid = state.grid_maps[self.current_map_type]
            self._render_grid_to_label(grid, self.current_map_type)
        else:
            self._set_text(self.map_label, f"Map '{self.current_map_type}' not available.")

        # 4. Ego terrain status (from the real live traversability grid)
        self._update_terrain_status(state)

        # 5. Road width / elevation profile / uncertainty / hazards
        if state.road_width_m is not None:
            self._set_text(self.lbl_road_width, f"{state.road_width_m:.2f} m")
        else:
            self._set_text(self.lbl_road_width, "N/A (no safe cell under ego)")

        self.elevation_widget.set_data(state.elevation_profile, state.elevation_hazards_m)
        self.centerline_chart.set_data(state.centerline_profile, state.centerline_markers)

        if state.mean_uncertainty is not None:
            self._set_text(self.lbl_uncertainty, f"Mean uncertainty: {state.mean_uncertainty*100:.1f}%")
        else:
            self._set_text(self.lbl_uncertainty, "Mean uncertainty: N/A")
        if state.occupied_fraction is not None:
            self._set_text(self.lbl_occupied, f"Occupied grid fraction: {state.occupied_fraction*100:.1f}%")
        else:
            self._set_text(self.lbl_occupied, "Occupied grid fraction: N/A")

        self._update_hazard_table(state.hazard_clusters)

        self._set_text(self.lbl_frame_idx, f"Frame: {state.frame_idx} | t={state.timestamp_s:.2f}s")
        self.set_accuracy_metrics(state.has_ground_truth, state.predictor_mode, state.accuracy_metrics)
        self._set_text(self.lbl_footer, 
            f"Frame {state.frame_idx} | {len(state.points):,} pts | "
            f"{len(state.tracks)} tracks | {len(state.hazard_clusters)} hazard clusters"
            + (f" | 3D view display: {len(state.display_points):,} pts (thinned; all metrics use all points)"
               if state.display_points is not None else "")
        )

    def _update_terrain_status(self, state: FrameState):
        trav = state.grid_maps.get("traversability")
        if trav is None:
            self._set_text(self.lbl_terrain_status, "UNKNOWN")
            self._set_style(self.lbl_terrain_status, f"font-size: {FONT_SIZE_HERO}; font-weight: 700; color: {COLOR_TEXT_DIM};")
            self._set_text(self.lbl_terrain_confidence, "Confidence: N/A")
            return

        valid = ~np.isnan(trav)
        if not np.any(valid):
            self._set_text(self.lbl_terrain_status, "UNKNOWN")
            self._set_style(self.lbl_terrain_status, f"font-size: {FONT_SIZE_HERO}; font-weight: 700; color: {COLOR_TEXT_DIM};")
            self._set_text(self.lbl_terrain_confidence, "Confidence: N/A")
            return

        avg_trav = float(np.nanmean(trav[valid]))
        if avg_trav >= TRAVERSABILITY_SAFE_MIN:
            text, color = "DRIVABLE", COLOR_SAFE
        elif avg_trav >= TRAVERSABILITY_CAUTION_MIN:
            text, color = "CAUTION", COLOR_CAUTION
        else:
            text, color = "NON-DRIVABLE", COLOR_CRITICAL

        self._set_text(self.lbl_terrain_status, text)
        self._set_style(self.lbl_terrain_status, f"font-size: {FONT_SIZE_HERO}; font-weight: 700; color: {color};")
        self._set_text(self.lbl_terrain_confidence, f"Avg. traversability: {avg_trav:.2f}")

    def _update_hazard_table(self, clusters: list[dict]):
        """Show the real hazard clusters, rewriting only cells whose text
        changed (rebuilding every item every frame is what made this slow)."""
        table = self.hazard_table
        if table.rowCount() != len(clusters):
            table.setRowCount(len(clusters))
        for row, c in enumerate(clusters):
            extent = c.get("extent_m")
            cells = (
                f"{c['distance_m']:.2f}",
                f"{c['min_traversability']:.2f}",
                c["severity"],
                f"{c['confidence']*100:.0f}%",
                c.get("kind", "Rough"),
                f"{extent:.2f}" if extent is not None else "N/A",
            )
            for col, text in enumerate(cells):
                item = table.item(row, col)
                if item is None:
                    item = QTableWidgetItem(text)
                    table.setItem(row, col, item)
                elif item.text() != text:
                    item.setText(text)
                if col == 2:
                    color = COLOR_CRITICAL if c["severity"] == "Critical" else COLOR_CAUTION
                    if item.foreground().color().name().lower() != QColor(color).name().lower():
                        item.setForeground(QColor(color))

    def _render_overhead_to_label(self, trav_grid, overhead_grid):
        """Render the OVERHEAD layer: real terrain coloring for ordinary
        ground cells, a solid fill for solid_obstacle cells, a cross-hatch
        overlay for overhead_obstacle cells (real vertical clearance
        underneath, from src/perception/overhead_detection.py), and a
        muted fill for indeterminate (too-sparse) cells.
        """
        rgba = overhead_to_rgba8(trav_grid, overhead_grid)
        if rgba is None:
            self.map_label.setText("Overhead clearance map not available.")
            return
        h, w = rgba.shape[:2]
        q_img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.map_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.map_label.setPixmap(scaled_pixmap)

    def _render_grid_to_label(self, grid: np.ndarray, map_type: str):
        rgba_8 = grid_to_rgba8(grid, map_type)
        if rgba_8 is None:
            self.map_label.setText("Grid is empty")
            return
        h, w, c = rgba_8.shape
        q_img = QImage(rgba_8.data, w, h, c * w, QImage.Format_RGBA8888)

        # Scale to fit label while keeping aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.map_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.map_label.setPixmap(scaled_pixmap)


class LoadDatasetDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Custom Dataset")
        self.setMinimumWidth(500)
        self.setStyleSheet(STYLESHEET)
        
        layout = QVBoxLayout(self)
        
        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(QLabel("Format:"))
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["SemanticKITTI", "RELLIS-3D"])
        self.fmt_combo.currentTextChanged.connect(self._on_format_changed)
        fmt_layout.addWidget(self.fmt_combo)
        layout.addLayout(fmt_layout)
        
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        path_layout.addWidget(self.path_edit)
        self.browse_btn = QPushButton("Browse...")
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)
        self.browse_btn.clicked.connect(self._browse)
        
        seq_layout = QHBoxLayout()
        seq_layout.addWidget(QLabel("Sequence:"))
        self.seq_combo = QComboBox()
        self.seq_combo.currentIndexChanged.connect(self._check_ready)
        seq_layout.addWidget(self.seq_combo)
        layout.addLayout(seq_layout)
        
        adv_group = QGroupBox("Advanced Thresholds")
        adv_layout = QGridLayout()
        
        # Row 0: ground_noise_m and clearance_threshold_m
        adv_layout.addWidget(QLabel("ground_noise_m:"), 0, 0)
        self.gn_spin = QDoubleSpinBox()
        self.gn_spin.setDecimals(2)
        self.gn_spin.setValue(0.30)
        self.gn_spin.setSingleStep(0.05)
        adv_layout.addWidget(self.gn_spin, 0, 1)

        adv_layout.addWidget(QLabel("clearance_threshold_m:"), 0, 2)
        self.ct_spin = QDoubleSpinBox()
        self.ct_spin.setDecimals(2)
        self.ct_spin.setValue(2.00)
        self.ct_spin.setSingleStep(0.10)
        adv_layout.addWidget(self.ct_spin, 0, 3)

        # Row 1: min_points and cluster_min_points
        adv_layout.addWidget(QLabel("min_points (overhead):"), 1, 0)
        self.mp_spin = QSpinBox()
        self.mp_spin.setValue(4)
        adv_layout.addWidget(self.mp_spin, 1, 1)

        adv_layout.addWidget(QLabel("cluster_min_points:"), 1, 2)
        self.cmp_spin = QSpinBox()
        self.cmp_spin.setValue(5)
        adv_layout.addWidget(self.cmp_spin, 1, 3)

        # Row 2: voxel_size_m and ground_threshold_m
        adv_layout.addWidget(QLabel("voxel_size_m:"), 2, 0)
        self.vs_spin = QDoubleSpinBox()
        self.vs_spin.setDecimals(2)
        self.vs_spin.setValue(0.20)
        self.vs_spin.setSingleStep(0.05)
        adv_layout.addWidget(self.vs_spin, 2, 1)

        adv_layout.addWidget(QLabel("ground_threshold_m:"), 2, 2)
        self.gt_spin = QDoubleSpinBox()
        self.gt_spin.setDecimals(2)
        self.gt_spin.setValue(0.25)
        self.gt_spin.setSingleStep(0.05)
        adv_layout.addWidget(self.gt_spin, 2, 3)

        adv_group.setLayout(adv_layout)
        layout.addWidget(adv_group)
        
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load")
        self.load_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        self.load_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def _on_format_changed(self):
        if self.path_edit.text():
            self._scan_sequences(self.path_edit.text())

    def _browse(self):
        from PyQt5.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Root Folder")
        if folder:
            self.path_edit.setText(folder)
            self._scan_sequences(folder)
            
    def _scan_sequences(self, folder):
        self.seq_combo.clear()
        fmt = self.fmt_combo.currentText()
        from pathlib import Path
        root = Path(folder)
        seqs = []
        try:
            if fmt == "SemanticKITTI":
                kitti_root = root
                if (root / "dataset" / "sequences").exists():
                    kitti_root = root / "dataset" / "sequences"
                elif (root / "sequences").exists():
                    kitti_root = root / "sequences"
                for d in kitti_root.iterdir():
                    if d.is_dir() and (d / "velodyne").exists():
                        bins = list((d / "velodyne").glob("*.bin"))
                        if bins:
                            seqs.append((d.name, len(bins)))
            elif fmt == "RELLIS-3D":
                if root.name == "Rellis-3D":
                    r3d = root
                else:
                    r3d = root / "Rellis-3D"
                    if not r3d.exists():
                        r3d = root
                for d in r3d.iterdir():
                    if d.is_dir() and (d / "os1_cloud_node_kitti_bin").exists():
                        bins = list((d / "os1_cloud_node_kitti_bin").glob("*.bin"))
                        if bins:
                            seqs.append((d.name, len(bins)))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to scan folder: {e}")
            
        if not seqs:
            p = Path(folder)
            if not p.exists():
                details = "Folder does not exist."
            else:
                try:
                    entries = list(p.iterdir())
                    sub_dirs = [e.name for e in entries if e.is_dir()][:8]
                    sub_files = [e.name for e in entries if e.is_file()][:8]
                    total_dirs = len([e for e in entries if e.is_dir()])
                    total_files = len([e for e in entries if e.is_file()])
                    details = f"Folder contains {len(entries)} items ({total_dirs} directories, {total_files} files).\n"
                    if sub_dirs:
                        details += f"Directories found: {', '.join(sub_dirs)}{' ...' if len(sub_dirs) < total_dirs else ''}\n"
                    if sub_files:
                        details += f"Files found: {', '.join(sub_files)}{' ...' if len(sub_files) < total_files else ''}\n"
                    if not entries:
                        details += "(Folder is empty)\n"
                    details += f"\nExpected {fmt} layout:\n"
                    if fmt == "SemanticKITTI":
                        details += "  - sequences/<id>/velodyne/*.bin\n  (or dataset/sequences/<id>/velodyne/*.bin)"
                    else:
                        details += "  - Rellis-3D/<id>/os1_cloud_node_kitti_bin/*.bin\n  (or <id>/os1_cloud_node_kitti_bin/*.bin)"
                except Exception as ex:
                    details = f"Unable to list folder contents: {ex}"
            QMessageBox.information(self, "No Sequences", f"No valid {fmt} sequences found in:\n{folder}\n\n{details}")
            return
            
        seqs.sort()
        for name, count in seqs:
            self.seq_combo.addItem(f"{name} ({count} frames)", userData=name)
        
    def _check_ready(self):
        self.load_btn.setEnabled(self.seq_combo.count() > 0)

    def get_result(self):
        fmt = "semantickitti" if self.fmt_combo.currentText() == "SemanticKITTI" else "rellis3d"
        path = self.path_edit.text()
        from pathlib import Path
        if fmt == "rellis3d" and Path(path).name == "Rellis-3D":
            path = str(Path(path).parent)
        elif fmt == "semantickitti":
            p = Path(path)
            if (p / "dataset" / "sequences").exists():
                path = str(p / "dataset" / "sequences")
            elif (p / "sequences").exists():
                path = str(p / "sequences")
        seq = self.seq_combo.currentData()
        kwargs = {
            "ground_noise_m": self.gn_spin.value(),
            "clearance_threshold_m": self.ct_spin.value(),
            "min_points": self.mp_spin.value(),
            "cluster_min_points": self.cmp_spin.value(),
            "voxel_size_m": self.vs_spin.value(),
            "ground_threshold_m": self.gt_spin.value(),
        }
        return fmt, path, seq, kwargs

