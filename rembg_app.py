"""
RemBG — AI & Lüminan Tabanlı Arkaplan Silici
Desteklenen Modlar:
  1. AI Segmentasyon  (rembg / U2Net, ISNet …)
  2. Koyu Arkaplan Sil (neon, çizgi sanatı, siyah bg)
  3. Açık Arkaplan Sil (beyaz / açık bg, flat tasarım)
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from collections import Counter
import os
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

# ─────────────────────────────────────────────
#  COLOUR TOKENS
# ─────────────────────────────────────────────
BG_DARK      = "#0d0d14"
BG_PANEL     = "#12121e"
BG_CARD      = "#1a1a2e"
BG_CARD2     = "#16213e"
ACCENT       = "#7c3aed"
ACCENT2      = "#a855f7"
ACCENT_GLOW  = "#c084fc"
SUCCESS      = "#22c55e"
WARNING      = "#f59e0b"
ERROR        = "#ef4444"
CYAN         = "#06b6d4"
TEXT_MAIN    = "#f1f0ff"
TEXT_DIM     = "#9ca3af"
TEXT_MUTED   = "#6b7280"
BORDER       = "#2d2b55"
BTN_HOVER    = "#6d28d9"
SUPPORTED    = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Renk önayarları: off | auto | vivid | soft
COLOR_PRESET_LABELS = {
    "off": "Kapalı",
    "auto": "Otomatik (seviye + ışık)",
    "vivid": "Canlı (renk + kontrast)",
    "soft": "Yumuşak (solukları aç)",
}


def human_size(n):
    for u in ("B","KB","MB","GB"):
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def load_image_exif_safe(path: Path | str) -> Image.Image:
    """EXIF Orientation uygula — dikey JPG/HEIC vb. için boyut/oran doğru olsun."""
    img = Image.open(path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img


# ─────────────────────────────────────────────
#  BACKGROUND REMOVAL ROUTINES
# ─────────────────────────────────────────────

def remove_dark_bg(img: Image.Image, threshold: int, softness: int,
                   despill: bool) -> Image.Image:
    """
    Siyah / koyu arkaplanı sil.
    Lüminan değeri düşük pikseller şeffaflaştırılır.
    Neon çizgi sanatı, tel kafes (wireframe) görseller için idealdir.
    """
    rgba = img.convert("RGBA")
    data = np.array(rgba, dtype=np.float32)

    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]

    # Perceived luminance (ITU-R BT.709)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Soft ramp: below threshold → transparent, above threshold+softness → opaque
    soft = max(softness, 1)
    alpha = np.clip((lum - threshold) / soft * 255.0, 0.0, 255.0)

    data[:, :, 3] = alpha

    if despill:
        # Boost saturation of semi-transparent fringe pixels
        # to reduce dark halo at edges
        mask = (alpha > 0) & (alpha < 200)
        factor = 1.3
        mx = np.maximum(r, np.maximum(g, b))
        for ch in [0, 1, 2]:
            data[:, :, ch] = np.where(
                mask,
                np.clip(mx + (data[:, :, ch] - mx) * factor, 0, 255),
                data[:, :, ch]
            )

    return Image.fromarray(data.astype(np.uint8), "RGBA")


def remove_light_bg(img: Image.Image, threshold: int, softness: int,
                    despill: bool) -> Image.Image:
    """
    Beyaz / açık arkaplanı sil.
    Lüminan değeri yüksek pikseller şeffaflaştırılır.
    Tarama, logo, flat illüstrasyon görseller için idealdir.
    """
    rgba = img.convert("RGBA")
    data = np.array(rgba, dtype=np.float32)

    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    soft = max(softness, 1)
    # above threshold → transparent
    alpha = np.clip((threshold - lum) / soft * 255.0 + 255.0, 0.0, 255.0)

    data[:, :, 3] = alpha

    if despill:
        mask = (alpha > 0) & (alpha < 200)
        factor = 1.3
        mn = np.minimum(r, np.minimum(g, b))
        for ch in [0, 1, 2]:
            data[:, :, ch] = np.where(
                mask,
                np.clip(mn + (data[:, :, ch] - mn) * factor, 0, 255),
                data[:, :, ch]
            )

    return Image.fromarray(data.astype(np.uint8), "RGBA")


def remove_ai_bg(img: Image.Image, session, alpha_matting: bool,
                 fg_thresh: int, bg_thresh: int, erode: int) -> Image.Image:
    """rembg ile AI segmentasyon."""
    from rembg import remove
    result = remove(
        img,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=fg_thresh,
        alpha_matting_background_threshold=bg_thresh,
        alpha_matting_erode_size=erode,
    )
    if result.mode != "RGBA":
        result = result.convert("RGBA")
    if result.size != img.size:
        result = result.resize(img.size, Image.LANCZOS)
    return result


def trim_transparent_rgba(
    img: Image.Image,
    margin_ratio: float = 0.04,
    min_margin: int = 14,
    alpha_thresh: int = 8,
) -> Image.Image:
    """
    Alfa kanalına göre içeriği saran bbox + pay ile kırpar; nesneyi görüntüde ortalar.
    Kenar payı: max(genişlik,yükseklik) * margin_ratio ile alt sınır min_margin.
    """
    rgba = img.convert("RGBA")
    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    ys, xs = np.where(alpha > alpha_thresh)
    if xs.size == 0:
        return rgba
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = rgba.size
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    m = max(min_margin, int(max(bw, bh) * margin_ratio))
    x0 = max(0, x0 - m)
    y0 = max(0, y0 - m)
    x1 = min(w - 1, x1 + m)
    y1 = min(h - 1, y1 + m)
    return rgba.crop((x0, y0, x1 + 1, y1 + 1))


def auto_levels_rgba(
    img: Image.Image,
    alpha_thresh: int = 20,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> Image.Image:
    """
    Ön plandaki piksellerin (alfa > eşik) histogramına göre R/G/B kanallarını gerer;
    düz / soluk ürün fotoğraflarında ışığı canlandırır.
    """
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
    a = arr[:, :, 3]
    m = a > alpha_thresh
    if not np.any(m):
        return img
    out = arr.copy()
    for c in range(3):
        sample = arr[:, :, c][m]
        lo, hi = np.percentile(sample, [low_pct, high_pct])
        lo = float(np.clip(lo, 0.0, 254.0))
        hi = float(np.maximum(hi, lo + 1.0))
        plane = out[:, :, c]
        scaled = (plane - lo) / (hi - lo) * 255.0
        out[:, :, c] = np.clip(scaled, 0, 255)
    out[:, :, 3] = arr[:, :, 3]
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def pil_enhance_rgb_keep_alpha(
    rgba: Image.Image,
    *,
    brightness: float = 1.0,
    contrast: float = 1.0,
    color: float = 1.0,
    sharpness: float = 1.0,
) -> Image.Image:
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    if brightness != 1.0:
        rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    if contrast != 1.0:
        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    if color != 1.0:
        rgb = ImageEnhance.Color(rgb).enhance(color)
    if sharpness != 1.0:
        rgb = ImageEnhance.Sharpness(rgb).enhance(sharpness)
    r, g, b = rgb.split()
    return Image.merge("RGBA", (r, g, b, a))


def apply_color_preset(img: Image.Image, preset: str) -> Image.Image:
    """Arka plan silindikten sonra uygulanır; alfa korunur."""
    if not preset or preset == "off":
        return img
    if preset == "auto":
        x = auto_levels_rgba(img, alpha_thresh=20, low_pct=2.0, high_pct=98.0)
        return pil_enhance_rgb_keep_alpha(
            x, brightness=1.05, contrast=1.06, color=1.08, sharpness=1.0,
        )
    if preset == "vivid":
        x = auto_levels_rgba(img, alpha_thresh=15, low_pct=1.5, high_pct=98.5)
        return pil_enhance_rgb_keep_alpha(
            x, brightness=1.03, contrast=1.14, color=1.22, sharpness=1.08,
        )
    if preset == "soft":
        x = auto_levels_rgba(img, alpha_thresh=25, low_pct=5.0, high_pct=95.0)
        return pil_enhance_rgb_keep_alpha(
            x, brightness=1.07, contrast=1.05, color=1.06, sharpness=1.02,
        )
    return img


def downscale_longest_side(img: Image.Image, max_side: int) -> Image.Image:
    """max_side > 0 ise uzun kenarı bu değere indir (en-boy oranı korunur)."""
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    ratio = max_side / float(m)
    nw = max(1, int(round(w * ratio)))
    nh = max(1, int(round(h * ratio)))
    return img.resize((nw, nh), Image.LANCZOS)


def save_rgba_png_compressed(
    img: Image.Image,
    path: Path,
    *,
    max_side: int = 0,
    compress_level: int = 9,
) -> None:
    """zlib seviye 9 + optimize; isteğe bağlı uzun kenar sınırı (dosya boyutunu düşürür)."""
    out = img.convert("RGBA")
    out = downscale_longest_side(out, max_side)
    out.save(
        path,
        "PNG",
        optimize=True,
        compress_level=min(9, max(0, compress_level)),
    )


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
class RemBGApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("RemBG Pro — Gelişmiş Arkaplan Silici")
        self.geometry("980x720")
        self.minsize(820, 580)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        self._files: list[Path] = []
        self._out_dir: Path | None = None  # None → çıktı kaynak dosyanın klasörüne
        self._running  = False
        self._stop_evt = threading.Event()
        self._done_count  = 0
        self._error_count = 0
        self._skip_count  = 0

        self._build_fonts()
        self._build_styles()
        self._build_ui()
        self._check_deps()

    # ── fonts ──────────────────────────────────
    def _build_fonts(self):
        self.F_TITLE = ("Segoe UI", 20, "bold")
        self.F_SUB   = ("Segoe UI", 10)
        self.F_LABEL = ("Segoe UI", 9, "bold")
        self.F_SMALL = ("Segoe UI", 9)
        self.F_LOG   = ("Consolas", 9)
        self.F_BTN   = ("Segoe UI", 10, "bold")
        self.F_STAT  = ("Segoe UI", 18, "bold")
        self.F_BADGE = ("Segoe UI", 9, "bold")

    # ── ttk styles ─────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Violet.Horizontal.TProgressbar",
                     troughcolor=BG_CARD, background=ACCENT2,
                     bordercolor=BG_CARD, lightcolor=ACCENT2,
                     darkcolor=ACCENT, thickness=8)
        s.configure("Dark.Vertical.TScrollbar",
                     troughcolor=BG_CARD, background=BORDER,
                     bordercolor=BG_CARD, arrowcolor=TEXT_DIM)
        s.map("Dark.Vertical.TScrollbar",
              background=[("active", ACCENT)])
        s.configure("TScale", background=BG_CARD,
                    troughcolor=BORDER, sliderlength=14)

    # ── UI skeleton ────────────────────────────
    def _build_ui(self):
        # HEADER
        hdr = tk.Frame(self, bg=BG_PANEL, height=62)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="✦ RemBG Pro", font=self.F_TITLE,
                 bg=BG_PANEL, fg=ACCENT_GLOW).pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(hdr, text="Gelişmiş Arkaplan Silici  ·  AI + kırpma",
                 font=self.F_SUB, bg=BG_PANEL, fg=TEXT_DIM).pack(side=tk.LEFT)
        self._status_badge = tk.Label(hdr, text="● Hazır", font=self.F_BADGE,
                                      bg=BG_PANEL, fg=SUCCESS)
        self._status_badge.pack(side=tk.RIGHT, padx=20)
        tk.Frame(self, bg=BORDER, height=1).pack(fill=tk.X)

        # BODY
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Sol panel: kaydırılabilir (uzun formlar için)
        left_outer = tk.Frame(body, bg=BG_DARK, width=348)
        left_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        left_outer.pack_propagate(False)

        self._left_canvas = tk.Canvas(
            left_outer, bg=BG_DARK, highlightthickness=0, bd=0,
        )
        self._left_scroll = ttk.Scrollbar(
            left_outer, orient=tk.VERTICAL,
            command=self._left_canvas.yview,
            style="Dark.Vertical.TScrollbar",
        )
        self._left_canvas.configure(yscrollcommand=self._left_scroll.set)
        self._left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        left_inner = tk.Frame(self._left_canvas, bg=BG_DARK)
        self._left_canvas_window = self._left_canvas.create_window(
            (0, 0), window=left_inner, anchor=tk.NW,
        )

        def _left_inner_cfg(_event=None):
            self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all"))

        def _left_canvas_cfg(event):
            self._left_canvas.itemconfigure(self._left_canvas_window, width=event.width)

        left_inner.bind("<Configure>", _left_inner_cfg)
        self._left_canvas.bind("<Configure>", _left_canvas_cfg)

        def _wheel(event):
            self._left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_wheel(_e=None):
            self._left_canvas.bind_all("<MouseWheel>", _wheel)

        def _unbind_wheel(_e=None):
            self._left_canvas.unbind_all("<MouseWheel>")

        left_outer.bind("<Enter>", _bind_wheel)
        left_outer.bind("<Leave>", _unbind_wheel)

        self._build_left(left_inner)

        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_right(right)

        self._build_statusbar()

    # ── LEFT PANEL ─────────────────────────────
    def _build_left(self, p):
        # ── MODE SELECTOR ──────────────────────
        self._section(p, "⚙️  İşlem Modu")
        self._mode_var = tk.StringVar(value="dark")

        mode_frame = tk.Frame(p, bg=BG_CARD,
                              highlightbackground=BORDER, highlightthickness=1)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        modes = [
            ("dark",  "🌑  Koyu Arkaplan Sil",
             "Siyah / koyu bg — Neon, tel kafes, çizgi sanatı"),
            ("light", "☀️  Açık Arkaplan Sil",
             "Beyaz / açık bg — Logo, tarama, flat illüstrasyon"),
            ("ai_studio", "🎯  AI Nesne Ayırma (stüdyo / ürün)",
             "Beyaz-gri arka plan fark etmez — nesneyi korur, çevreyi siler (ISNet)"),
            ("ai",    "🤖  AI Segmentasyon (rembg)",
             "Model seç — genel foto, insan, detay"),
        ]
        for val, label, tip in modes:
            rb_frame = tk.Frame(mode_frame, bg=BG_CARD)
            rb_frame.pack(fill=tk.X)
            rb = tk.Radiobutton(rb_frame, text=label, variable=self._mode_var,
                                value=val, font=self.F_SMALL,
                                bg=BG_CARD, fg=TEXT_MAIN, selectcolor=ACCENT,
                                activebackground=BG_CARD,
                                activeforeground=ACCENT_GLOW, bd=0,
                                cursor="hand2",
                                command=self._on_mode_change)
            rb.pack(anchor=tk.W, padx=10, pady=(5, 0))
            tk.Label(rb_frame, text=tip, font=("Segoe UI", 8),
                     bg=BG_CARD, fg=TEXT_MUTED).pack(anchor=tk.W,
                                                      padx=26, pady=(0, 5))

        # ── LUM PANEL (dark / light modes) ─────
        self._lum_panel = tk.Frame(p, bg=BG_DARK)
        self._lum_panel.pack(fill=tk.X)
        self._build_lum_panel(self._lum_panel)

        # ── AI PANEL ───────────────────────────
        self._ai_panel = tk.Frame(p, bg=BG_DARK)
        self._ai_panel.pack(fill=tk.X)
        self._build_ai_panel(self._ai_panel)

        # Stüdyo modu bilgi (sadece ai_studio)
        self._studio_frame = tk.Frame(p, bg=BG_CARD,
                                      highlightbackground=BORDER, highlightthickness=1)
        tk.Label(
            self._studio_frame,
            text="ISNet — varsayılan hızlı (alpha matting kapalı).\n"
                 "İstersen aşağıdan matting aç; yumuşak saçak ama çok daha yavaş (CPU).",
            font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_MUTED, justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=10, pady=(8, 4))
        self._studio_matting_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self._studio_frame,
            text="Alpha matting (yumuşak kenar — yavaş, pymatting)",
            variable=self._studio_matting_var,
            font=self.F_SMALL,
            bg=BG_CARD, fg=TEXT_MAIN, selectcolor=ACCENT,
            activebackground=BG_CARD, activeforeground=ACCENT_GLOW,
            bd=0, cursor="hand2",
        ).pack(anchor=tk.W, padx=10, pady=(0, 8))

        # Çıktı: kırpma
        self._section(p, "📐  Çıktı")
        out_frm = tk.Frame(p, bg=BG_CARD,
                           highlightbackground=BORDER, highlightthickness=1)
        out_frm.pack(fill=tk.X, pady=(0, 8))
        self._trim_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            out_frm,
            text="Fazla boşluğu kırp ve nesneyi ortala (şeffaf kenarları at)",
            variable=self._trim_var, font=self.F_SMALL,
            bg=BG_CARD, fg=TEXT_MAIN, selectcolor=ACCENT,
            activebackground=BG_CARD, activeforeground=ACCENT_GLOW,
            bd=0, cursor="hand2",
        ).pack(anchor=tk.W, padx=10, pady=(8, 4))
        row_m = tk.Frame(out_frm, bg=BG_CARD)
        row_m.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Label(row_m, text="Kenar payı (%)", font=self.F_SMALL,
                 bg=BG_CARD, fg=TEXT_DIM).pack(side=tk.LEFT)
        self._margin_lbl = tk.Label(row_m, text="4", width=4,
                                    font=self.F_SMALL, bg=BG_CARD, fg=ACCENT_GLOW)
        self._margin_lbl.pack(side=tk.RIGHT)
        self._margin_var = tk.IntVar(value=4)
        ttk.Scale(
            out_frm, from_=2, to=12,
            variable=self._margin_var, orient=tk.HORIZONTAL,
            command=lambda v: self._margin_lbl.config(text=str(int(float(v)))),
        ).pack(fill=tk.X, padx=10, pady=(0, 6))

        row_side = tk.Frame(out_frm, bg=BG_CARD)
        row_side.pack(fill=tk.X, padx=10, pady=(0, 2))
        tk.Label(row_side, text="Max uzun kenar (px)", font=self.F_SMALL,
                 bg=BG_CARD, fg=TEXT_MAIN).pack(side=tk.LEFT)
        self._max_side_lbl = tk.Label(row_side, text="2048", width=5,
                                      font=self.F_SMALL, bg=BG_CARD, fg=ACCENT_GLOW)
        self._max_side_lbl.pack(side=tk.RIGHT)
        self._max_side_var = tk.IntVar(value=2048)
        ttk.Scale(
            out_frm, from_=0, to=4096,
            variable=self._max_side_var, orient=tk.HORIZONTAL,
            command=lambda v: self._max_side_lbl.config(
                text=str(int(float(v))) if int(float(v)) > 0 else "0 (tam)"
            ),
        ).pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Label(
            out_frm,
            text="0 = çözünürlük aynı, yalnızca PNG sıkıştırma  |  2048 ≈ web/ürün (~2–4 MB)\n"
                 "Düşük değer = daha küçük dosya, biraz daha yumuşak detay",
            font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_MUTED, justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=10, pady=(0, 6))

        tk.Frame(out_frm, bg=BORDER, height=1).pack(fill=tk.X, padx=10, pady=(2, 6))
        tk.Label(out_frm, text="Renk / ışık", font=self.F_LABEL,
                 bg=BG_CARD, fg=ACCENT_GLOW).pack(anchor=tk.W, padx=10, pady=(0, 4))
        self._color_preset = tk.StringVar(value="off")
        color_opts = [
            ("off", "Kapalı", "Sadece kesim; renge dokunulmaz"),
            ("auto", "Otomatik", "Histogram denge + hafif parlaklık / doygunluk"),
            ("vivid", "Canlı", "Doygunluk, kontrast, hafif keskinlik"),
            ("soft", "Yumuşak", "Soluk görselleri nazikçe açar"),
        ]
        for val, title, tip in color_opts:
            cf = tk.Frame(out_frm, bg=BG_CARD)
            cf.pack(fill=tk.X)
            tk.Radiobutton(
                cf, text=title, variable=self._color_preset, value=val,
                font=self.F_SMALL, bg=BG_CARD, fg=TEXT_MAIN, selectcolor=ACCENT,
                activebackground=BG_CARD, activeforeground=ACCENT_GLOW,
                bd=0, cursor="hand2",
            ).pack(anchor=tk.W, padx=10, pady=(3, 0))
            tk.Label(cf, text=tip, font=("Segoe UI", 8),
                     bg=BG_CARD, fg=TEXT_MUTED).pack(anchor=tk.W, padx=28, pady=(0, 4))

        self._on_mode_change()          # show correct panel

        # ── FILE / FOLDER ──────────────────────
        self._section(p, "📂  Dosya / Klasör Seç")
        btn_wrap = tk.Frame(p, bg=BG_DARK)
        btn_wrap.pack(fill=tk.X, pady=(0, 4))
        self._mk_btn(btn_wrap, "📄  Dosya Seç",    self._pick_files).pack(fill=tk.X, pady=2)
        self._mk_btn(btn_wrap, "📁  Klasör Seç",   self._pick_folder).pack(fill=tk.X, pady=2)
        self._mk_btn(btn_wrap, "🗑️  Listeyi Temizle",
                     self._clear_queue, bg=BG_CARD2,
                     hover=BG_CARD, color=TEXT_MUTED).pack(fill=tk.X, pady=2)

        self._section(p, "💾  Çıktı klasörü")
        outdir_card = tk.Frame(p, bg=BG_CARD,
                               highlightbackground=BORDER, highlightthickness=1)
        outdir_card.pack(fill=tk.X, pady=(0, 8))
        self._outdir_lbl = tk.Label(
            outdir_card,
            text="Kaynak dosyanın yanına kaydet (varsayılan)",
            font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_MUTED,
            wraplength=300, justify=tk.LEFT,
        )
        self._outdir_lbl.pack(anchor=tk.W, padx=10, pady=(8, 6))
        od_row = tk.Frame(outdir_card, bg=BG_CARD)
        od_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        self._mk_btn(od_row, "📂  Klasör seç", self._pick_out_dir,
                     font=self.F_SMALL, pady=6).pack(side=tk.LEFT, fill=tk.X,
                                                      expand=True, padx=(0, 4))
        self._mk_btn(od_row, "↩  Varsayılan", self._reset_out_dir,
                     bg=BG_CARD2, hover=BG_CARD, color=TEXT_MUTED,
                     font=self.F_SMALL, pady=6).pack(side=tk.LEFT, fill=tk.X,
                                                     expand=True)

        self._queue_lbl = tk.Label(p, text="0 dosya sıraya eklendi",
                                   font=self.F_SMALL, bg=BG_DARK, fg=TEXT_DIM)
        self._queue_lbl.pack(pady=(2, 8))

        # ── RUN / STOP ─────────────────────────
        self._btn_run = self._mk_btn(p, "▶   İşlemi Başlat",
                                     self._start,
                                     bg=ACCENT, hover=BTN_HOVER,
                                     color=TEXT_MAIN, font=self.F_BTN, pady=11)
        self._btn_run.pack(fill=tk.X, pady=(0, 3))

        self._btn_stop = self._mk_btn(p, "⏹   Durdur",
                                      self._stop,
                                      bg=ERROR, hover="#b91c1c",
                                      color=TEXT_MAIN, font=self.F_BTN, pady=8)
        self._btn_stop.pack(fill=tk.X)
        self._btn_stop.config(state=tk.DISABLED)

        # ── STATS ──────────────────────────────
        stat_row = tk.Frame(p, bg=BG_DARK)
        stat_row.pack(fill=tk.X, pady=(14, 0))
        self._stat_done  = self._mk_stat(stat_row, "0", "Tamamlandı", SUCCESS)
        self._stat_error = self._mk_stat(stat_row, "0", "Hata",       ERROR)
        self._stat_skip  = self._mk_stat(stat_row, "0", "Atlandı",    WARNING)

    # ── LUM PANEL ──────────────────────────────
    def _build_lum_panel(self, p):
        self._section(p, "🎛️  Lüminan Ayarları")
        frm = tk.Frame(p, bg=BG_CARD,
                       highlightbackground=BORDER, highlightthickness=1)
        frm.pack(fill=tk.X, pady=(0, 8))

        # Threshold
        row1 = tk.Frame(frm, bg=BG_CARD)
        row1.pack(fill=tk.X, padx=10, pady=(8, 2))
        tk.Label(row1, text="Eşik (Threshold)", font=self.F_SMALL,
                 bg=BG_CARD, fg=TEXT_MAIN).pack(side=tk.LEFT)
        self._thresh_lbl = tk.Label(row1, text="35", width=3,
                                    font=self.F_SMALL, bg=BG_CARD, fg=ACCENT_GLOW)
        self._thresh_lbl.pack(side=tk.RIGHT)

        self._thresh_var = tk.IntVar(value=35)
        thresh_scale = ttk.Scale(frm, from_=0, to=200,
                                 variable=self._thresh_var, orient=tk.HORIZONTAL,
                                 command=lambda v: self._thresh_lbl.config(
                                     text=str(int(float(v)))))
        thresh_scale.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Label(frm, text="Düşük → daha az şeffaf  |  Yüksek → daha geniş silme",
                 font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_MUTED).pack(
                     padx=10, anchor=tk.W, pady=(0, 6))

        # Softness
        row2 = tk.Frame(frm, bg=BG_CARD)
        row2.pack(fill=tk.X, padx=10, pady=(4, 2))
        tk.Label(row2, text="Yumuşaklık (Softness)", font=self.F_SMALL,
                 bg=BG_CARD, fg=TEXT_MAIN).pack(side=tk.LEFT)
        self._soft_lbl = tk.Label(row2, text="25", width=3,
                                  font=self.F_SMALL, bg=BG_CARD, fg=ACCENT_GLOW)
        self._soft_lbl.pack(side=tk.RIGHT)

        self._soft_var = tk.IntVar(value=25)
        soft_scale = ttk.Scale(frm, from_=1, to=120,
                               variable=self._soft_var, orient=tk.HORIZONTAL,
                               command=lambda v: self._soft_lbl.config(
                                   text=str(int(float(v)))))
        soft_scale.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Label(frm, text="Düşük → sert kenar  |  Yüksek → geçişli kenar",
                 font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_MUTED).pack(
                     padx=10, anchor=tk.W, pady=(0, 6))

        # Despill
        self._despill_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frm, text="Kenar rengi düzeltme (despill)",
                       variable=self._despill_var, font=self.F_SMALL,
                       bg=BG_CARD, fg=TEXT_MAIN, selectcolor=ACCENT,
                       activebackground=BG_CARD, activeforeground=ACCENT_GLOW,
                       bd=0, cursor="hand2").pack(anchor=tk.W, padx=10,
                                                  pady=(0, 8))

    # ── AI PANEL ───────────────────────────────
    def _build_ai_panel(self, p):
        self._section(p, "🤖  AI Model & Ayarlar")
        frm = tk.Frame(p, bg=BG_CARD,
                       highlightbackground=BORDER, highlightthickness=1)
        frm.pack(fill=tk.X, pady=(0, 8))

        self._model_var = tk.StringVar(value="isnet-general-use")
        models = [
            ("isnet-general-use", "ISNet  – Yüksek Detay (Önerilen)"),
            ("u2net",             "U2Net  – Genel Amaç"),
            ("u2net_human_seg",   "U2Net Human – İnsan / Portre"),
            ("silueta",           "Silueta – Hafif & Hızlı"),
            ("birefnet-general",  "BiRefNet – Ultra Detay (Yavaş)"),
        ]
        for val, label in models:
            tk.Radiobutton(frm, text=label, variable=self._model_var, value=val,
                           font=self.F_SMALL, bg=BG_CARD, fg=TEXT_MAIN,
                           selectcolor=ACCENT, activebackground=BG_CARD,
                           activeforeground=ACCENT_GLOW, bd=0,
                           cursor="hand2").pack(anchor=tk.W, padx=10, pady=2)

        self._alpha_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            frm,
            text="Alpha matting (yumuşak kenar — CPU’da yavaş)",
            variable=self._alpha_var, font=self.F_SMALL,
            bg=BG_CARD, fg=TEXT_MAIN, selectcolor=ACCENT,
            activebackground=BG_CARD, activeforeground=ACCENT_GLOW,
            bd=0, cursor="hand2",
        ).pack(anchor=tk.W, padx=10, pady=(4, 8))

    # ── RIGHT LOG + PROGRESS ───────────────────
    def _build_right(self, p):
        hr = tk.Frame(p, bg=BG_DARK)
        hr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(hr, text="📋  İşlem Günlüğü",
                 font=self.F_LABEL, bg=BG_DARK, fg=TEXT_MAIN).pack(side=tk.LEFT)
        self._mk_btn(hr, "🧹 Temizle", self._clear_log,
                     color=TEXT_MUTED, hover=BG_CARD,
                     font=self.F_SMALL, pady=2).pack(side=tk.RIGHT)

        log_card = tk.Frame(p, bg=BG_CARD,
                            highlightbackground=BORDER, highlightthickness=1)
        log_card.pack(fill=tk.BOTH, expand=True)

        self._log = tk.Text(log_card, bg=BG_CARD, fg=TEXT_MAIN,
                            font=self.F_LOG, bd=0, wrap=tk.WORD,
                            state=tk.DISABLED,
                            selectbackground=ACCENT)
        sb = ttk.Scrollbar(log_card, command=self._log.yview,
                            style="Dark.Vertical.TScrollbar")
        self._log.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._log.tag_config("info",  foreground=TEXT_DIM)
        self._log.tag_config("ok",    foreground=SUCCESS)
        self._log.tag_config("err",   foreground=ERROR)
        self._log.tag_config("warn",  foreground=WARNING)
        self._log.tag_config("title", foreground=ACCENT_GLOW)
        self._log.tag_config("dim",   foreground=TEXT_MUTED)
        self._log.tag_config("cyan",  foreground=CYAN)

        # Progress
        pg = tk.Frame(p, bg=BG_DARK)
        pg.pack(fill=tk.X, pady=(8, 0))
        self._prog_lbl = tk.Label(pg, text="", font=self.F_SMALL,
                                  bg=BG_DARK, fg=TEXT_DIM)
        self._prog_lbl.pack(anchor=tk.W)
        self._bar = ttk.Progressbar(pg, mode="determinate",
                                     style="Violet.Horizontal.TProgressbar")
        self._bar.pack(fill=tk.X, pady=(3, 0))

    # ── STATUS BAR ─────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self, bg=BG_PANEL, height=26)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)
        tk.Frame(bar, bg=BORDER, height=1).pack(fill=tk.X, side=tk.TOP)
        self._sb = tk.Label(bar, text="RemBG Pro hazır.",
                            font=self.F_SMALL, bg=BG_PANEL, fg=TEXT_MUTED)
        self._sb.pack(side=tk.LEFT, padx=12)

    # ── HELPER WIDGETS ─────────────────────────
    def _section(self, parent, title):
        tk.Label(parent, text=title, font=self.F_LABEL,
                 bg=BG_DARK, fg=ACCENT_GLOW).pack(anchor=tk.W, pady=(10, 4))

    def _mk_btn(self, parent, text, cmd, bg=BG_CARD2, hover=BTN_HOVER,
                color=TEXT_MAIN, font=None, pady=6):
        font = font or self.F_BTN
        b = tk.Button(parent, text=text, command=cmd, font=font,
                       bg=bg, fg=color, activebackground=hover,
                       activeforeground=TEXT_MAIN, bd=0,
                       padx=12, pady=pady, cursor="hand2",
                       relief=tk.FLAT)
        b.bind("<Enter>", lambda e, _b=b, _h=hover: _b.config(bg=_h))
        b.bind("<Leave>", lambda e, _b=b, _bg=bg: _b.config(bg=_bg))
        return b

    def _mk_stat(self, parent, value, label, color):
        f = tk.Frame(parent, bg=BG_CARD,
                     highlightbackground=BORDER, highlightthickness=1)
        f.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)
        v = tk.Label(f, text=value, font=self.F_STAT, bg=BG_CARD, fg=color)
        v.pack(pady=(8, 0))
        tk.Label(f, text=label, font=self.F_SMALL,
                 bg=BG_CARD, fg=TEXT_MUTED).pack(pady=(0, 8))
        return v

    # ── MODE SWITCH ────────────────────────────
    def _on_mode_change(self):
        mode = self._mode_var.get()
        if mode in ("dark", "light"):
            self._lum_panel.pack(fill=tk.X)
            self._ai_panel.pack_forget()
            self._studio_frame.pack_forget()
        elif mode == "ai_studio":
            self._lum_panel.pack_forget()
            self._ai_panel.pack_forget()
            self._studio_frame.pack(fill=tk.X, pady=(0, 8))
        else:
            self._lum_panel.pack_forget()
            self._studio_frame.pack_forget()
            self._ai_panel.pack(fill=tk.X)

    # ── DEPENDENCY CHECK ───────────────────────
    def _check_deps(self):
        try:
            import rembg  # noqa
            self._log_w("✦ rembg yüklü — AI modu kullanılabilir.\n", "ok")
        except ImportError:
            self._log_w("⚠  rembg kurulu değil (AI modu çalışmaz)\n", "warn")
            self._log_w("   pip install rembg\n", "warn")
        self._log_w("✦ Lüminan modları hazır (rembg gerekmez).\n", "ok")
        self._log_w(
            "ℹ AI hızı: onnxruntime CPU (varsayılan). GPU için CUDA + onnxruntime-gpu gerekir.\n",
            "dim",
        )
        self._log_w(
            "ℹ Alpha matting kapalı = çok daha hızlı (Cholesky/pymatting yok).\n",
            "dim",
        )
        self._log_w("✦ Dosya veya klasör seçip işlemi başlatın.\n\n", "title")

    # ── FILE/FOLDER PICK ───────────────────────
    def _pick_files(self):
        files = filedialog.askopenfilenames(
            title="Dosyaları Seçin",
            filetypes=[
                ("Resim Dosyaları",
                 "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif"),
                ("Tüm Dosyalar", "*.*")])
        if not files:
            return
        added = sum(1 for f in files
                    if Path(f).suffix.lower() in SUPPORTED
                    and not self._dup(Path(f)))
        self._update_q()
        self._log_w(f"✔  {added} dosya eklendi.\n", "ok")

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Klasör Seçin")
        if not folder:
            return
        fp = Path(folder)
        added = 0
        for ext in SUPPORTED:
            for p in fp.rglob(f"*{ext}"):
                if not self._dup(p):
                    added += 1
        self._update_q()
        self._log_w(f"✔  {added} dosya eklendi: {fp.name}\n", "ok")

    def _dup(self, p: Path) -> bool:
        if p in self._files:
            return True
        self._files.append(p)
        return False

    def _update_q(self):
        n = len(self._files)
        self._queue_lbl.config(
            text=f"{n} dosya sıraya eklendi",
            fg=ACCENT_GLOW if n > 0 else TEXT_DIM)

    def _clear_queue(self):
        self._files.clear()
        self._update_q()
        self._log_w("🗑  Liste temizlendi.\n", "dim")

    def _short_path(self, p: Path, max_len: int = 52) -> str:
        s = str(p.resolve())
        if len(s) <= max_len:
            return s
        return "…" + s[-(max_len - 1) :]

    def _update_outdir_label(self):
        if self._out_dir is None:
            self._outdir_lbl.config(
                text="Kaynak dosyanın yanına kaydet (varsayılan)",
                fg=TEXT_MUTED,
            )
        else:
            self._outdir_lbl.config(
                text=self._short_path(self._out_dir),
                fg=ACCENT_GLOW,
            )

    def _pick_out_dir(self):
        d = filedialog.askdirectory(title="PNG çıktı klasörü")
        if not d:
            return
        self._out_dir = Path(d)
        try:
            self._out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Hata", f"Klasör oluşturulamadı:\n{e}")
            self._out_dir = None
            return
        self._update_outdir_label()
        self._log_w(f"✔  Çıktı klasörü: {self._out_dir.resolve()}\n", "ok")

    def _reset_out_dir(self):
        self._out_dir = None
        self._update_outdir_label()
        self._log_w("↩  Çıktı: kaynak klasörler (varsayılan).\n", "dim")

    # ── LOG ────────────────────────────────────
    def _log_w(self, msg, tag="info"):
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, msg, tag)
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _clear_log(self):
        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)

    # ── START / STOP ───────────────────────────
    def _start(self):
        if self._running:
            return
        if not self._files:
            messagebox.showwarning("Uyarı", "Önce dosya veya klasör seçin.")
            return

        mode = self._mode_var.get()
        if mode in ("ai", "ai_studio"):
            try:
                import rembg  # noqa
            except ImportError:
                messagebox.showerror(
                    "Hata",
                    "rembg kurulu değil.\n"
                    "Koyu/Açık Arkaplan modlarını kullanın\n"
                    "veya: pip install rembg")
                return

        self._running = True
        self._stop_evt.clear()
        self._btn_run.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._status_badge.config(text="● İşleniyor…", fg=WARNING)

        self._done_count = self._error_count = self._skip_count = 0
        for lbl in (self._stat_done, self._stat_error, self._stat_skip):
            lbl.config(text="0")

        params = dict(
            files     = list(self._files),
            mode      = mode,
            threshold = self._thresh_var.get(),
            softness  = self._soft_var.get(),
            despill   = self._despill_var.get(),
            model     = self._model_var.get(),
            alpha     = self._alpha_var.get(),
            trim      = self._trim_var.get(),
            margin_pct= self._margin_var.get(),
            out_dir      = self._out_dir,
            max_side     = self._max_side_var.get(),
            color_preset   = self._color_preset.get(),
            studio_matting = self._studio_matting_var.get(),
        )
        threading.Thread(target=self._worker, kwargs=params, daemon=True).start()

    def _stop(self):
        self._stop_evt.set()
        self._log_w("\n⏹  Kullanıcı tarafından durduruldu.\n", "warn")
        self._btn_stop.config(state=tk.DISABLED)

    # ── WORKER ─────────────────────────────────
    def _worker(self, files, mode, threshold, softness, despill,
                model, alpha, trim, margin_pct, out_dir, max_side,
                color_preset, studio_matting):
        total = len(files)
        margin_note = (
            f"  Kırpma : {'açık' if trim else 'kapalı'}  (kenar payı %{margin_pct})\n"
        )
        side_note = (
            f"  Boyut  : uzun kenar ≤ {max_side} px  |  PNG zlib=9\n"
            if max_side > 0
            else "  Boyut  : orijinal çözünürlük  |  PNG zlib=9\n"
        )
        def _hdr():
            lines = [f"\n{'─'*56}\n", f"  Mod    : {mode.upper()}\n"]
            if mode in ("dark", "light"):
                lines.append(
                    f"  Eşik   : {threshold}  Yumuşaklık: {softness}\n"
                )
            lines.append(margin_note)
            lines.append(side_note)
            if out_dir is not None:
                lines.append(f"  Çıktı  : {out_dir.resolve()}\n")
            else:
                lines.append("  Çıktı  : her dosyanın kaynak klasörü\n")
            lines.append(
                f"  Renk   : {COLOR_PRESET_LABELS.get(color_preset, color_preset)}\n"
            )
            if mode == "ai_studio":
                lines.append(
                    "  Matting: "
                    + (
                        "açık (yavaş)\n"
                        if studio_matting
                        else "kapalı — hızlı (önerilen)\n"
                    )
                )
            elif mode == "ai":
                lines.append(
                    "  Matting: "
                    + ("açık (yavaş)\n" if alpha else "kapalı — hızlı\n")
                )
            lines.extend([f"  Toplam : {total} dosya\n", f"{'─'*56}\n\n"])
            self._log_w("".join(lines), "title")

        self.after(0, _hdr)

        self.after(0, lambda: self._bar.config(maximum=total, value=0))

        # AI session (only if needed)
        session = None
        ai_model = model
        if mode == "ai_studio":
            ai_model = "isnet-general-use"
        if mode in ("ai", "ai_studio"):
            self.after(0, lambda: self._sb.config(text="Model yükleniyor…"))
            self.after(0, lambda: self._log_w("⏳ AI modeli yükleniyor…\n", "warn"))
            try:
                from rembg import new_session
                session = new_session(ai_model)
                self.after(0, lambda m=ai_model:
                           self._log_w(f"✔  Model hazır: {m}\n\n", "ok"))
            except Exception as e:
                self.after(0, lambda err=str(e):
                           self._log_w(f"❌ Model yüklenemedi: {err}\n", "err"))
                self._finish()
                return

        t0 = time.time()
        stem_counts = Counter(f.stem for f in files)
        stem_idx = Counter()

        for idx, fp in enumerate(files, 1):
            if self._stop_evt.is_set():
                break

            info = f"[{idx}/{total}]  {fp.name}"
            self.after(0, lambda s=info: self._sb.config(text=s))
            self.after(0, lambda s=info: self._log_w(f"{s}\n", "info"))

            base = out_dir if out_dir is not None else fp.parent
            stem_idx[fp.stem] += 1
            if stem_counts[fp.stem] > 1:
                fname = f"{fp.stem}_{stem_idx[fp.stem]}_rmbg.png"
            else:
                fname = fp.stem + "_rmbg.png"
            out = base / fname
            if out.exists():
                self.after(0, lambda o=out:
                           self._log_w(f"   ↷ Atlandı (mevcut): {o.name}\n", "warn"))
                self._skip_count += 1
                self.after(0, lambda: self._stat_skip.config(
                    text=str(self._skip_count)))
                self.after(0, lambda i=idx: self._bar.config(value=i))
                continue

            try:
                img = load_image_exif_safe(fp)
                orig = img.size

                if mode == "dark":
                    result = remove_dark_bg(img, threshold, softness, despill)
                elif mode == "light":
                    result = remove_light_bg(img, threshold, softness, despill)
                elif mode == "ai_studio":
                    result = remove_ai_bg(
                        img, session, studio_matting,
                        fg_thresh=240, bg_thresh=10, erode=10,
                    )
                else:
                    result = remove_ai_bg(
                        img, session, alpha,
                        fg_thresh=240, bg_thresh=10, erode=10,
                    )

                # Ensure resolution untouched (kırpmadan önce)
                if result.size != orig:
                    result = result.resize(orig, Image.LANCZOS)

                if trim:
                    mr = max(0.02, min(0.15, margin_pct / 100.0))
                    result = trim_transparent_rgba(
                        result, margin_ratio=mr, min_margin=14, alpha_thresh=8
                    )

                result = apply_color_preset(result, color_preset)

                save_rgba_png_compressed(
                    result, out, max_side=max_side, compress_level=9,
                )
                sz = human_size(out.stat().st_size)
                self.after(0, lambda o=out, s=sz:
                           self._log_w(f"   ✔ Kaydedildi → {o.name}  ({s})\n", "ok"))
                self._done_count += 1
                self.after(0, lambda: self._stat_done.config(
                    text=str(self._done_count)))

            except Exception as e:
                self.after(0, lambda err=str(e):
                           self._log_w(f"   ❌ Hata: {err}\n", "err"))
                self._error_count += 1
                self.after(0, lambda: self._stat_error.config(
                    text=str(self._error_count)))

            self.after(0, lambda i=idx: self._bar.config(value=i))
            self.after(0, lambda i=idx:
                       self._prog_lbl.config(text=f"{i}/{total}  tamamlandı"))

        elapsed = time.time() - t0
        self.after(0, lambda: self._log_w(
            f"\n{'─'*56}\n"
            f"  ✅  {self._done_count} tamamlandı  "
            f"❌ {self._error_count} hata  "
            f"⏭ {self._skip_count} atlandı\n"
            f"  ⏱   {elapsed:.1f} sn\n"
            f"{'─'*56}\n\n", "title"))
        self._finish()

    def _finish(self):
        self._running = False
        self.after(0, self._on_finish_ui)

    def _on_finish_ui(self):
        self._btn_run.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.DISABLED)
        self._status_badge.config(text="● Hazır", fg=SUCCESS)
        self._sb.config(text=f"Tamamlandı — {self._done_count} dosya işlendi.")
        self._prog_lbl.config(text="")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = RemBGApp()
    app.mainloop()
