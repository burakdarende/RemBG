"""
RemBG Web — Railway Deploy Sürümü
Hafif Flask uygulaması: in-memory işlem, disk yok, lazy model yükleme.
"""
import os
import io
import gc
import time
import logging
import numpy as np
from pathlib import Path
from functools import lru_cache

from flask import (
    Flask, request, jsonify, send_file,
    render_template_string, abort
)
from PIL import Image, ImageEnhance, ImageOps
from werkzeug.utils import secure_filename

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
MAX_FILE_MB   = int(os.environ.get("MAX_FILE_MB", "10"))
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES + 512  # small overhead


# ── Lazy AI session cache ───────────────────────────────────────────────────
_session_cache: dict = {}

def get_session(model_name: str):
    """Model sadece ilk istek geldiğinde yüklenir (lazy loading)."""
    if model_name not in _session_cache:
        log.info(f"Model yükleniyor: {model_name}")
        from rembg import new_session
        _session_cache[model_name] = new_session(model_name)
        log.info(f"Model hazır: {model_name}")
    return _session_cache[model_name]


# ── Background removal helpers ─────────────────────────────────────────────

def remove_dark_bg(img: Image.Image, threshold: int, softness: int, despill: bool) -> Image.Image:
    rgba = img.convert("RGBA")
    data = np.array(rgba, dtype=np.float32)
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    soft = max(softness, 1)
    alpha = np.clip((lum - threshold) / soft * 255.0, 0.0, 255.0)
    data[:, :, 3] = alpha
    if despill:
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


def remove_light_bg(img: Image.Image, threshold: int, softness: int, despill: bool) -> Image.Image:
    rgba = img.convert("RGBA")
    data = np.array(rgba, dtype=np.float32)
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    soft = max(softness, 1)
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


def remove_ai_bg(img: Image.Image, model: str, alpha_matting: bool) -> Image.Image:
    from rembg import remove
    session = get_session(model)
    result = remove(
        img,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
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


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": time.time()})


@app.route("/process", methods=["POST"])
def process():
    # ── Validate file ───────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "Boş dosya adı"}), 400

    ext = Path(secure_filename(f.filename)).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Desteklenmeyen format: {ext}"}), 415

    # ── Read params ─────────────────────────────────────────────────────
    mode       = request.form.get("mode", "dark")
    threshold  = int(request.form.get("threshold", 35))
    softness   = int(request.form.get("softness", 25))
    despill    = request.form.get("despill", "true").lower() == "true"
    model      = request.form.get("model", "silueta")
    do_alpha   = request.form.get("alpha", "false").lower() == "true"
    studio_matting = request.form.get("studio_matting", "false").lower() == "true"
    do_trim    = request.form.get("trim", "false").lower() == "true"
    margin_pct = int(request.form.get("margin_pct", "4"))
    max_side     = int(request.form.get("max_side", "2048"))
    color_preset = request.form.get("color_preset", "off")

    # Clamp values
    threshold = max(0, min(threshold, 255))
    softness  = max(1, min(softness, 200))
    margin_pct = max(2, min(margin_pct, 12))
    max_side = max(0, min(max_side, 8192))
    if color_preset not in ("off", "auto", "vivid", "soft"):
        color_preset = "off"

    # ── Load image in-memory ────────────────────────────────────────────
    try:
        raw_bytes = f.read()
        if len(raw_bytes) > MAX_FILE_BYTES:
            return jsonify({"error": f"Dosya çok büyük (max {MAX_FILE_MB} MB)"}), 413

        img = Image.open(io.BytesIO(raw_bytes))
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        orig_size = img.size
        log.info(f"İşleniyor: mode={mode} model={model} size={orig_size}")
    except Exception as e:
        log.warning(f"Resim açılamadı: {e}")
        return jsonify({"error": "Resim açılamadı"}), 400
    finally:
        del raw_bytes  # free early

    # ── Process ─────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        if mode == "dark":
            result = remove_dark_bg(img, threshold, softness, despill)
        elif mode == "light":
            result = remove_light_bg(img, threshold, softness, despill)
        elif mode == "ai":
            result = remove_ai_bg(img, model, do_alpha)
        elif mode == "ai_studio":
            result = remove_ai_bg(img, "isnet-general-use", studio_matting)
        else:
            return jsonify({"error": "Geçersiz mod"}), 400
    except Exception as e:
        log.error(f"İşlem hatası: {e}")
        return jsonify({"error": f"İşlem başarısız: {str(e)}"}), 500
    finally:
        del img
        gc.collect()  # lightweight GC nudge

    elapsed = time.time() - t0
    log.info(f"Tamamlandı: {elapsed:.2f}s")

    # ── Ensure original resolution (kırpmadan önce) ──────────────────────
    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)

    if do_trim:
        mr = max(0.02, min(0.15, margin_pct / 100.0))
        result = trim_transparent_rgba(
            result, margin_ratio=mr, min_margin=14, alpha_thresh=8
        )

    result = apply_color_preset(result, color_preset)
    result = downscale_longest_side(result.convert("RGBA"), max_side)

    # ── Encode to PNG in memory (zlib 9) ────────────────────────────────
    buf = io.BytesIO()
    result.save(buf, format="PNG", optimize=True, compress_level=9)
    buf.seek(0)
    del result
    gc.collect()

    stem = Path(secure_filename(f.filename)).stem
    download_name = f"{stem}_rmbg.png"

    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name=download_name,
    )


# ── Embedded HTML (no static folder needed) ────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RemBG Pro — AI Arkaplan Silici</title>
<meta name="description" content="Yapay zeka destekli arkaplan silme aracı. Koyu, açık veya AI segmentasyon modlarıyla PNG olarak indir.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #0d0d14;
    --panel:    #12121e;
    --card:     #1a1a2e;
    --card2:    #16213e;
    --accent:   #7c3aed;
    --accent2:  #a855f7;
    --glow:     #c084fc;
    --success:  #22c55e;
    --warning:  #f59e0b;
    --error:    #ef4444;
    --cyan:     #06b6d4;
    --text:     #f1f0ff;
    --dim:      #9ca3af;
    --muted:    #6b7280;
    --border:   #2d2b55;
    --radius:   12px;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Header ── */
  header {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 0 24px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
  }
  header .logo {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--glow);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  header .logo span { font-size: .85rem; color: var(--dim); font-weight: 400; }
  .status-dot {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: .8rem;
    color: var(--success);
  }
  .status-dot::before {
    content: "";
    width: 8px; height: 8px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 6px currentColor;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── Main layout ── */
  main {
    flex: 1;
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 16px;
    padding: 20px;
    max-width: 1200px;
    width: 100%;
    margin: 0 auto;
  }

  /* ── Card ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
  }
  .section-title {
    font-size: .75rem;
    font-weight: 600;
    color: var(--glow);
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  /* ── Upload zone ── */
  #drop-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color .2s, background .2s;
    position: relative;
    overflow: hidden;
  }
  #drop-zone.drag-over, #drop-zone:hover {
    border-color: var(--accent2);
    background: rgba(124,58,237,.06);
  }
  #drop-zone .icon { font-size: 2.5rem; margin-bottom: 10px; }
  #drop-zone p { color: var(--dim); font-size: .85rem; line-height: 1.6; }
  #drop-zone strong { color: var(--glow); }
  #file-input { display: none; }

  /* ── Preview area ── */
  #preview-wrap {
    display: none;
    gap: 12px;
    margin-top: 12px;
  }
  #preview-wrap.visible { display: flex; align-items: flex-start; gap: 12px; }
  .preview-box {
    flex: 1;
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    position: relative;
  }
  .preview-box label {
    display: block;
    font-size: .7rem;
    color: var(--muted);
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
  }
  .preview-box img {
    width: 100%;
    height: 180px;
    object-fit: contain;
    display: block;
    background: repeating-conic-gradient(#1e1e2e 0% 25%, transparent 0% 50%) 0 0 / 14px 14px;
  }

  /* ── Mode selector ── */
  .mode-grid {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .mode-option {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    border: 1px solid transparent;
    transition: background .15s, border-color .15s;
  }
  .mode-option:hover { background: rgba(255,255,255,.04); }
  .mode-option input[type="radio"] { accent-color: var(--accent2); margin-top: 3px; flex-shrink: 0; }
  .mode-option.selected { background: rgba(124,58,237,.12); border-color: var(--accent); }
  .mode-option .mode-title { font-size: .85rem; font-weight: 500; }
  .mode-option .mode-desc { font-size: .73rem; color: var(--muted); margin-top: 2px; }

  /* ── Sliders ── */
  .param-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
  }
  .param-row label { font-size: .8rem; color: var(--dim); }
  .param-row .val { font-size: .8rem; color: var(--glow); font-weight: 600; min-width: 28px; text-align: right; }
  input[type="range"] {
    width: 100%;
    accent-color: var(--accent2);
    margin-bottom: 10px;
  }
  .checkbox-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: .82rem;
    color: var(--dim);
    cursor: pointer;
    margin-top: 4px;
  }
  .checkbox-row input { accent-color: var(--accent2); width: 14px; height: 14px; }

  /* ── AI model selector ── */
  #ai-panel { display: none; }
  .model-option {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border-radius: 8px;
    cursor: pointer;
    font-size: .82rem;
    color: var(--dim);
    transition: background .15s;
  }
  .model-option:hover { background: rgba(255,255,255,.04); }
  .model-option input { accent-color: var(--accent2); }
  .model-option.selected { color: var(--text); }

  /* ── Process button ── */
  #btn-process {
    width: 100%;
    padding: 12px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: var(--radius);
    font-size: .95rem;
    font-weight: 600;
    cursor: pointer;
    margin-top: 14px;
    transition: background .2s, transform .1s, box-shadow .2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    letter-spacing: .02em;
  }
  #btn-process:hover:not(:disabled) {
    background: #6d28d9;
    box-shadow: 0 0 20px rgba(124,58,237,.4);
    transform: translateY(-1px);
  }
  #btn-process:disabled { opacity: .5; cursor: not-allowed; transform: none; }

  /* ── Progress / status ── */
  #progress-wrap { display: none; margin-top: 12px; }
  #progress-wrap.visible { display: block; }
  .progress-bar-track {
    height: 6px;
    background: var(--card2);
    border-radius: 99px;
    overflow: hidden;
    margin-top: 8px;
  }
  .progress-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), var(--glow));
    transition: width .3s ease;
    width: 0%;
  }
  .progress-bar-fill.indeterminate {
    width: 40%;
    animation: slide 1.2s infinite ease-in-out;
  }
  @keyframes slide {
    0% { margin-left: -40%; }
    100% { margin-left: 140%; }
  }
  #status-text { font-size: .82rem; color: var(--dim); }

  /* ── Result panel ── */
  #result-panel { display: none; }
  #result-panel.visible { display: block; }
  #result-img {
    width: 100%;
    max-height: 420px;
    object-fit: contain;
    border-radius: 8px;
    background: repeating-conic-gradient(#1e1e2e 0% 25%, transparent 0% 50%) 0 0 / 14px 14px;
    display: block;
    margin-bottom: 12px;
  }
  #btn-download {
    width: 100%;
    padding: 10px;
    background: var(--success);
    color: #000;
    border: none;
    border-radius: var(--radius);
    font-size: .9rem;
    font-weight: 600;
    cursor: pointer;
    transition: background .2s, transform .1s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  #btn-download:hover { background: #16a34a; transform: translateY(-1px); }
  .result-meta { font-size: .75rem; color: var(--muted); margin-top: 8px; text-align: center; }

  /* ── Error toast ── */
  #error-box {
    display: none;
    margin-top: 10px;
    padding: 10px 14px;
    background: rgba(239,68,68,.15);
    border: 1px solid var(--error);
    border-radius: 8px;
    font-size: .82rem;
    color: var(--error);
  }
  #error-box.visible { display: block; }

  /* ── Right placeholder ── */
  #right-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    min-height: 300px;
    color: var(--muted);
    gap: 12px;
    font-size: .9rem;
  }
  #right-placeholder .big-icon { font-size: 4rem; opacity: .3; }

  /* ── Footer ── */
  footer {
    text-align: center;
    padding: 16px;
    font-size: .75rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
  }

  /* ── Responsive ── */
  @media (max-width: 720px) {
    main { grid-template-columns: 1fr; }
    #result-img { max-height: 260px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">✦ RemBG Pro <span>· AI Arkaplan Silici</span></div>
  <div class="status-dot" id="status-badge">Hazır</div>
</header>

<main>
  <!-- LEFT: Controls -->
  <aside>

    <!-- Upload -->
    <div class="card" style="margin-bottom:14px">
      <div class="section-title">📂 Dosya Seç</div>
      <div id="drop-zone">
        <div class="icon">🖼️</div>
        <p>Resmi buraya sürükle & bırak<br>veya <strong>tıklayarak seç</strong></p>
        <p style="margin-top:8px;font-size:.73rem;color:var(--muted)">JPG · PNG · WEBP · BMP · TIFF — max 10 MB</p>
      </div>
      <input type="file" id="file-input" accept=".jpg,.jpeg,.png,.webp,.bmp,.tiff,.tif">

      <div id="preview-wrap">
        <div class="preview-box">
          <label>Orijinal</label>
          <img id="preview-orig" alt="Orijinal">
        </div>
      </div>
    </div>

    <!-- Mode -->
    <div class="card" style="margin-bottom:14px">
      <div class="section-title">⚙️ İşlem Modu</div>
      <div class="mode-grid">
        <label class="mode-option selected">
          <input type="radio" name="mode" value="dark" checked>
          <div>
            <div class="mode-title">🌑 Koyu Arkaplan Sil</div>
            <div class="mode-desc">Siyah / koyu bg — Neon, tel kafes, çizgi sanatı</div>
          </div>
        </label>
        <label class="mode-option">
          <input type="radio" name="mode" value="light">
          <div>
            <div class="mode-title">☀️ Açık Arkaplan Sil</div>
            <div class="mode-desc">Beyaz / açık bg — Logo, tarama, flat illüstrasyon</div>
          </div>
        </label>
        <label class="mode-option">
          <input type="radio" name="mode" value="ai_studio">
          <div>
            <div class="mode-title">🎯 AI Nesne (stüdyo / ürün)</div>
            <div class="mode-desc">ISNet — hızlı varsayılan; matting isteğe bağlı</div>
          </div>
        </label>
        <label class="mode-option">
          <input type="radio" name="mode" value="ai">
          <div>
            <div class="mode-title">🤖 AI Segmentasyon</div>
            <div class="mode-desc">Fotoğraf, insan, nesne — Model seç</div>
          </div>
        </label>
      </div>
    </div>

    <!-- Lum params -->
    <div class="card" id="lum-panel" style="margin-bottom:14px">
      <div class="section-title">🎛️ Lüminan Ayarları</div>

      <div class="param-row">
        <label>Eşik (Threshold)</label>
        <span class="val" id="thresh-val">35</span>
      </div>
      <input type="range" id="threshold" min="0" max="200" value="35">

      <div class="param-row">
        <label>Yumuşaklık (Softness)</label>
        <span class="val" id="soft-val">25</span>
      </div>
      <input type="range" id="softness" min="1" max="120" value="25">

      <label class="checkbox-row">
        <input type="checkbox" id="despill" checked>
        Kenar rengi düzeltme (despill)
      </label>
    </div>

    <div class="card" id="studio-panel" style="margin-bottom:14px;display:none">
      <div class="section-title">🎯 Stüdyo modu</div>
      <p style="font-size:.82rem;color:var(--muted);line-height:1.55">
        Varsayılan hızlı: sadece ISNet (CPU). İnce saçak istiyorsan matting aç — yavaşlar.
      </p>
      <label class="checkbox-row" style="margin-top:8px">
        <input type="checkbox" id="studio-matting">
        Alpha matting (yumuşak kenar, çok yavaş — pymatting)
      </label>
    </div>

    <!-- AI params -->
    <div class="card" id="ai-panel" style="margin-bottom:14px">
      <div class="section-title">🤖 AI Model</div>
      <label class="model-option selected"><input type="radio" name="model" value="silueta" checked> Silueta — Hafif &amp; Hızlı (Önerilen)</label>
      <label class="model-option"><input type="radio" name="model" value="isnet-general-use"> ISNet — Yüksek Detay</label>
      <label class="model-option"><input type="radio" name="model" value="u2net"> U2Net — Genel Amaç</label>
      <label class="model-option"><input type="radio" name="model" value="u2net_human_seg"> U2Net Human — İnsan / Portre</label>
      <label class="checkbox-row" style="margin-top:10px">
        <input type="checkbox" id="alpha-matting">
        Alpha matting (yumuşak kenar, yavaşlatır)
      </label>
    </div>

    <div class="card" style="margin-bottom:14px">
      <div class="section-title">📐 Çıktı</div>
      <label class="checkbox-row">
        <input type="checkbox" id="trim-crop" checked>
        Fazla boşluğu kırp (şeffaf alanları at, nesneyi ortala)
      </label>
      <div class="param-row" style="margin-top:10px">
        <label>Kenar payı (%)</label>
        <span class="val" id="margin-val">4</span>
      </div>
      <input type="range" id="margin-pct" min="2" max="12" value="4">
      <div class="param-row" style="margin-top:10px">
        <label>Max uzun kenar (px)</label>
        <span class="val" id="max-side-val">2048</span>
      </div>
      <input type="range" id="max-side" min="0" max="4096" step="64" value="2048">
      <p style="font-size:.73rem;color:var(--muted);margin-top:6px;line-height:1.45">
        0 = boyut aynı, yalnızca PNG sıkıştırma · 2048 ≈ daha küçük dosya (~2–5 MB)
      </p>
      <div class="section-title" style="margin-top:12px">Renk / ışık</div>
      <div class="mode-grid" style="gap:2px">
        <label class="mode-option" style="padding:6px 10px">
          <input type="radio" name="color_preset" value="off" checked>
          <div><div class="mode-title">Kapalı</div><div class="mode-desc">Ham renk</div></div>
        </label>
        <label class="mode-option" style="padding:6px 10px">
          <input type="radio" name="color_preset" value="auto">
          <div><div class="mode-title">Otomatik</div><div class="mode-desc">Histogram + hafif canlılık</div></div>
        </label>
        <label class="mode-option" style="padding:6px 10px">
          <input type="radio" name="color_preset" value="vivid">
          <div><div class="mode-title">Canlı</div><div class="mode-desc">Doygunluk + kontrast</div></div>
        </label>
        <label class="mode-option" style="padding:6px 10px">
          <input type="radio" name="color_preset" value="soft">
          <div><div class="mode-title">Yumuşak</div><div class="mode-desc">Solukları nazikçe aç</div></div>
        </label>
      </div>
    </div>

    <!-- Process button -->
    <button id="btn-process" disabled>
      <span>▶</span> İşlemi Başlat
    </button>

    <div id="progress-wrap">
      <span id="status-text">İşleniyor…</span>
      <div class="progress-bar-track">
        <div class="progress-bar-fill indeterminate" id="pbar"></div>
      </div>
    </div>

    <div id="error-box"></div>

  </aside>

  <!-- RIGHT: Result -->
  <section>
    <div id="right-placeholder">
      <div class="big-icon">✨</div>
      <p>Bir resim yükle ve işlemi başlat</p>
    </div>

    <div id="result-panel" class="card">
      <div class="section-title">✅ Sonuç</div>
      <img id="result-img" alt="Sonuç">
      <a id="btn-download" href="#" download>
        ⬇ PNG Olarak İndir
      </a>
      <div class="result-meta" id="result-meta"></div>
    </div>
  </section>
</main>

<footer>RemBG Pro · Railway Edition · In-memory processing — hiçbir dosya sunucuda saklanmaz</footer>

<script>
const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const btnProcess  = document.getElementById('btn-process');
const progressWrap = document.getElementById('progress-wrap');
const statusText  = document.getElementById('status-text');
const pbar        = document.getElementById('pbar');
const resultPanel = document.getElementById('result-panel');
const placeholder = document.getElementById('right-placeholder');
const resultImg   = document.getElementById('result-img');
const btnDownload = document.getElementById('btn-download');
const resultMeta  = document.getElementById('result-meta');
const errorBox    = document.getElementById('error-box');
const previewWrap = document.getElementById('preview-wrap');
const previewOrig = document.getElementById('preview-orig');
const statusBadge = document.getElementById('status-badge');
const lumPanel    = document.getElementById('lum-panel');
const aiPanel     = document.getElementById('ai-panel');
const studioPanel = document.getElementById('studio-panel');

let selectedFile = null;

// ── Drop zone ────────────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(f) {
  const allowed = ['image/jpeg','image/png','image/webp','image/bmp','image/tiff'];
  if (!allowed.includes(f.type) && !f.name.match(/\.(jpg|jpeg|png|webp|bmp|tiff|tif)$/i)) {
    showError('Desteklenmeyen dosya formatı.'); return;
  }
  if (f.size > 10 * 1024 * 1024) {
    showError('Dosya 10 MB\'dan büyük olamaz.'); return;
  }
  selectedFile = f;
  const url = URL.createObjectURL(f);
  previewOrig.src = url;
  previewWrap.classList.add('visible');
  btnProcess.disabled = false;
  hideError();
  resultPanel.classList.remove('visible');
  placeholder.style.display = 'none';
}

// ── Mode switch ──────────────────────────────────────────────────────────
document.querySelectorAll('input[name="mode"]').forEach(rb => {
  rb.addEventListener('change', () => {
    document.querySelectorAll('.mode-option').forEach(el => el.classList.remove('selected'));
    rb.closest('.mode-option').classList.add('selected');
    if (rb.value === 'ai') {
      lumPanel.style.display = 'none';
      aiPanel.style.display = 'block';
      studioPanel.style.display = 'none';
    } else if (rb.value === 'ai_studio') {
      lumPanel.style.display = 'none';
      aiPanel.style.display = 'none';
      studioPanel.style.display = 'block';
    } else {
      lumPanel.style.display = 'block';
      aiPanel.style.display = 'none';
      studioPanel.style.display = 'none';
    }
  });
});

document.querySelectorAll('input[name="model"]').forEach(rb => {
  rb.addEventListener('change', () => {
    document.querySelectorAll('.model-option').forEach(el => el.classList.remove('selected'));
    rb.closest('.model-option').classList.add('selected');
  });
});

// ── Sliders ──────────────────────────────────────────────────────────────
document.getElementById('threshold').addEventListener('input', function() {
  document.getElementById('thresh-val').textContent = this.value;
});
document.getElementById('softness').addEventListener('input', function() {
  document.getElementById('soft-val').textContent = this.value;
});
document.getElementById('margin-pct').addEventListener('input', function() {
  document.getElementById('margin-val').textContent = this.value;
});
function updateMaxSideLabel(v) {
  const n = parseInt(v, 10);
  document.getElementById('max-side-val').textContent = n === 0 ? '0 (tam)' : String(n);
}
document.getElementById('max-side').addEventListener('input', function() {
  updateMaxSideLabel(this.value);
});

// ── Process ──────────────────────────────────────────────────────────────
btnProcess.addEventListener('click', async () => {
  if (!selectedFile) return;

  const mode      = document.querySelector('input[name="mode"]:checked').value;
  const threshold = document.getElementById('threshold').value;
  const softness  = document.getElementById('softness').value;
  const despill   = document.getElementById('despill').checked;
  const model     = document.querySelector('input[name="model"]:checked').value;
  const alpha     = document.getElementById('alpha-matting').checked;

  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('mode', mode);
  fd.append('threshold', threshold);
  fd.append('softness', softness);
  fd.append('despill', despill);
  fd.append('model', model);
  fd.append('alpha', alpha);
  fd.append('trim', document.getElementById('trim-crop').checked);
  fd.append('margin_pct', document.getElementById('margin-pct').value);
  fd.append('max_side', document.getElementById('max-side').value);
  fd.append('color_preset', document.querySelector('input[name="color_preset"]:checked').value);
  fd.append('studio_matting', document.getElementById('studio-matting').checked);

  // UI: loading
  btnProcess.disabled = true;
  progressWrap.classList.add('visible');
  pbar.classList.add('indeterminate');
  statusText.textContent = (mode === 'ai' || mode === 'ai_studio')
    ? 'AI modeli yükleniyor (ilk çalıştırmada ~30 sn)…' : 'İşleniyor…';
  resultPanel.classList.remove('visible');
  hideError();
  setStatus('İşleniyor…', 'warning');

  const t0 = Date.now();
  try {
    const resp = await fetch('/process', { method: 'POST', body: fd });

    if (!resp.ok) {
      const json = await resp.json().catch(() => ({ error: 'Bilinmeyen hata' }));
      throw new Error(json.error || `HTTP ${resp.status}`);
    }

    const blob = await resp.blob();
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const url  = URL.createObjectURL(blob);
    const name = selectedFile.name.replace(/\.[^.]+$/, '') + '_rmbg.png';

    resultImg.src = url;
    btnDownload.href = url;
    btnDownload.download = name;
    resultMeta.textContent = `${(blob.size / 1024).toFixed(0)} KB · ${elapsed} sn`;

    resultPanel.classList.add('visible');
    placeholder.style.display = 'none';
    setStatus('Hazır', 'success');
  } catch(err) {
    showError(err.message);
    setStatus('Hata', 'error');
  } finally {
    btnProcess.disabled = false;
    progressWrap.classList.remove('visible');
    pbar.classList.remove('indeterminate');
  }
});

function showError(msg) {
  errorBox.textContent = '❌ ' + msg;
  errorBox.classList.add('visible');
}
function hideError() { errorBox.classList.remove('visible'); }

function setStatus(text, type) {
  const colors = { success: '#22c55e', warning: '#f59e0b', error: '#ef4444' };
  statusBadge.textContent = text;
  statusBadge.style.color = colors[type] || '#22c55e';
}
</script>
</body>
</html>
"""


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    log.info(f"RemBG Web başlatılıyor — port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
