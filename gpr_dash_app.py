

"""
AI-Engine for Buried Object Detection — Plotly Dash Edition
AVNL-OFMK

Converted from the supplied Streamlit application while retaining its
SEG-Y preprocessing, Roboflow inference, tiling/NMS, annotation and CSV
export logic.

Run:
    pip install -r requirements_dash.txt
    python gpr_dash_app.py

Environment:
    ROBOFLOW_API_KEY=...
"""

import base64

POSTIMAGES_GPR_REFERENCE = "https://postimg.cc/8F64q25w"

# Official institutional branding assets.
# Original artwork is fetched from the official/public source once and then
# served through same-origin Flask routes so it renders reliably in the
# authenticated application.
AVNL_OFFICIAL_LOGO_URL = "https://avnl.co.in/files/inline-images/AVNL%20-%20Logo%20final-01%20%281%29.png"
AVNL_LOGO_URL = "/brand-logo/avnl"
CMERI_LOGO_URL = "/assets/cmeri_cmeri_logo.png"
import io
import json
import os
import csv
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, session, redirect, request, send_file, abort
from werkzeug.security import check_password_hash, generate_password_hash

from dash import Dash, dcc, html, dash_table, Input, Output, State, ALL, ctx
from dash.exceptions import PreventUpdate

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt

# ── Application-wide logger ────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("GPR_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gpr_detection_app")

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING BACKEND (ported from Process_sgy_jpeg.py v4)
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.style.use("dark_background")


def pp_read_sgy(file_bytes: bytes, filename: str) -> tuple:
    """Read a SEG-Y file from bytes. Returns (data, dt_ns, meta_str)."""
    import tempfile
    import obspy
    with tempfile.NamedTemporaryFile(suffix=".sgy", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    st_obj = obspy.read(tmp_path)
    os.unlink(tmp_path)
    data = np.array([tr.data for tr in st_obj], dtype=np.float32).T
    dt_ns = float(st_obj[0].stats.delta) * 1e9
    n_samples, n_traces = data.shape
    meta = (f"File: {filename}  |  Shape: {n_samples}×{n_traces}  |  "
            f"dt: {dt_ns:.4f} ns  |  Fs: {1e3/dt_ns:.1f} MHz  |  "
            f"Range: [{data.min():.4g}, {data.max():.4g}]")
    return data, dt_ns, meta


def pp_dewow(data: np.ndarray, window: int = 10) -> np.ndarray:
    trend = uniform_filter1d(data, size=window, axis=0, mode="reflect")
    return (data - trend).astype(np.float32)


def pp_bandpass(data: np.ndarray, low_MHz: float, high_MHz: float,
                dt_ns: float, order: int = 4) -> np.ndarray:
    fs_MHz = 1e3 / dt_ns
    nyq = fs_MHz / 2.0
    lo = float(np.clip(low_MHz / nyq, 1e-4, 0.9999))
    hi = float(np.clip(high_MHz / nyq, 1e-4, 0.9999))
    if lo >= hi:
        return data
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, data, axis=0).astype(np.float32)


def pp_background_removal(data: np.ndarray, mode: str = "mean") -> np.ndarray:
    bg = np.median(data, axis=1, keepdims=True) if mode == "median" \
        else np.mean(data, axis=1, keepdims=True)
    return (data - bg).astype(np.float32)


def pp_trace_normalise(data: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    rms = np.sqrt(np.mean(data ** 2, axis=0, keepdims=True)) + eps
    return (data / rms).astype(np.float32)


def pp_apply_gain(data: np.ndarray, mode: str = "linear",
                  gain_db: float = 30.0, agc_window: int = 20) -> tuple:
    n = data.shape[0]
    d = np.linspace(0, 1, n, dtype=np.float32)
    mg = 10 ** (gain_db / 20.0)
    if mode == "linear":
        gv = (1 + (mg - 1) * d).reshape(-1, 1)
    elif mode == "quadratic":
        gv = (1 + (mg - 1) * d ** 2).reshape(-1, 1)
    elif mode == "agc":
        pad = np.pad(data, ((agc_window, agc_window), (0, 0)), mode="edge")
        local_rms = np.array([
            np.sqrt(np.mean(pad[i:i + 2 * agc_window + 1] ** 2, axis=0))
            for i in range(n)
        ], dtype=np.float32) + 1e-9
        return (data / local_rms).astype(np.float32), np.ones(n, dtype=np.float32)
    else:
        raise ValueError(f"Unknown gain mode: {mode!r}")
    return (data * gv).astype(np.float32), gv.ravel()


def pp_normalise_uint8(data: np.ndarray, plo: float = 1.0, phi: float = 99.0) -> np.ndarray:
    img = np.nan_to_num(data.copy())
    lo = float(np.percentile(img, plo))
    hi = float(np.percentile(img, phi))
    span = hi - lo
    if span < 1e-12:
        return np.zeros_like(img, dtype=np.uint8)
    return np.clip((img - lo) / span * 255, 0, 255).astype(np.uint8)


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()


def pp_plot_bscan_plain(data: np.ndarray,
                        target_size: Optional[Tuple[int, int]] = None,
                        plo: float = 1.0, phi: float = 99.0) -> Image.Image:
    img_u8 = pp_normalise_uint8(data, plo, phi)
    pil = Image.fromarray(img_u8, mode="L").convert("RGB")
    if target_size is not None:
        pil = pil.resize(target_size, Image.Resampling.LANCZOS)
    return pil


def pp_run_pipeline(file_bytes: bytes, filename: str, cfg: dict) -> dict:
    """Run the full preprocessing pipeline on one SGY file."""
    result: dict = {"status": "OK", "log": [], "filename": filename}

    def _log(msg: str):
        result["log"].append(msg)

    try:
        _log(f"📖 Reading {filename}…")
        raw, dt_ns, meta = pp_read_sgy(file_bytes, filename)
        result["raw"] = raw
        result["dt_ns"] = dt_ns
        result["meta"] = meta
        result["n_samples"], result["n_traces"] = raw.shape
        _log(f"   {meta}")

        data = raw.copy()
        if cfg["apply_dewow"]:
            _log(f"🔧 Dewow (window={cfg['dewow_window']})…")
            data = pp_dewow(data, window=cfg["dewow_window"])
        result["after_dewow"] = data.copy()

        if cfg["apply_bandpass"]:
            _log(f"🔧 Bandpass {cfg['bp_low_MHz']:.0f}–{cfg['bp_high_MHz']:.0f} MHz…")
            data = pp_bandpass(data, cfg["bp_low_MHz"], cfg["bp_high_MHz"],
                               dt_ns, cfg["bp_order"])
        result["after_bandpass"] = data.copy()

        _log(f"🔧 Background removal (mode={cfg['bg_mode']})…")
        bg = pp_background_removal(data, mode=cfg["bg_mode"])
        result["after_bg"] = bg.copy()

        if cfg["trace_normalise"]:
            _log("🔧 Trace normalisation…")
            bg = pp_trace_normalise(bg)

        _log(f"🔧 Gain mode={cfg['gain_mode']} db={cfg['gain_db']:.1f}…")
        gained, gv = pp_apply_gain(bg, mode=cfg["gain_mode"],
                                   gain_db=cfg["gain_db"],
                                   agc_window=cfg["agc_window"])
        result["gained"] = gained
        result["gv"] = gv

        _log("🖼 Building 640×640 JPEG…")
        _ref_lo = float(np.percentile(bg, 1.0))
        _ref_hi = float(np.percentile(bg, 99.0))
        _ref_span = max(_ref_hi - _ref_lo, 1e-9)
        img_u8 = np.clip((gained - _ref_lo) / _ref_span * 255, 0, 255).astype(np.uint8)
        pil_out = Image.fromarray(img_u8, mode="L")
        result["output_pil_full"] = pil_out.convert("RGB")
        w, h = cfg["resize_shape"]
        pil_sq = pil_out.resize((w, h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil_sq.save(buf, format="JPEG", quality=cfg["jpeg_quality"])
        result["output_jpeg_bytes"] = buf.getvalue()
        result["output_pil"] = pil_sq

        _log("✅ Pipeline complete.")

    except Exception as exc:
        result["status"] = "FAILED"
        result["log"].append(f"❌ ERROR: {exc}")
        result["log"].append(traceback.format_exc())

    return result


_PP_DEFAULT_CFG: dict = {
    "gain_mode": "linear",
    "gain_db": 30.0,
    "agc_window": 20,
    "bg_mode": "mean",
    "apply_dewow": True,
    "dewow_window": 39,
    "apply_bandpass": True,
    "bp_low_MHz": 100.0,
    "bp_high_MHz": 900.0,
    "bp_order": 4,
    "trace_normalise": False,
    "resize_shape": (640, 640),
    "jpeg_quality": 95,
    "cmap": "gray",
}

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Roboflow credential loading mirrors the supplied Streamlit app. Priority:
#   1) current Flask session (temporary admin override)
#   2) ROBOFLOW_API_KEY environment variable
#   3) .streamlit/secrets.toml (ROBOFLOW_API_KEY = "...")
#   4) secrets.toml beside this script
#   5) the fallback credential used by the supplied Streamlit application
#
# The fallback is retained here specifically so the Dash build behaves like the
# supplied Streamlit build when neither environment variables nor secrets.toml
# are present. It can be removed for production deployments in favour of an
# environment/secret-only policy.
_FALLBACK_KEY = "reykCRkfdScF0S9rTdJq"
def _read_toml_api_key(path: Path) -> str:
    try:
        if not path.exists():
            return ""
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        value = data.get("ROBOFLOW_API_KEY", "")
        return str(value).strip() if value else ""
    except Exception as exc:
        logger.warning("Unable to read Roboflow secret file %s: %s", path, exc)
        return ""


def _load_api_key() -> str:
    env_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if env_key:
        return env_key

    base = Path(__file__).resolve().parent
    for secret_path in (
        base / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
        base / "secrets.toml",
        Path.cwd() / "secrets.toml",
    ):
        key = _read_toml_api_key(secret_path)
        if key:
            return key
    return _FALLBACK_KEY


ROBOFLOW_API_KEY: str = _load_api_key()


def _get_api_key() -> str:
    """Return the active credential without exposing it to the client."""
    session_key = str(session.get("roboflow_api_key", "")).strip()
    return session_key or ROBOFLOW_API_KEY

MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "rawgprburieobjectdetection/1")
ROBOFLOW_URL = f"https://detect.roboflow.com/{MODEL_ID}"
API_TIMEOUT = 20
API_RETRIES = 1
JPEG_QUALITY = 92

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

TILE_THRESHOLD = 1280
TILE_SIZE = 640
TILE_OVERLAP = 128
IOU_THRESHOLD = 0.45

TARGET_INFER_SIZE = 640
MIN_EDGE_FOR_MULTISCALE = 320
PAD_TO_SQUARE = True

CLASS_META: Dict[str, dict] = {
    "landmine": {"color": "#ff3030", "icon": "💣"},
    "mine": {"color": "#ff5555", "icon": "💣"},
    "ied": {"color": "#ff2020", "icon": "💣"},
    "threat": {"color": "#ff6600", "icon": "⚠️"},
    "metal": {"color": "#ff8c00", "icon": "🔩"},
    "pipe": {"color": "#ffa500", "icon": "🔧"},
    "cable": {"color": "#ffd700", "icon": "⚡"},
    "utility": {"color": "#ffe066", "icon": "🔌"},
    "rock": {"color": "#00ffb4", "icon": "🪨"},
    "root": {"color": "#7ac9a9", "icon": "🌿"},
    "void": {"color": "#00bfff", "icon": "⭕"},
    "clutter": {"color": "#aaaaaa", "icon": "📦"},
}
DEFAULT_META: dict = {"color": "#00ff8c", "icon": "❓"}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_meta(cls_name: str) -> dict:
    low = cls_name.lower()
    for k, v in CLASS_META.items():
        if k in low:
            return v
    return DEFAULT_META


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", image.size, (0, 0, 0))
        bg.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[3])
        return bg
    return image.convert("RGB")


def _encode_jpeg(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL API CALL
# ─────────────────────────────────────────────────────────────────────────────
def _call_api(img_b64: str, confidence: int, overlap: int) -> Dict[str, Any]:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "Roboflow API key not configured. The Dash app accepts the same "
            "ROBOFLOW_API_KEY configuration as the supplied Streamlit app. "
            "Set it in .streamlit/secrets.toml, secrets.toml, or the "
            "ROBOFLOW_API_KEY environment variable."
        )
    last_exc: Optional[Exception] = None
    for attempt in range(1, API_RETRIES + 1):
        try:
            logger.info("Roboflow inference request: model=%s confidence=%s overlap=%s", MODEL_ID, confidence, overlap)
            resp = requests.post(
                ROBOFLOW_URL,
                params={
                    "api_key": api_key,
                    "confidence": confidence,
                    "overlap": overlap,
                },
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=API_TIMEOUT,
            )
            logger.info("Roboflow response: HTTP %s", resp.status_code)
            if not resp.ok:
                detail = resp.text[:1000].strip()
                raise RuntimeError(f"Roboflow HTTP {resp.status_code}: {detail or 'empty response'}")
            try:
                return resp.json()
            except ValueError as exc:
                raise RuntimeError(f"Roboflow returned non-JSON response: {resp.text[:1000]}") from exc
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < API_RETRIES:
                time.sleep(2 ** (attempt - 1))
        except requests.exceptions.HTTPError:
            raise
    raise RuntimeError(f"API unreachable after {API_RETRIES} attempts: {last_exc}")


# ─────────────────────────────────────────────────────────────────────────────
# NMS HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a: dict, b: dict) -> float:
    ax1, ay1 = a["x"] - a["width"] / 2, a["y"] - a["height"] / 2
    ax2, ay2 = a["x"] + a["width"] / 2, a["y"] + a["height"] / 2
    bx1, by1 = b["x"] - b["width"] / 2, b["y"] - b["height"] / 2
    bx2, by2 = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _nms(predictions: List[dict], iou_thresh: float = IOU_THRESHOLD) -> List[dict]:
    if not predictions:
        return []
    by_class: Dict[str, List[dict]] = {}
    for p in predictions:
        by_class.setdefault(p.get("class", ""), []).append(p)
    kept: List[dict] = []
    for cls_preds in by_class.values():
        cls_preds = sorted(cls_preds, key=lambda p: p.get("confidence", 0), reverse=True)
        while cls_preds:
            best = cls_preds.pop(0)
            kept.append(best)
            cls_preds = [p for p in cls_preds if _iou(best, p) < iou_thresh]
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# TILED INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def _infer_tile(tile: Image.Image, offset_x: int, offset_y: int,
                confidence: int, overlap: int) -> List[dict]:
    tw, th = tile.size
    result = _call_api(_encode_jpeg(tile), confidence, overlap)
    preds = result.get("predictions", [])
    out: List[dict] = []
    for p in preds:
        cx = max(p["width"] / 2, min(tw - p["width"] / 2, p["x"]))
        cy = max(p["height"] / 2, min(th - p["height"] / 2, p["y"]))
        new_p = dict(p)
        new_p["x"] = cx + offset_x
        new_p["y"] = cy + offset_y
        out.append(new_p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SMALL-IMAGE ADAPTIVE UPSCALING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _upscale_for_inference(rgb: Image.Image,
                           target: int = TARGET_INFER_SIZE,
                           pad_square: bool = PAD_TO_SQUARE
                           ) -> Tuple[Image.Image, float, int, int]:
    W, H = rgb.size
    longest = max(W, H)
    scale = max(1.0, target / longest)
    new_w = max(1, round(W * scale))
    new_h = max(1, round(H * scale))
    upscaled = rgb.resize((new_w, new_h), Image.LANCZOS)
    pad_left = pad_top = 0

    if pad_square:
        # Reflect padding is more suitable for GPR imagery than black borders
        # because the detector does not see an artificial high-contrast frame.
        side = max(new_w, new_h)
        pad_left = (side - new_w) // 2
        pad_right = side - new_w - pad_left
        pad_top = (side - new_h) // 2
        pad_bottom = side - new_h - pad_top
        arr = np.asarray(upscaled, dtype=np.uint8)
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        mode = "reflect" if min(new_w, new_h) > 1 else "edge"
        padded_arr = np.pad(arr, pad_width, mode=mode)
        padded = Image.fromarray(padded_arr, mode="RGB")
        return padded, scale, pad_left, pad_top

    return upscaled, scale, 0, 0


def _remap_preds_to_original(preds: List[dict],
                             scale: float,
                             pad_left: int,
                             pad_top: int,
                             orig_w: int,
                             orig_h: int) -> List[dict]:
    remapped = []
    for p in preds:
        cx = (p["x"] - pad_left) / scale
        cy = (p["y"] - pad_top) / scale
        bw = p["width"] / scale
        bh = p["height"] / scale
        cx = max(bw / 2, min(orig_w - bw / 2, cx))
        cy = max(bh / 2, min(orig_h - bh / 2, cy))
        new_p = dict(p)
        new_p["x"] = cx
        new_p["y"] = cy
        new_p["width"] = min(bw, orig_w)
        new_p["height"] = min(bh, orig_h)
        remapped.append(new_p)
    return remapped


def _infer_at_scale(rgb: Image.Image,
                    scale_factor: float,
                    confidence: int,
                    overlap: int,
                    pad_square: bool = PAD_TO_SQUARE) -> List[dict]:
    orig_w, orig_h = rgb.size
    if scale_factor <= 1.0:
        prepared, scale, pl, pt = _upscale_for_inference(rgb, TARGET_INFER_SIZE, pad_square)
    else:
        new_w = round(orig_w * scale_factor)
        new_h = round(orig_h * scale_factor)
        upscaled = rgb.resize((new_w, new_h), Image.LANCZOS)
        prepared, extra_scale, pl, pt = _upscale_for_inference(
            upscaled, max(new_w, new_h, TARGET_INFER_SIZE), pad_square)
        scale = scale_factor * extra_scale

    result = _call_api(_encode_jpeg(prepared), confidence, overlap)
    preds = result.get("predictions", [])
    return _remap_preds_to_original(preds, scale, pl, pt, orig_w, orig_h)


def run_inference(image: Image.Image, confidence: int, overlap: int,
                  tile: bool = True, tile_px: int = TILE_SIZE,
                  tile_ov: int = TILE_OVERLAP,
                  multi_scale: bool = True,
                  pad_square: bool = PAD_TO_SQUARE) -> Dict[str, Any]:
    rgb = _to_rgb(image)
    W, H = rgb.size
    longest = max(W, H)

    if longest < TARGET_INFER_SIZE:
        all_preds: List[dict] = []
        all_preds.extend(_infer_at_scale(rgb, 1.0, confidence, overlap, pad_square))
        if multi_scale and longest < MIN_EDGE_FOR_MULTISCALE:
            for extra in (2.0, 3.0):
                try:
                    all_preds.extend(_infer_at_scale(rgb, extra, confidence, overlap, pad_square))
                except Exception as exc:
                    logger.debug("Multi-scale pass ×%.1f skipped: %s", extra, exc)
        merged = _nms(all_preds)
        return {"predictions": merged, "image": {"width": W, "height": H}}

    if not tile or longest <= TILE_THRESHOLD:
        result = _call_api(_encode_jpeg(rgb), confidence, overlap)
        return result

    stride = tile_px - tile_ov
    all_preds = []
    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))

    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_px, W)
            y1 = min(y0 + tile_px, H)
            tx0 = max(0, x1 - tile_px)
            ty0 = max(0, y1 - tile_px)
            tile_img = rgb.crop((tx0, ty0, tx0 + tile_px, ty0 + tile_px))
            tile_preds = _infer_tile(tile_img, tx0, ty0, confidence, overlap)
            all_preds.extend(tile_preds)

    merged = _nms(all_preds)
    return {"predictions": merged, "image": {"width": W, "height": H}}


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def draw_detections(image: Image.Image, predictions: List[dict],
                    min_render_size: int = 512) -> Image.Image:
    img_rgb = image.convert("RGB")
    orig_w, orig_h = img_rgb.size
    longest = max(orig_w, orig_h)

    if longest < min_render_size:
        render_scale = min_render_size / longest
        render_w = max(1, round(orig_w * render_scale))
        render_h = max(1, round(orig_h * render_scale))
        canvas = img_rgb.resize((render_w, render_h), Image.NEAREST)
    else:
        render_scale = 1.0
        render_w, render_h = orig_w, orig_h
        canvas = img_rgb.copy()

    draw = ImageDraw.Draw(canvas, "RGBA")

    scale_factor = max(render_w, render_h) / 640.0
    box_width = max(1, round(2 * scale_factor))
    tick_len = max(4, round(12 * scale_factor))
    tick_width = max(1, round(3 * scale_factor))
    dot_r = max(4, round(10 * scale_factor))
    font_size_b = max(8, round(15 * scale_factor))
    font_size_r = max(7, round(12 * scale_factor))
    label_yoff = max(10, round(22 * scale_factor))

    try:
        fnt_b = ImageFont.truetype(FONT_BOLD, font_size_b)
        fnt = ImageFont.truetype(FONT_REG, font_size_r)
    except OSError:
        fnt_b = fnt = ImageFont.load_default()

    for i, pred in enumerate(predictions):
        x = pred["x"] * render_scale
        y = pred["y"] * render_scale
        bw = pred["width"] * render_scale
        bh = pred["height"] * render_scale
        x1, y1 = int(x - bw / 2), int(y - bh / 2)
        x2, y2 = int(x + bw / 2), int(y + bh / 2)

        cls = pred.get("class", "unknown")
        meta = get_meta(cls)
        col = "#ff0000"
        r, g, b_c = 255, 0, 0

        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b_c, 22), outline=col, width=box_width)

        t = tick_len
        for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            draw.line([(px, py + dy * t), (px, py), (px + dx * t, py)], fill=col, width=tick_width)

        draw.ellipse([x1 + 2, y1 + 2, x1 + 2 + dot_r * 2, y1 + 2 + dot_r * 2], fill=(r, g, b_c, 200))
        draw.text((x1 + dot_r // 2 + 2, y1 + 3), str(i + 1), fill="white", font=fnt)

        label = f" {cls.upper()} "
        lx, ly = x1, y1 - label_yoff
        if ly < 0:
            ly = y2 + 2
        try:
            bb = draw.textbbox((lx, ly), label, font=fnt_b)
        except AttributeError:
            bb = (lx, ly, lx + len(label) * font_size_b // 2, ly + font_size_b + 4)
        draw.rectangle([bb[0] - 1, bb[1] - 1, bb[2] + 1, bb[3] + 1], fill=(r, g, b_c, 190))
        draw.text((lx, ly), label, fill="white", font=fnt_b)

    if render_scale != 1.0:
        canvas = canvas.resize((orig_w, orig_h), Image.LANCZOS)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def preds_to_csv(preds: List[dict], filename: str = "scan") -> bytes:
    fieldnames = ["#", "File", "Class", "Confidence_%",
                  "Center_X", "Center_Y", "Width", "Height"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for i, p in enumerate(preds):
        cls = p.get("class", "—")
        writer.writerow({
            "#": i + 1,
            "File": filename,
            "Class": cls,
            "Confidence_%": f"{p.get('confidence', 0) * 100:.1f}",
            "Center_X": f"{p.get('x', 0):.0f}",
            "Center_Y": f"{p.get('y', 0):.0f}",
            "Width": f"{p.get('width', 0):.0f}",
            "Height": f"{p.get('height', 0):.0f}",
        })
    return buf.getvalue().encode()


# ── Tiled inference defaults ─────────────────────────────────────────────────
use_tiles = True
tile_size_ui = TILE_SIZE
tile_overlap_ui = TILE_OVERLAP

# ---------------------------------------------------------------------------
# Dash application configuration
# ---------------------------------------------------------------------------

SERVER_SECRET = os.environ.get("DASH_SECRET_KEY", "change-this-in-production")

# The source app used bcrypt hashes. Werkzeug's check_password_hash cannot
# validate those hashes directly, so bcrypt is used when available.
try:
    import bcrypt
except ImportError:
    bcrypt = None

AUTH_USERS = {
    "GPRAdmin": {
        "name": "GPR Administrator",
        "role": "admin",
        "password": "$2b$12$uQWfUK5KDH2SgnJSe8qANekJdMEUVkAR0abNsCsGhx1ODBEbtZsFC",
    },
    "GPRUser": {
        "name": "GPR Operator",
        "role": "user",
        "password": "$2b$12$euUbzy/jpUA4JnG6vFfI/O6CH3gD2clqU24y5ygdFKm4zQKuFOs02",
    },
    "RoboGPR": {
        "name": "Robo GPR Agent",
        "role": "user",
        "password": "$2b$12$xKfYjcua9JmlR7/78ZetlG4.fXCxBji4b8uRpy" if False else
                   "$2b$12$xKfYfjNcqbkO2.c9HfJVcua9JmlR7/78ZetlG4.fXCxBji4b8uRpy",
    },
}

def verify_user(username, password):
    user = AUTH_USERS.get(username)
    if not user:
        return False
    stored = user["password"]
    if bcrypt is not None and stored.startswith("$2"):
        try:
            return bcrypt.checkpw(password.encode(), stored.encode())
        except Exception:
            return False
    return check_password_hash(stored, password)

server = Flask(__name__)
server.secret_key = SERVER_SECRET

@server.route("/logout")
def logout():
    session.clear()
    return redirect("/")

app = Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True,
    title="AVNL-OFMK · AI-Engine",
    update_title="Processing…",
)

# ---------------------------------------------------------------------------
# ORIGINAL INSTITUTIONAL LOGO DELIVERY
# ---------------------------------------------------------------------------
_BRAND_CACHE_DIR = Path(__file__).resolve().parent / "assets" / ".brand_cache"
_BRAND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_BRAND_SPECS = {
    "avnl": (AVNL_OFFICIAL_LOGO_URL, "avnl_official.png"),
}

def _get_brand_logo(kind: str):
    spec = _BRAND_SPECS.get(kind)
    if spec is None:
        abort(404)
    url, filename = spec
    target = _BRAND_CACHE_DIR / filename
    if not target.exists() or target.stat().st_size < 1024:
        try:
            r = requests.get(url, timeout=20, allow_redirects=True,
                             headers={"User-Agent": "GPR-Dash-Application/1.0"})
            r.raise_for_status()
            ctype = (r.headers.get("content-type") or "").lower()
            if "image" not in ctype and not r.content.startswith((b"\x89PNG", b"\xff\xd8", b"RIFF")):
                raise RuntimeError(f"unexpected logo content type: {ctype}")
            target.write_bytes(r.content)
        except Exception as exc:
            logger.warning("Could not cache %s original logo: %s", kind, exc)
            if not target.exists():
                abort(503)
    return send_file(target, conditional=True, max_age=86400)

@server.route("/brand-logo/<kind>")
def brand_logo(kind):
    return _get_brand_logo(kind)

# ---------------------------------------------------------------------------
# Design helpers
# ---------------------------------------------------------------------------

def img_data_uri(img):
    if img is None:
        return None
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def img_from_data_uri(uri):
    if not uri:
        return None
    _, payload = uri.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")

def uploaded_to_bytes(contents):
    if not contents:
        return b""
    _, payload = contents.split(",", 1)
    return base64.b64decode(payload)

def predictions_rows(preds):
    return [{
        "#": i + 1,
        "Class": p.get("class", "—"),
        "Confidence": f"{p.get('confidence', 0)*100:.1f}%",
        "Center X": f"{p.get('x', 0):.0f}",
        "Center Y": f"{p.get('y', 0):.0f}",
        "Width": f"{p.get('width', 0):.0f}",
        "Height": f"{p.get('height', 0):.0f}",
    } for i, p in enumerate(preds)]

def detection_cards(preds):
    cards = []
    for i, p in enumerate(preds, 1):
        cls = p.get("class", "unknown")
        meta = get_meta(cls)
        conf = p.get("confidence", 0) * 100
        tone = meta.get("color", "#4F8CFF")
        cards.append(html.Div([
            html.Div([
                html.Div([
                    html.Div(meta.get("icon", "OBJ"), className="det-icon"),
                    html.Div([
                        html.Div(cls.upper(), className="det-name"),
                        html.Div(f"OBJECT {i:02d}", className="det-tag"),
                    ], className="det-title-group")
                ], className="det-ident"),
                html.Div(f"{conf:.1f}%", className="det-conf",
                         style={"--det-color": tone}),
            ], className="det-head"),
            html.Div(
                html.Div(className="conf-fill",
                         style={"width": f"{min(conf,100):.1f}%",
                                "background": tone}),
                className="conf-track"
            ),
            html.Div([
                html.Span(f"CENTER  {p.get('x',0):.0f} × {p.get('y',0):.0f} px"),
                html.Span(f"SIZE  {p.get('width',0):.0f} × {p.get('height',0):.0f} px"),
            ], className="det-foot"),
        ], className="det-card"))
    return cards or [html.Div("No detections returned.", className="muted")]


def metric_card(title, value, accent=""):
    return html.Div([
        html.Div(title, className="metric-label"),
        html.Div(value, className="metric-value"),
    ], className="metric-card", style={"--accent": accent or "#4F8CFF"})


def kv_card(title, rows):
    return html.Section([
        html.Div([
            html.Span(className="section-mark"),
            html.Span(title, className="sec-label"),
        ], className="section-heading"),
        html.Div([
            html.Div([
                html.Span(k, className="kv-key"),
                html.Span(v, className="kv-val")
            ], className="kv-row") for k, v in rows
        ], className="kv-block")
    ], className="card info-card")


def nav_button(label, page):
    return html.Button(
        [html.Span(label.split("  ")[0], className="nav-index"),
         html.Span(label.split("  ")[-1], className="nav-text")],
        id={"type":"nav","page":page},
        className="nav-btn"
    )


def asset_svg(name: str) -> str:
    """Return a static asset URL served by Dash."""
    return f"/assets/{name}"

def login_layout(error=None):
    # The login background is rendered as a real <img> element rather than
    # relying only on CSS background-image. This guarantees that the exact
    # supplied animated SVG is loaded by Dash and remains visible behind the
    # glassmorphic login card.
    return html.Div([
        html.Img(
            src=asset_svg("GPR_Drone_Live_Scan.svg"),
            className="login-background-svg",
            alt="Live GPR drone scan background",
            draggable="false",
        ),
        html.Div(className="login-background-overlay"),
        html.Div([
            html.Div([
                html.Div("AVNL-OFMK", className="login-logo"),
                html.Div("GPR ANALYTICS PLATFORM", className="login-kicker"),
                html.Div("AI-ENGINE FOR BURIED OBJECT DETECTION", className="login-sub"),
            ], className="login-brand"),
            html.Div([
                html.Div("Secure access", className="login-title"),
                html.Div("Sign in to continue to the detection workspace.",
                         className="login-description"),
                dcc.Input(id="login-user", placeholder="Username", type="text",
                          className="login-input"),
                dcc.Input(id="login-pass", placeholder="Password", type="password",
                          className="login-input"),
                html.Button("Sign in to workspace", id="login-submit", className="login-btn"),
                html.Div(error or "", id="login-error", className="login-error"),
            ], className="login-form"),
            html.Div([
                html.Span("SECURE SESSION"),
                html.Span("•"),
                html.Span("YOLOv26 NANO"),
                html.Span("•"),
                html.Span("GPR / SEG-Y")
            ], className="login-footer"),
        ], className="login-card")
    ], className="login-screen")


def app_layout():
    return html.Div([
        dcc.Store(id="app-state", storage_type="memory",
                  data={"history": [], "total_scans": 0, "last_preds": [],
                        "last_image": None, "last_annotated": None, "last_result": None}),
        dcc.Store(id="current-page", storage_type="session", data="single"),
        dcc.Download(id="download"),

        html.Div([
            html.Div([
                html.Div(className="gpr-background-image"),
                html.Div(className="gpr-background-overlay"),
            ], className="gpr-background"),
            html.Aside([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Img(src=AVNL_LOGO_URL, className="institution-logo institution-logo-avnl",
                                     alt="Armoured Vehicles Nigam Limited (AVNL) logo"),
                        ], className="institution-logo-frame"),
                        html.Div([
                            html.Div("AVNL · OFMK", className="side-brand"),
                            html.Div("ORDNANCE FACTORY MEDAK", className="side-brand-sub"),
                        ], className="side-brand-copy"),
                    ], className="side-brand-row"),
                    html.Div([
                        html.Div([
                            html.Img(src=CMERI_LOGO_URL, className="institution-logo institution-logo-cmeri",
                                     alt="CSIR-CMERI logo"),
                        ], className="institution-logo-frame institution-logo-frame-cmeri"),
                        html.Div("CSIR-CMERI · DURGAPUR", className="institution-partner-label"),
                    ], className="institution-partner-row"),
                ], className="side-brand-wrap"),

                html.Div(id="profile"),

                html.Div([
                    html.Div("WORKSPACE", className="side-nav-label"),
                    nav_button("01  SINGLE SCAN", "single"),
                    nav_button("02  BATCH ANALYSIS", "batch"),
                    nav_button("03  SCAN HISTORY", "history"),
                    nav_button("04  GUIDE", "guide"),
                ], className="side-navigation"),

                html.Div([
                    html.Div("DETECTION PARAMETERS", className="side-nav-label"),
                    html.Label("Confidence threshold", className="field-label"),
                    dcc.Slider(id="confidence", min=10, max=90, step=1, value=35,
                               marks={10:"10",35:"35",50:"50",70:"70",90:"90"}),

                    html.Label("Overlap threshold", className="field-label"),
                    dcc.Slider(id="overlap", min=5, max=60, step=1, value=30,
                               marks={5:"5",25:"25",40:"40",60:"60"}),

                    html.Div("INFERENCE MODE", className="field-label"),
                    dcc.Checklist(id="engine-options",
                        options=[
                            {"label":" Multiscale inference","value":"multi"},
                            {"label":" Pad to square","value":"pad"},
                            {"label":" Tiled inference","value":"tiles"},
                        ], value=["multi","pad","tiles"], className="checklist"),
                ], className="side-panel"),

                html.Div([
                    html.Div("DISPLAY", className="side-nav-label"),
                    dcc.Checklist(id="display-options",
                        options=[
                            {"label":" Detection cards","value":"cards"},
                            {"label":" Detection table","value":"table"},
                            {"label":" Raw JSON response","value":"json"},
                        ], value=["cards","table"], className="checklist"),
                ], className="side-panel"),

                html.Div(id="api-config-panel"),
                html.Div(id="engine-info", className="engine-status-panel"),

                html.A("Sign out", href="/logout", className="logout-btn"),
            ], className="sidebar"),

            html.Main([
                html.Header([
                    html.Div([
                        html.Div("GPR ANALYTICS", className="eyebrow"),
                        html.H1([
                            "AI Engine for ",
                            html.Span("Buried Object Detection", className="accent")
                        ], className="gpr-head-title"),
                        html.Div([
                            html.Span([html.Span(className="live-dot"), " SYSTEM ONLINE"],
                                      className="live-indicator"),
                            html.Span("·"),
                            html.Span(datetime.now().strftime("%d %b %Y")),
                            html.Span("·"),
                            html.Span("YOLOv26 Nano")
                        ], className="gpr-head-sub"),
                    ], className="header-copy"),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Img(src=AVNL_LOGO_URL, className="header-institution-logo header-avnl-logo",
                                         alt="AVNL logo"),
                            ], className="header-logo-frame"),
                            html.Div([
                                html.Div("AVNL · OFMK", className="header-institution-name"),
                                html.Div("Ordnance Factory Medak", className="header-institution-sub"),
                            ]),
                        ], className="header-institution"),
                        html.Div(className="header-logo-divider"),
                        html.Div([
                            html.Div([
                                html.Img(src=CMERI_LOGO_URL, className="header-institution-logo header-cmeri-logo",
                                         alt="CSIR-CMERI logo"),
                            ], className="header-logo-frame header-logo-frame-cmeri"),
                            html.Div([
                                html.Div("CSIR-CMERI", className="header-institution-name"),
                                html.Div("Durgapur", className="header-institution-sub"),
                            ]),
                        ], className="header-institution"),
                        html.Div([
                            html.Div("DETECTION WORKSPACE", className="header-badge"),
                            html.Div("SECURE GPR ANALYTICS", className="header-live-label"),
                        ], className="header-badge-stack"),
                    ], className="header-identity"),
                ], className="gpr-header"),

                html.Div([
                    html.Div([
                        html.Span("Detection workspace", className="tabs-title"),
                        html.Span("Select an analysis mode", className="tabs-subtitle"),
                    ], className="tabs-intro"),
                    html.Div([
                        html.Div(["01", "Single scan"], className="tab-chip active"),
                        html.Div(["02", "Batch analysis"], className="tab-chip"),
                        html.Div(["03", "Scan history"], className="tab-chip"),
                        html.Div(["04", "Guide"], className="tab-chip"),
                    ], className="tabs-actions"),
                ], className="tabs"),

                html.Div(id="page-content")
            ], className="main")
        ], className="shell")
    ])


# ---------------------------------------------------------------------------
# Page components
# ---------------------------------------------------------------------------

def single_page():
    return html.Div([
        html.Div([
            html.Section([
                html.Div([
                    html.Div([
                        html.Span(className="section-mark"),
                        html.Span("01 · INPUT", className="sec-label"),
                    ], className="section-heading"),
                    html.H2("Upload GPR scan", className="page-title"),
                    html.P("Add a SEG-Y raw file for automatic preprocessing, or upload a ready-made PNG/JPEG B-scan.",
                           className="help-text"),
                ], className="card-heading"),

                dcc.Upload(id="single-file", children=html.Div([
                    html.Div("DROP SCAN HERE", className="upload-title"),
                    html.Div(["or ", html.Span("browse files", className="upload-link")],
                             className="upload-subtitle"),
                    html.Div("SEG-Y · SEGY · PNG · JPG · JPEG", className="upload-meta"),
                ]), className="upload-zone", multiple=False,
                           accept=".sgy,.segy,.png,.jpg,.jpeg"),

                html.Div(id="single-preview"),

                html.Details([
                    html.Summary("Advanced SEG-Y preprocessing"),
                    html.Div([
                        html.Div([
                            html.Div("Gain mode", className="field-label"),
                            dcc.Dropdown(id="pp-gain-mode",
                                         options=[{"label":"Linear","value":"linear"}],
                                         value="linear", clearable=False),
                            dcc.RadioItems(id="pp-gain-preset",
                                           options=[{"label":x,"value":x} for x in ["Min","Med","Max"]],
                                           value="Min", inline=True, className="radio-row"),
                        ]),
                        html.Div([
                            html.Div("Background / output", className="field-label"),
                            html.Div("Mean subtraction · trace normalisation · grayscale output",
                                     className="help-text"),
                        ]),
                    ], className="pp-grid")
                ], id="pp-config", open=False),
            ], className="card upload-card"),

            html.Section([
                html.Div([
                    html.Div([
                        html.Span(className="section-mark"),
                        html.Span("02 · INFERENCE", className="sec-label"),
                    ], className="section-heading"),
                    html.H2("Run detection", className="page-title"),
                    html.P("Execute the configured model against the current scan.",
                           className="help-text"),
                ], className="card-heading"),
                html.Button("Run inference", id="run-single", className="primary-btn"),
                html.Div([
                    html.Span(className="status-dot"),
                    html.Span("Ready for inference.", id="single-status")
                ], className="inference-status"),
                dcc.Loading(
                    id="single-inference-loading",
                    type="circle",
                    color="#4F8CFF",
                    children=html.Div(id="single-output", className="output-area")
                ),
            ], className="card inference-card")
        ], className="two-col"),

        html.Div([
            html.Div("Pipeline", className="section-kicker"),
            html.Div([
                html.Div(["01", html.Span("Upload & preview")], className="pipeline-step"),
                html.Div(["02", html.Span("Preprocess")], className="pipeline-step"),
                html.Div(["03", html.Span("Detect")], className="pipeline-step"),
                html.Div(["04", html.Span("Review & export")], className="pipeline-step"),
            ], className="pipeline")
        ], className="pipeline-card")
    ])


def batch_page():
    return html.Div([
        html.Section([
            html.Div([
                html.Div([
                    html.Span(className="section-mark"),
                    html.Span("01 · BATCH INPUT", className="sec-label"),
                ], className="section-heading"),
                html.H2("Process multiple scans", className="page-title"),
                html.P("Queue GPR images or SEG-Y files and process the complete set with one run.",
                       className="help-text"),
            ], className="card-heading"),
            dcc.Upload(id="batch-files", children=html.Div([
                html.Div("DROP MULTIPLE SCANS", className="upload-title"),
                html.Div(["or ", html.Span("browse files", className="upload-link")],
                         className="upload-subtitle"),
                html.Div("SEG-Y · PNG · JPG · BMP · TIFF", className="upload-meta"),
            ]), className="upload-zone", multiple=True,
                       accept=".sgy,.segy,.png,.jpg,.jpeg,.bmp,.tiff"),
            html.Div(id="batch-queue"),
        ], className="card"),

        html.Section([
            html.Div([
                html.Div([
                    html.Span(className="section-mark"),
                    html.Span("02 · PROCESSING", className="sec-label"),
                ], className="section-heading"),
                html.H2("Batch controls", className="page-title"),
            ], className="card-heading"),
            html.Div([
                html.Div([
                    html.Div("Gain level", className="field-label"),
                    dcc.RadioItems(id="batch-gain-preset",
                                   options=[{"label":x,"value":x} for x in ["Min","Med","Max"]],
                                   value="Min", inline=True, className="radio-row"),
                ]),
                html.Button("Process all scans", id="run-batch", className="primary-btn"),
            ], className="batch-controls")
        ], className="card"),

        html.Div(id="batch-output")
    ], className="page-stack")


def history_page():
    return html.Div([
        html.Section([
            html.Div([
                html.Div([
                    html.Span(className="section-mark"),
                    html.Span("SESSION", className="sec-label"),
                ], className="section-heading"),
                html.H2("Scan history", className="page-title"),
                html.P("Review scans processed during the current authenticated session.",
                       className="help-text"),
            ], className="card-heading"),
            html.Div(id="history-output"),
            html.Button("Clear session history", id="clear-history", className="danger-btn")
        ], className="card")
    ])


def guide_page():
    return html.Div([
        html.Div([
            kv_card("Recommended settings", [
                ("High-clutter soil", "Confidence ≥ 50%"),
                ("Clean / dry soil", "Confidence ≥ 35%"),
                ("Dense object fields", "Overlap ≤ 25%"),
                ("Sparse scenes", "Overlap ≤ 40%"),
            ]),
            kv_card("SEG-Y preprocessing", [
                ("Dewow", "Enabled · window 39"),
                ("Bandpass", "Enabled · 100–900 MHz · order 4"),
                ("Background", "Mean trace removal"),
                ("Gain", "Linear · 30 dB"),
                ("Output", "640 × 640 JPEG · quality 95"),
            ]),
        ], className="guide-grid"),
        html.Div([
            kv_card("Batch processing", [
                ("SEG-Y", "Dewow + bandpass + background removal + gain"),
                ("Images", "PNG/JPEG processed directly"),
                ("Report", "Consolidated CSV + per-file previews"),
            ]),
            kv_card("Detection engine", [
                ("Model", MODEL_ID),
                ("Provider", "Roboflow"),
                ("Inference", "YOLOv26 Nano"),
                ("Tiling", f"{TILE_SIZE}px / {TILE_OVERLAP}px overlap"),
            ]),
        ], className="guide-grid"),
    ], className="page-stack")

# ---------------------------------------------------------------------------
# Authentication / routing
# ---------------------------------------------------------------------------

@app.callback(
    Output("auth-overlay", "style"),
    Output("login-error", "children"),
    Input("login-submit", "n_clicks"),
    Input("auth-refresh", "n_intervals"),
    State("login-user", "value"),
    State("login-pass", "value"),
    prevent_initial_call=False,
)
def do_login(n, _refresh, username, password):
    """Keep authentication UI mounted while preserving the Dash component tree."""
    if session.get("authenticated"):
        return {"display": "none"}, ""
    if not n:
        return {}, ""
    username = username or ""
    password = password or ""
    if verify_user(username, password):
        session["authenticated"] = True
        session["username"] = username
        logger.info("User authenticated: %s", username)
        return {"display": "none"}, ""
    logger.warning("Authentication failed for user: %s", username)
    return {}, "⛔ ACCESS DENIED — Invalid credentials"

@app.callback(
    Output("page-content", "children"),
    Input("current-page", "data"),
    Input("login-submit", "n_clicks"),
)
def render_page(page, _):
    return {"single": single_page, "batch": batch_page,
            "history": history_page, "guide": guide_page}.get(page, single_page)()

@app.callback(
    Output("current-page", "data"),
    Input({"type":"nav","page":ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def navigate(clicks):
    triggered = ctx.triggered_id
    if not triggered:
        raise PreventUpdate
    return triggered["page"]

@app.callback(
    Output("api-config-panel", "children"),
    Input("auth-refresh", "n_intervals"),
    Input("login-submit", "n_clicks"),
)
def api_config_panel(_, __):
    user = AUTH_USERS.get(session.get("username"), {})
    if user.get("role") != "admin":
        return html.Div()
    configured = bool(_get_api_key())
    status = "CONFIGURED" if configured else "NOT CONFIGURED"
    status_color = "#4F8CFF" if configured else "#F05252"
    return html.Div([
        html.Div("ROBOFLOW API", className="sec-label"),
        html.Div([
            html.Span("● ", style={"color": status_color}),
            html.Span(status),
        ], style={"fontFamily":"IBM Plex Mono,monospace","fontSize":"0.65rem",
                  "letterSpacing":"1.5px","marginBottom":"8px","color":status_color}),
        dcc.Input(
            id="api-key-input", type="password", placeholder="Paste API key for this session",
            debounce=True, className="api-key-input"
        ),
        html.Button("SAVE KEY", id="api-key-save", className="secondary-btn",
                    style={"marginTop":"7px","width":"100%"}),
        html.Div(id="api-key-status", style={"fontSize":"0.65rem","marginTop":"6px"})
    ], className="api-config")

@app.callback(
    Output("api-key-status", "children"),
    Input("api-key-save", "n_clicks"),
    State("api-key-input", "value"),
    prevent_initial_call=True,
)
def save_api_key(n_clicks, value):
    if AUTH_USERS.get(session.get("username"), {}).get("role") != "admin":
        return "Administrator access required."
    key = (value or "").strip()
    if not key:
        session.pop("roboflow_api_key", None)
        return "Session key cleared; using configured secret/environment key."
    if len(key) < 8:
        return "Invalid key format."
    session["roboflow_api_key"] = key
    logger.info("Roboflow API key updated for current authenticated session.")
    return "✓ API key saved for this session."

@app.callback(
    Output("profile", "children"),
    Output("engine-info", "children"),
    Input("auth-refresh", "n_intervals"),
    Input("login-submit", "n_clicks"),
)
def profile(_, __):
    user = AUTH_USERS.get(session.get("username"), {})
    name = user.get("name", "Not authenticated")
    role = "ADMINISTRATOR" if user.get("role") == "admin" else ("OPERATOR" if user else "GUEST")
    color = "#4F8CFF" if role == "ADMINISTRATOR" else "#7C8CF8"
    profile = html.Div([
        html.Div(name, className="user-name"),
        html.Div([html.Span("●", style={"color":color}), f" {role}",
                  html.Span(session.get("username", ""), className="user-username")],
                 className="role-line")
    ], className="user-profile")
    key_source = "SESSION" if session.get("roboflow_api_key") else (
        "ENVIRONMENT / SECRETS" if ROBOFLOW_API_KEY else "NOT CONFIGURED"
    )
    rows = [
        ("Model", MODEL_ID), ("Provider", "Roboflow"),
        ("Type", "YOLOv26 Detection"), ("Multiscale", "ON"),
        ("Tile", f"{TILE_SIZE} px"), ("Overlap", f"{TILE_OVERLAP} px"),
        ("Input", "SEG-Y / PNG / JPG"), ("Target", "640 × 640"),
        ("API Key", key_source),
    ]
    info = kv_card("ENGINE STATUS", rows)
    return profile, info

# ---------------------------------------------------------------------------
# Root layout
# ---------------------------------------------------------------------------

# The entire Dash component tree is present from the first request.
# Authentication is a fixed overlay rather than a dynamic replacement of
# the root component. This is substantially more reliable across Dash 2.x/3.x
# versions and prevents the white/black blank-page failure seen with dynamic
# root layouts.
app.layout = html.Div([
    dcc.Interval(id="auth-refresh", interval=60_000, n_intervals=0),
    app_layout(),
    html.Div(login_layout(), id="auth-overlay", className="auth-overlay"),
])

# ---------------------------------------------------------------------------
# Single scan
# ---------------------------------------------------------------------------

@app.callback(
    Output("single-preview", "children"),
    Output("run-single", "disabled"),
    Input("single-file", "contents"),
    State("single-file", "filename"),
    prevent_initial_call=True,
)
def preview_single(contents, filename):
    if not contents or not filename:
        return html.Div(), True
    try:
        data = uploaded_to_bytes(contents)
        if filename.lower().endswith((".sgy",".segy")):
            raw, dt, meta = pp_read_sgy(data, filename)
            pil = pp_plot_bscan_plain(raw, target_size=(640,640))
            return html.Div([
                html.Img(src=img_data_uri(pil), className="scan-image"),
                html.Div(meta, className="meta-line")
            ]), False
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return html.Div([
            html.Img(src=img_data_uri(img), className="scan-image"),
            html.Div(f"{filename} · {img.size[0]} × {img.size[1]} px", className="meta-line")
        ]), False
    except Exception as e:
        return html.Div(f"Preview failed: {e}", className="error-box"), True

@app.callback(
    Output("single-output", "children"),
    Output("app-state", "data"),
    Input("run-single", "n_clicks"),
    State("single-file", "contents"),
    State("single-file", "filename"),
    State("confidence", "value"),
    State("overlap", "value"),
    State("engine-options", "value"),
    State("pp-gain-preset", "value"),
    State("app-state", "data"),
    State("display-options", "value"),
    prevent_initial_call=True,
)
def run_single(n, contents, filename, confidence, overlap, engine_opts,
               gain_preset, state, display_opts):
    if not n or not contents or not filename:
        raise PreventUpdate
    state = state or {"history":[],"total_scans":0}
    try:
        raw_bytes = uploaded_to_bytes(contents)
        is_sgy = filename.lower().endswith((".sgy",".segy"))
        if is_sgy:
            gain_db = {"Min":5.0,"Med":15.0,"Max":25.0}.get(gain_preset,5.0)
            cfg = dict(_PP_DEFAULT_CFG)
            cfg.update({
                "gain_mode":"linear", "gain_db":gain_db,
                "apply_dewow":True, "dewow_window":39,
                "apply_bandpass":True, "bp_low_MHz":100.0,
                "bp_high_MHz":900.0, "bp_order":4,
                "trace_normalise":True,
            })
            pp = pp_run_pipeline(raw_bytes, filename, cfg)
            if pp["status"] != "OK":
                return html.Div([html.Div("Preprocessing failed", className="error-box"),
                                 html.Pre("\n".join(pp["log"]))]), state
            img = pp["output_pil"]
            meta = pp.get("meta","")
        else:
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            side = min(img.size)
            img = img.resize((side,side), Image.Resampling.LANCZOS)
            meta = f"{filename} · {img.size[0]} × {img.size[1]} px"

        t0 = time.time()
        result = run_inference(
            img, int(confidence), int(overlap),
            tile="tiles" in (engine_opts or []),
            tile_px=TILE_SIZE, tile_ov=TILE_OVERLAP,
            multi_scale="multi" in (engine_opts or []),
            pad_square="pad" in (engine_opts or []),
        )
        elapsed = (time.time()-t0)*1000
        preds = result.get("predictions", [])
        annotated = draw_detections(img, preds)
        ann_uri = img_data_uri(annotated)
        img_uri = img_data_uri(img)

        state["total_scans"] = int(state.get("total_scans",0)) + 1
        record = {
            "id":state["total_scans"], "file":filename,
            "time":datetime.now().strftime("%H:%M:%S"),
            "preds":preds, "size":f"{img.size[0]}×{img.size[1]}",
            "ms":f"{elapsed:.0f}ms", "image":img_uri,
        }
        state.setdefault("history", []).append(record)
        state["last_preds"] = preds
        state["last_image"] = img_uri
        state["last_annotated"] = ann_uri
        state["last_result"] = result

        parts = [
            html.Div([html.Img(src=ann_uri, className="result-image")], className="result-frame"),
            html.Div([
                metric_card("OBJECTS", len(preds)),
                metric_card("AVG CONF.", f"{(sum(p.get('confidence',0) for p in preds)/len(preds)*100 if preds else 0):.1f}%"),
                metric_card("LATENCY", f"{elapsed:.0f} ms"),
            ], className="metric-grid"),
            html.Div([
                html.Button("⬇  ANNOTATED IMAGE", id="download-ann", className="secondary-btn"),
                html.Button("⬇  CSV REPORT", id="download-csv", className="secondary-btn"),
            ], className="download-row"),
        ]
        if "cards" in (display_opts or []):
            parts += [html.Div("OBJECT DETAILS", className="sec-label"), html.Div(detection_cards(preds))]
        if "table" in (display_opts or []) and preds:
            parts += [html.Div("DETECTION TABLE", className="sec-label"),
                      dash_table.DataTable(
                          data=predictions_rows(preds),
                          columns=[{"name":k,"id":k} for k in predictions_rows(preds)[0]],
                          style_table={"overflowX":"auto"}, style_as_list_view=True,
                          style_header={"backgroundColor":"#111827","color":"#DCE7F3","fontWeight":"600"},
                          style_cell={"backgroundColor":"#0B1220","color":"#D7E0EA","border":"1px solid #223047"}
                      )]
        if "json" in (display_opts or []):
            parts += [html.Details([html.Summary("📄 Raw JSON Response"),
                                    html.Pre(json.dumps(result, indent=2))])]

        return html.Div(parts), state
    except Exception as e:
        logger.exception("Single inference failed")
        return html.Div(f"{type(e).__name__}: {e}", className="error-box"), state

@app.callback(
    Output("download", "data"),
    Input("download-ann", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def download_annotation(n, state):
    if not n or not state or not state.get("last_annotated"):
        raise PreventUpdate
    img = img_from_data_uri(state["last_annotated"])
    return dcc.send_bytes(lambda b: img.save(b, format="PNG"), "gpr_annotated.png")

@app.callback(
    Output("download", "data", allow_duplicate=True),
    Input("download-csv", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def download_csv(n, state):
    if not n or not state:
        raise PreventUpdate
    preds = state.get("last_preds", [])
    return dcc.send_bytes(lambda b: b.write(preds_to_csv(preds, "scan")), "gpr_report.csv")

# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

@app.callback(
    Output("batch-queue", "children"),
    Input("batch-files", "filename"),
)
def batch_queue(names):
    if not names:
        return html.Div("No files queued.", className="empty-state")
    sgy = sum(str(n).lower().endswith((".sgy",".segy")) for n in names)
    return html.Div(f"📂 {len(names)} scan(s) queued · {sgy} SEG-Y · {len(names)-sgy} images",
                    className="queue-banner")

@app.callback(
    Output("batch-output", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("run-batch", "n_clicks"),
    State("batch-files", "contents"),
    State("batch-files", "filename"),
    State("confidence", "value"),
    State("overlap", "value"),
    State("engine-options", "value"),
    State("batch-gain-preset", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def run_batch(n, contents, names, confidence, overlap, engine_opts, gain_preset, state):
    if not n or not contents or not names:
        raise PreventUpdate
    state = state or {"history":[],"total_scans":0}
    results, details, all_preds = [], [], []
    total_det = successful = 0
    gain_db = {"Min":5.0,"Med":15.0,"Max":25.0}.get(gain_preset,5.0)

    for content, filename in zip(contents, names):
        is_sgy = filename.lower().endswith((".sgy",".segy"))
        try:
            raw = uploaded_to_bytes(content)
            if is_sgy:
                cfg = dict(_PP_DEFAULT_CFG)
                cfg.update({"gain_mode":"linear","gain_db":gain_db,
                            "apply_dewow":True,"dewow_window":39,
                            "apply_bandpass":True,"bp_low_MHz":100.0,
                            "bp_high_MHz":900.0,"bp_order":4,
                            "trace_normalise":True})
                pp = pp_run_pipeline(raw, filename, cfg)
                if pp["status"] != "OK":
                    raise RuntimeError("Preprocessing failed")
                img = pp["output_pil"]
            else:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                side = min(img.size)
                img = img.resize((side,side), Image.Resampling.LANCZOS)

            res = run_inference(img, int(confidence), int(overlap),
                                tile="tiles" in (engine_opts or []),
                                tile_px=TILE_SIZE, tile_ov=TILE_OVERLAP,
                                multi_scale="multi" in (engine_opts or []),
                                pad_square="pad" in (engine_opts or []))
            preds = res.get("predictions", [])
            ann = draw_detections(img, preds)
            avg = sum(p.get("confidence",0) for p in preds)/len(preds)*100 if preds else 0
            classes = sorted(set(p.get("class","") for p in preds))
            results.append({"File":filename,"Type":"SEG-Y" if is_sgy else "Image",
                            "Detections":len(preds),"Avg Conf %":f"{avg:.1f}",
                            "Classes":", ".join(classes) or "—","Status":"✅ OK"})
            details.append({"name":filename,"image":img_data_uri(ann),
                            "preds":preds,"error":None})
            total_det += len(preds); successful += 1
            state["total_scans"] = int(state.get("total_scans",0))+1
            state.setdefault("history",[]).append({
                "id":state["total_scans"],"file":filename,
                "time":datetime.now().strftime("%H:%M:%S"),"preds":preds,
                "size":f"{img.size[0]}×{img.size[1]}","ms":"—",
                "image":img_data_uri(img)
            })
            for p in preds:
                q = dict(p); q["_file"] = filename; all_preds.append(q)
        except Exception as e:
            results.append({"File":filename,"Type":"SEG-Y" if is_sgy else "Image",
                            "Detections":"ERR","Avg Conf %":"—",
                            "Classes":str(e)[:70],"Status":"❌"})
            details.append({"name":filename,"image":None,"preds":[],"error":str(e)})

    valid = [r for r in results if isinstance(r["Detections"],int)]
    cards = html.Div([
        metric_card("SCANS", len(names)),
        metric_card("SUCCESSFUL", successful),
        metric_card("OBJECTS FOUND", total_det),
        metric_card("CLEAN SCANS", sum(r["Detections"]==0 for r in valid)),
    ], className="metric-grid")
    table = dash_table.DataTable(
        data=results, columns=[{"name":k,"id":k} for k in results[0]] if results else [],
        style_table={"overflowX":"auto"}, style_as_list_view=True,
        style_header={"backgroundColor":"#111827","color":"#DCE7F3","fontWeight":"600"},
        style_cell={"backgroundColor":"#0B1220","color":"#D7E0EA","border":"1px solid #223047"}
    )
    previews = []
    for d in details:
        children = [html.Div(f"{d['name']} · {len(d['preds'])} detection(s)", className="preview-title")]
        if d["image"]:
            children.append(html.Img(src=d["image"], className="result-image"))
            children.extend(detection_cards(d["preds"]))
        else:
            children.append(html.Div(d["error"], className="error-box"))
        previews.append(html.Details(children))
    return html.Div([
        html.Div("BATCH SUMMARY", className="sec-label"), cards,
        html.Div("DETAILED REPORT", className="sec-label"),
        html.Div([html.Button("⬇  BATCH CSV REPORT", id="download-batch-csv", className="secondary-btn")],
                 className="download-row"),
        table,
        html.Div("PER-FILE PREVIEWS", className="sec-label"), html.Div(previews, className="preview-list")
    ]), state

# Batch report download. The current batch table is reconstructed from session state.
@app.callback(
    Output("download", "data", allow_duplicate=True),
    Input("download-batch-csv", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def download_batch_csv(n, state):
    if not n or not state:
        raise PreventUpdate
    history = state.get("history", [])
    rows = []
    for r in history:
        preds = r.get("preds", [])
        avg = (sum(p.get("confidence", 0) for p in preds) / len(preds) * 100) if preds else 0
        rows.append({
            "File": r.get("file", ""),
            "Type": "GPR scan",
            "Detections": len(preds),
            "Avg Conf %": f"{avg:.1f}",
            "Classes": ", ".join(sorted(set(p.get("class", "") for p in preds))) or "—",
            "Size": r.get("size", ""),
            "Latency": r.get("ms", "—"),
        })
    if not rows:
        raise PreventUpdate
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return dcc.send_string(buf.getvalue(), "gpr_batch_report.csv")

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@app.callback(
    Output("history-output", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("current-page", "data"),
    Input("clear-history", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def render_history(page, clear_clicks, state):
    state = state or {"history":[],"total_scans":0}
    if ctx.triggered_id == "clear-history":
        state = {"history":[],"total_scans":0,"last_preds":[],"last_image":None,"last_result":None}
    history = list(reversed(state.get("history",[])))
    if not history:
        return html.Div("📋 NO SCANS LOGGED YET", className="empty-state"), state
    rows = []
    for r in history:
        preds = r.get("preds",[])
        avg = sum(p.get("confidence",0) for p in preds)/len(preds)*100 if preds else 0
        rows.append({"Scan #":r["id"],"File":r["file"],"Time":r["time"],
                     "Size":r["size"],"Detections":len(preds),
                     "Avg Conf %":f"{avg:.1f}",
                     "Classes":", ".join(sorted(set(p.get("class","") for p in preds))) or "—",
                     "Latency":r.get("ms","—")})
    table = dash_table.DataTable(
        data=rows, columns=[{"name":k,"id":k} for k in rows[0]],
        style_table={"overflowX":"auto"}, style_as_list_view=True,
        style_header={"backgroundColor":"#111827","color":"#DCE7F3","fontWeight":"600"},
        style_cell={"backgroundColor":"#0B1220","color":"#D7E0EA","border":"1px solid #223047"}
    )
    previews=[]
    for r in history:
        ann = draw_detections(img_from_data_uri(r["image"]), r["preds"]) if r.get("image") else None
        if ann:
            previews.append(html.Details([
                html.Summary(f"Scan #{r['id']} · {r['file']} · {len(r['preds'])} object(s)"),
                html.Img(src=img_data_uri(ann), className="result-image"),
                html.Div(detection_cards(r["preds"]))
            ]))
    return html.Div([table, html.Div("SCAN PREVIEWS", className="sec-label"),
                     html.Div(previews, className="preview-list")]), state

# ---------------------------------------------------------------------------
# Root layout and server entry point
# ---------------------------------------------------------------------------

# IMPORTANT: keep the complete component tree mounted from the first request.
# Replacing app.layout with a bare "root" div causes Dash pages/callback targets
# to disappear and produces a blank page. The authentication screen is an
# overlay, so the application remains mounted and all navigation targets exist.
app.layout = html.Div([
    dcc.Interval(id="auth-refresh", interval=60_000, n_intervals=0),
    app_layout(),
    html.Div(login_layout(), id="auth-overlay", className="auth-overlay"),
])

@server.route("/health")
def health():
    return {
        "status": "ok",
        "application": "GPR Dash",
        "authenticated": bool(session.get("authenticated")),
        "roboflow_configured": bool(ROBOFLOW_API_KEY),
        "model": MODEL_ID,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8050")),
        debug=os.environ.get("DASH_DEBUG", "0") == "1",
    )




