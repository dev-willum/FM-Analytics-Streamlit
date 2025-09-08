# fmApp.py — FM Analytics
# - Role Score = weighted composite of per-stat percentiles (0–100), not re-percentiled
# - Plotly hoverlabel: white background, black text & border
# - Scatter plots restricted to players in selected sidebar positions (BASELINE_DF)
# - Unique position tokens, minimum minutes filter, pizzas, leaders, player finder, PCA, etc.

import os, io, re, json
from io import BytesIO
from uuid import uuid4
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# ======== CONFIG =========
# =========================
st.set_page_config(page_title="FM Analytics", layout="wide")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(APP_DIR, "fonts")
GABARITO_REG = os.path.join(FONT_DIR, "Gabarito-Regular.ttf")
GABARITO_BOLD = os.path.join(FONT_DIR, "Gabarito-Bold.ttf")

def _fontprops_or_fallback(ttf_path: str, fallback_family: str = "DejaVu Sans"):
    try:
        if os.path.isfile(ttf_path):
            try:
                fm.fontManager.addfont(ttf_path)
            except Exception:
                pass
            return fm.FontProperties(fname=ttf_path)
    except Exception:
        pass
    return fm.FontProperties(family=fallback_family)

font_normal = _fontprops_or_fallback(GABARITO_REG)
font_bold   = _fontprops_or_fallback(GABARITO_BOLD)

mpl.rcParams["font.family"] = ["Gabarito", "DejaVu Sans", "Arial", "sans-serif"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Gabarito:wght@400;700&display=swap');
:root, body, .stApp { --app-font: 'Gabarito', 'DejaVu Sans', Arial, sans-serif; }
.stApp * { font-family: var(--app-font) !important; }
</style>
""", unsafe_allow_html=True)

POSTER_BG = "#f1ffcd"
FONT_FAMILY = "Gabarito, DejaVu Sans, Arial, sans-serif"

# =========================
# ======= UTILITIES =======
# =========================
def _clean_num(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(",", "", regex=False)
              .str.replace("%", "", regex=False)
              .str.replace("−", "-", regex=False)
              .str.strip())

def _maybe_numeric(col: pd.Series) -> pd.Series:
    s_clean = _clean_num(col)
    num = pd.to_numeric(s_clean, errors="coerce")
    valid = int(num.notna().sum())
    if valid >= max(4, int(0.55 * len(num))):
        return num
    return col

def _series_for(df: pd.DataFrame, col_name: str) -> pd.Series:
    v = df[col_name]
    if isinstance(v, pd.Series):
        return v
    if getattr(v, "shape", None) and len(v.shape) == 2 and v.shape[1] == 1:
        return v.iloc[:, 0]
    if getattr(v, "shape", None) and len(v.shape) == 2:
        scores = []
        for i in range(v.shape[1]):
            s = pd.to_numeric(_clean_num(v.iloc[:, i]), errors="coerce")
            scores.append(int(s.notna().sum()))
        best_i = int(np.argmax(scores))
        return v.iloc[:, best_i]
    return v

def find_col(df: pd.DataFrame, names: List[str]) -> str | None:
    cols = {re.sub(r"\s+", "", str(c)).lower(): str(c) for c in df.columns}
    for nm in names:
        key = re.sub(r"\s+", "", str(nm)).lower()
        if key in cols:
            return cols[key]
    return None

# percentile direction map
LESS_IS_BETTER: Dict[str, bool] = {}
def set_less_is_better(stat: str, flag: bool) -> None:
    LESS_IS_BETTER[str(stat)] = bool(flag)
def is_less_better(stat: str) -> bool:
    return bool(LESS_IS_BETTER.get(str(stat), False))

# ---- Percentile helpers ----
def compute_percentiles_for(player_row: pd.Series, stat_cols: List[str], base_df: pd.DataFrame):
    rows = []
    for s in stat_cols:
        if s not in base_df.columns:
            continue
        series = pd.to_numeric(base_df[s], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        v = pd.to_numeric(player_row.get(s), errors="coerce")
        if series.empty or pd.isna(v):
            rows.append((s, np.nan, None if pd.isna(v) else float(v))); continue
        arr = series.to_numpy(); val = float(v)
        if is_less_better(s): arr, val = -arr, -val
        arr.sort()
        lo = np.searchsorted(arr, val, side="left")
        hi = np.searchsorted(arr, val, side="right")
        pct = ((lo + hi) / 2) / arr.size * 100.0
        rows.append((s, float(np.clip(pct, 0.0, 100.0)), float(v)))
    return rows

def column_percentiles(v_series: pd.Series, base_series: pd.Series, less_better: bool) -> pd.Series:
    """Return percentiles (0–100) of v_series vs base_series, mid-rank method."""
    ref = pd.to_numeric(base_series, errors="coerce").dropna().to_numpy()
    x = pd.to_numeric(v_series, errors="coerce").to_numpy()
    out = np.full_like(x, np.nan, dtype=float)
    if ref.size == 0:
        return pd.Series(out, index=v_series.index)
    if less_better:
        ref = -ref; x = -x
    ref.sort()
    lo = np.searchsorted(ref, x, side="left")
    hi = np.searchsorted(ref, x, side="right")
    mask = ~np.isnan(x)
    out[mask] = ((lo[mask] + hi[mask]) / 2) / ref.size * 100.0
    out = np.clip(out, 0.0, 100.0)
    return pd.Series(out, index=v_series.index)

# Plotly styling helpers: black axes + white hover
def _plotly_axes_black(fig):
    fig.update_layout(
        font=dict(family=FONT_FAMILY, color="#000"),
        hoverlabel=dict(bgcolor="white", bordercolor="#000", font=dict(color="#000", family=FONT_FAMILY))
    )
    fig.update_xaxes(title_font=dict(color="#000"), tickfont=dict(color="#000"))
    fig.update_yaxes(title_font=dict(color="#000"), tickfont=dict(color="#000"))
    return fig

def _plotly_polar_black(fig, rmax=100):
    fig.update_layout(
        font=dict(family=FONT_FAMILY, color="#000"),
        hoverlabel=dict(bgcolor="white", bordercolor="#000", font=dict(color="#000", family=FONT_FAMILY)),
        polar=dict(
            radialaxis=dict(range=[0, rmax], tickfont=dict(color="#000"),
                            gridcolor="rgba(0,0,0,0.15)", linecolor="#000", showline=True),
            angularaxis=dict(tickfont=dict(color="#000"))
        )
    )
    return fig

# =========================
# ====== HTML PARSING =====
# =========================
def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for c in out.columns:
        out[c] = _series_for(out, c)
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    return out

def read_fm_html(path_or_buf) -> pd.DataFrame:
    try:
        tables = pd.read_html(path_or_buf, flavor="bs4", header=0)
    except Exception:
        tables = pd.read_html(path_or_buf, header=0)
    if not tables:
        raise RuntimeError("No tables found in HTML.")
    def _score(dfx: pd.DataFrame) -> tuple[int, int]:
        coerced = dfx.apply(pd.to_numeric, errors="coerce")
        return (dfx.shape[1], int(coerced.notna().to_numpy().sum()))
    idx = max(range(len(tables)), key=lambda i: _score(tables[i]))
    df = _sanitize_df(tables[idx])
    if df.shape[1] >= 2:
        c0 = df.columns[0]
        as_num = pd.to_numeric(df[c0], errors="coerce")
        if as_num.notna().mean() > 0.9 and as_num.dropna().between(0, 9999).mean() > 0.8:
            df = df.drop(columns=[c0])
    return df

# =========================
# ====== REMAPPING ========
# =========================
RENAME_MAP = {
    "Name":"Name","Position":"Pos","Age":"Age","Weight":"Weight","Height":"Height",
    "Inf":"Info","Club":"Club","Division":"League","Nat":"Nat","2nd Nat":"2nd Nat",
    "Home-Grown Status":"Home-Grown Status","Personality":"Personality","Media Handling":"Media Handling",
    "Wage":"Wage","Transfer Value":"Transfer Value","Asking Price":"Asking Price","Preferred Foot":"Preferred Foot",
    "Yel":"Yellow Cards","xG":"Expected Goals","Starts":"Starts","Red":"Red Cards","PoM":"Player of the Match",
    "Pen/R":"Pens Scored Ratio","Pens S":"Pens Scored","Pens Saved Ratio":"Pens Saved Ratio","Pens Saved":"Pens Saved",
    "Pens Faced":"Pens Faced","Pens":"Pens","Mins":"Minutes","Gls/90":"Goals / 90","Conc":"Conceded","Gls":"Goals",
    "Fls":"Fouls","FA":"Fouled","xG/90":"xG/90","xG-OP":"xG Overperformance","xA/90":"Expected Assists/90","xA":"Expected Assists",
    "Con/90":"Conceded/90","Clean Sheets":"Clean Sheets","Cln/90":"Clean Sheets/90","Av Rat":"Avg Rating",
    "Mins/Gl":"Minutes / Goal","Ast":"Assist","Hdrs A":"Headers Attempted","Apps":"Appearances",
    "Tck/90":"Tackles/90","Tck W":"Tackles Won","Tck A":"Tackles Attempted","Tck R":"Tackle Ratio",
    "Shot/90":"Shots/90","Shot %":"Shot on Target Ratio","ShT/90":"SoT/90","ShT":"Shots on Target",
    "Shots Outside Box/90":"Shots Outside Box/90","Shts Blckd/90":"Shots Blocked/90","Shts Blckd":"Shots Blocked",
    "Shots":"Shots","Svt":"Saves Tipped","Svp":"Saves Parried","Svh":"Saves Held","Sv %":"Save %",
    "Pr passes/90":"Progressive Passes/90","Pr Passes":"Progressive Passes",
    "Pres C/90":"Pressures Completed/90","Pres C":"Pressures Completed","Pres A/90":"Pressures Attempted/90","Pres A":"Pressures Attempted",
    "Poss Won/90":"Possession Won/90","Poss Lost/90":"Possession Lost/90","Ps C/90":"Passes Completed/90","Ps C":"Passes Completed",
    "Ps A/90":"Passes Attempted/90","Pas A":"Passes Attempted","Pas %":"Pass Completion%",
    "OP-KP/90":"Open Play Key Passes/90","OP-KP":"Open Play Key Passes",
    "OP-Crs C/90":"Open Play Crosses Completed/90","OP-Crs C":"Open Play Crosses Completed",
    "OP-Crs A/90":"Open Play Crosses Attempted/90","OP-Crs A":"Open Play Crosses Attempted","OP-Cr %":"Open Play Cross Completion Ratio",
    "Off":"Offsides","Gl Mst":"Mistakes Leading to Goal","K Tck/90":"Key Tackles/90","K Tck":"Key Tackles",
    "K Ps/90":"Key Passes/90","K Pas":"Key Passes","K Hdrs/90":"Key Headers/90","Int/90":"Interceptions/90","Itc":"Interceptions",
    "Sprints/90":"Sprint/90","Hdr %":"Header Win Rate","Hdrs W/90":"Headers won/90","Hdrs":"Headers","Hdrs L/90":"Headers Lost/90",
    "Goals Outside Box":"Goals Outside Box","xSv %":"Expected Save %","xGP/90":"Expected Goals Prevented/90","xGP":"Expected Goals Prevented",
    "Drb/90":"Dribbles/90","Drb":"Dribbles","Distance":"Distance Covered (KM)","Cr C/90":"Crosses Completed/90","Cr C":"Crosses Completed",
    "Crs A/90":"Crosses Attempted/90","Cr A":"Crosses Attempted","Cr C/A":"Cross Completion Ratio","Conv %":"Conversion Rate",
    "Clr/90":"Clearances/90","Clear":"Clearances","CCC":"Chances Created","Ch C/90":"Chances Created/90","Blk/90":"Blocks/90","Blk":"Blocks",
}

CREATE_PER90_FROM_TOTAL = {
    "Tackles Won": "Tackles Won/90",
    "Interceptions": "Interceptions/90",
    "Shots Blocked": "Shots Blocked/90",
    "Blocks": "Blocks/90",
    "Clearances": "Clearances/90",
    "Chances Created": "Chances Created/90",
    "Expected Goals Prevented": "Expected Goals Prevented/90",
    "Crosses Completed": "Crosses Completed/90",
    "Crosses Attempted": "Crosses Attempted/90",
    "Key Passes": "Key Passes/90",
    "Key Headers": "Key Headers/90",
}

# less-is-better defaults
def set_defaults_less_is_better():
    set_less_is_better("Mistakes Leading to Goal", True)
    set_less_is_better("Conceded/90", True)
    set_less_is_better("Red Cards", True)
    set_less_is_better("Yellow Cards", True)
    set_less_is_better("Offsides", True)
    for s in ["Shots Blocked/90","Blocks/90","Interceptions/90","Clearances/90"]:
        set_less_is_better(s, False)
set_defaults_less_is_better()

def apply_hard_remap(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in list(df.columns):
        df[c] = _series_for(df, c)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # rename to canonical
    df.columns = [RENAME_MAP.get(c, c) for c in df.columns]

    # type coercion
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = _maybe_numeric(df[c])

    # per90 synthesis via minutes
    min_name = find_col(df, ["Minutes","Mins","Min","Time Played"])
    denom = None
    if min_name:
        mins = pd.to_numeric(df[min_name], errors="coerce")
        denom = (mins / 90.0).replace(0, np.nan)

    if denom is not None:
        for src, tgt in CREATE_PER90_FROM_TOTAL.items():
            if src in df.columns and tgt not in df.columns:
                s = pd.to_numeric(df[src], errors="coerce")
                df[tgt] = (s / denom).round(2)
        if "Distance Covered (KM)" in df.columns and "Distance Covered (KM)/90" not in df.columns:
            s = pd.to_numeric(df["Distance Covered (KM)"], errors="coerce")
            df["Distance Covered (KM)/90"] = (s / denom).round(2)

    # xG/Shot
    xg_col = find_col(df, ["Expected Goals","xG"])
    shots_col = find_col(df, ["Shots"])
    if xg_col and shots_col:
        xg = pd.to_numeric(df[xg_col], errors="coerce")
        sh = pd.to_numeric(df[shots_col], errors="coerce").replace(0, np.nan)
        df["xG/Shot"] = (xg / sh).round(2)

    # round floats
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(2)
    return df

# =========================
# ===== POS TOKENISER =====
# =========================
def expand_positions(pos_str: str | float) -> List[str]:
    """'D (RLC), DM, M (C)' → ['D (R)','D (L)','D (C)','DM','M (C)'] — atomic only & unique."""
    if pd.isna(pos_str):
        return []
    s = str(pos_str)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    tokens: List[str] = []
    for p in parts:
        m = re.match(r"^([A-Z]{1,3})\s*\(([^)]+)\)$", p)
        if m:
            base = m.group(1).upper()
            ins = re.sub(r"[^A-Z]", "", m.group(2).upper())
            tokens += [f"{base} ({ch})" for ch in ins] if ins else [base]
        else:
            tokens.append(p.upper())
    return sorted(set(tokens))

# =========================
# ======= PLOTTING ========
# =========================
def get_contrast_text_color(hex_color):
    r, g, b = mcolors.hex2color(hex_color)
    brightness = (r*299 + g*587 + b*114) * 255 / 1000
    return "#000000" if brightness > 140 else "#F2F2F2"

def plotly_pizza(player_row: pd.Series, stat_cols: List[str], base_df: pd.DataFrame,
                 weights: Dict[str, float] | None = None, apply_weights_to_slices: bool = False,
                 sort_by_weight: bool = False) -> go.Figure:
    base = base_df if base_df is not None and not base_df.empty else _CURRENT_DF
    pts = compute_percentiles_for(player_row, stat_cols, base)
    stats = []; pcts = []; vals = []
    for s, p, v in pts:
        if not np.isnan(p):
            stats.append(s); pcts.append(float(p)); vals.append(v)
    if not stats:
        return go.Figure()

    if weights is None:
        weights = {s: 1.0 for s in stats}
    else:
        weights = {s: float(weights.get(s, 1.0)) for s in stats}

    order = list(range(len(stats)))
    if sort_by_weight:
        order = sorted(order, key=lambda i: weights[stats[i]], reverse=True)

    stats = [stats[i] for i in order]
    vals  = [vals[i]  for i in order]
    w_arr = np.array([weights[s] for s in stats], dtype=float)
    pcts  = [pcts[i] for i in order]

    if apply_weights_to_slices and len(w_arr) > 0:
        m = np.nanmax(w_arr) if np.isfinite(w_arr).any() else 1.0
        scale = (w_arr / m)
        pcts = (np.array(pcts, dtype=float) * scale).clip(0, 100).tolist()

    n = len(stats)
    thetas = np.linspace(0, 360, n, endpoint=False)
    width = 360 / max(n, 1) * 0.92
    slice_colors = (["#2E4374", "#1A78CF", "#D70232", "#FF9300", "#44C3A1",
                     "#CA228D", "#E1C340", "#7575A9", "#9DDFD3"] * 6)[:n]

    fig = go.Figure(go.Barpolar(
        r=pcts,
        theta=thetas,
        width=[width]*n,
        marker=dict(color=slice_colors, line=dict(color="#000", width=1)),
        customdata=np.array([stats, pcts, vals, w_arr], dtype=object).T,
        hovertemplate="<b>%{customdata[0]}</b><br>Percentile (shown): %{customdata[1]:.1f}"
                      "<br>Value: %{customdata[2]}<br>Weight: %{customdata[3]:.2f}<extra></extra>"
    ))
    fig.update_layout(
        template="simple_white",
        paper_bgcolor=POSTER_BG,
        plot_bgcolor=POSTER_BG,
        showlegend=False,
        polar=dict(
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                tickmode="array",
                tickvals=thetas.tolist(),
                ticktext=stats
            ),
            radialaxis=dict(
                tickmode="auto",
                ticks="outside"
            )
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    _plotly_polar_black(fig, rmax=100)
    return fig

# =========================
# ======= ARCHETYPES ======
# =========================
ARCHETYPES: Dict[str, List[str]] = {
    # Goalkeepers
    "GK — Shot Stopper": [
        "Save %", "Expected Save %", "Expected Goals Prevented/90",
        "Saves Held", "Saves Parried", "Saves Tipped",
        "Conceded/90", "Clean Sheets/90"
    ],
    "GK — Sweeper Keeper": [
        "Passes Attempted/90", "Passes Completed/90", "Pass Completion%",
        "Progressive Passes/90", "Expected Goals Prevented/90", "Save %",
        "Saves Held", "Saves Parried"
    ],

    # Centre-backs
    "CB — Stopper": [
        "Tackles/90", "Tackles Won/90", "Tackle Ratio",
        "Interceptions/90", "Blocks/90", "Shots Blocked/90",
        "Clearances/90", "Headers won/90", "Header Win Rate"
    ],
    "CB — Ball Playing": [
        "Passes Attempted/90", "Passes Completed/90", "Pass Completion%",
        "Progressive Passes/90", "Key Passes/90", "Interceptions/90",
        "Tackles/90", "Chances Created/90"
    ],

    # Full-backs / Wing-backs
    "FB — Overlapping": [
        "Crosses Attempted/90", "Crosses Completed/90", "Cross Completion Ratio",
        "Open Play Crosses Attempted/90", "Open Play Crosses Completed/90",
        "Open Play Key Passes/90", "Key Passes/90", "Dribbles/90",
        "Tackles/90", "Interceptions/90"
    ],
    "FB — Inverted": [
        "Passes Attempted/90", "Passes Completed/90", "Pass Completion%",
        "Progressive Passes/90", "Open Play Key Passes/90",
        "Interceptions/90", "Tackles/90", "Chances Created/90"
    ],

    # Defensive Midfield
    "DM — Ball Winner": [
        "Tackles/90", "Tackles Won/90", "Tackle Ratio",
        "Interceptions/90", "Blocks/90", "Shots Blocked/90",
        "Possession Won/90", "Pressures Completed/90"
    ],
    "DM — Deep-Lying Playmaker": [
        "Passes Attempted/90", "Passes Completed/90", "Pass Completion%",
        "Progressive Passes/90", "Open Play Key Passes/90",
        "Key Passes/90", "Interceptions/90"
    ],

    # Central Midfield
    "CM — Box to Box": [
        "Progressive Passes/90", "Open Play Key Passes/90",
        "Dribbles/90", "Pressures Completed/90",
        "Tackles/90", "Interceptions/90",
        "Shots/90", "SoT/90"
    ],
    "CM — Progresser": [
        "Progressive Passes/90", "Passes Completed/90", "Passes Attempted/90",
        "Open Play Key Passes/90", "Key Passes/90", "Dribbles/90",
        "Chances Created/90"
    ],

    # Attacking Midfield
    "AM — Classic 10": [
        "Open Play Key Passes/90", "Key Passes/90", "Chances Created/90",
        "Assists/90", "Progressive Passes/90", "Dribbles/90",
        "Shots/90"
    ],
    "AM — Shadow Striker": [
        "Shots/90", "SoT/90", "Dribbles/90",
        "Chances Created/90", "Key Passes/90",
        "Conversion Rate", "Goals / 90"
    ],

    # Wingers
    "Winger — Classic": [
        "Crosses Attempted/90", "Crosses Completed/90", "Cross Completion Ratio",
        "Open Play Crosses Attempted/90", "Open Play Crosses Completed/90",
        "Open Play Key Passes/90", "Dribbles/90", "Assists/90"
    ],
    "Winger — Inverted": [
        "Shots/90", "SoT/90", "Dribbles/90",
        "Open Play Key Passes/90", "Chances Created/90",
        "Conversion Rate", "Progressive Passes/90"
    ],

    # Strikers
    "ST — Poacher": [
        "Shots/90", "SoT/90", "Conversion Rate",
        "Goals / 90", "xG/90", "xG/Shot"
    ],
    "ST — Target Man": [
        "Headers won/90", "Header Win Rate", "Aerial Duels Attempted/90",
        "Shots/90", "SoT/90", "Key Passes/90"
    ],
}

# ---- Default weights per archetype (examples filled) ----
DEFAULT_ARCHETYPE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "GK — Shot Stopper": {
        "Save %": 1.90, "Expected Save %": 1.60, "Expected Goals Prevented/90": 1.80,
        "Saves Held": 1.40, "Saves Parried": 1.10, "Saves Tipped": 1.00,
        "Conceded/90": 1.40, "Clean Sheets/90": 1.20,
    },
    "GK — Sweeper Keeper": {
        "Passes Attempted/90": 1.60, "Passes Completed/90": 1.50, "Pass Completion%": 1.40,
        "Progressive Passes/90": 1.80, "Expected Goals Prevented/90": 1.10, "Save %": 1.20,
        "Saves Held": 1.00, "Saves Parried": 1.00,
    },
    "CB — Stopper": {
        "Tackles/90": 1.70, "Tackles Won/90": 1.90, "Tackle Ratio": 1.30,
        "Interceptions/90": 1.70, "Blocks/90": 1.50, "Shots Blocked/90": 1.50,
        "Clearances/90": 1.60, "Headers won/90": 1.40, "Header Win Rate": 1.40,
    },
    "CB — Ball Playing": {
        "Passes Attempted/90": 1.60, "Passes Completed/90": 1.50, "Pass Completion%": 1.40,
        "Progressive Passes/90": 1.80, "Key Passes/90": 1.30, "Interceptions/90": 1.00,
        "Tackles/90": 0.90, "Chances Created/90": 1.20,
    },
    "FB — Overlapping": {
        "Crosses Attempted/90": 1.60, "Crosses Completed/90": 1.70, "Cross Completion Ratio": 1.40,
        "Open Play Crosses Attempted/90": 1.70, "Open Play Crosses Completed/90": 1.80,
        "Open Play Key Passes/90": 1.50, "Key Passes/90": 1.30, "Dribbles/90": 1.40,
        "Tackles/90": 1.10, "Interceptions/90": 1.10,
    },
    "FB — Inverted": {
        "Passes Attempted/90": 1.50, "Passes Completed/90": 1.50, "Pass Completion%": 1.60,
        "Progressive Passes/90": 1.80, "Open Play Key Passes/90": 1.60, "Interceptions/90": 1.20,
        "Tackles/90": 1.10, "Chances Created/90": 1.20,
    },
    "DM — Ball Winner": {
        "Tackles/90": 1.70, "Tackles Won/90": 1.80, "Tackle Ratio": 1.40,
        "Interceptions/90": 1.70, "Blocks/90": 1.30, "Shots Blocked/90": 1.30,
        "Possession Won/90": 1.60, "Pressures Completed/90": 1.20,
    },
    "DM — Deep-Lying Playmaker": {
        "Passes Attempted/90": 1.70, "Passes Completed/90": 1.50, "Pass Completion%": 1.40,
        "Progressive Passes/90": 1.80, "Open Play Key Passes/90": 1.40,
        "Key Passes/90": 1.20, "Interceptions/90": 1.00,
    },
    "CM — Box to Box": {
        "Progressive Passes/90": 1.50, "Open Play Key Passes/90": 1.30, "Dribbles/90": 1.30,
        "Pressures Completed/90": 1.20, "Tackles/90": 1.30, "Interceptions/90": 1.30,
        "Shots/90": 1.20, "SoT/90": 1.20,
    },
    "CM — Progresser": {
        "Progressive Passes/90": 1.80, "Passes Completed/90": 1.40, "Passes Attempted/90": 1.50,
        "Open Play Key Passes/90": 1.60, "Key Passes/90": 1.40, "Dribbles/90": 1.30,
        "Chances Created/90": 1.60,
    },
    "AM — Classic 10": {
        "Open Play Key Passes/90": 1.80, "Key Passes/90": 1.70, "Chances Created/90": 1.80,
        "Assists/90": 1.60, "Progressive Passes/90": 1.40, "Dribbles/90": 1.30,
        "Shots/90": 1.10,
    },
    "AM — Shadow Striker": {
        "Shots/90": 1.60, "SoT/90": 1.60, "Dribbles/90": 1.20,
        "Chances Created/90": 1.20, "Key Passes/90": 1.20,
        "Conversion Rate": 1.70, "Goals / 90": 1.90,
    },
    "Winger — Classic": {
        "Crosses Attempted/90": 1.50, "Crosses Completed/90": 1.70, "Cross Completion Ratio": 1.40,
        "Open Play Crosses Attempted/90": 1.60, "Open Play Crosses Completed/90": 1.80,
        "Open Play Key Passes/90": 1.50, "Dribbles/90": 1.60, "Assists/90": 1.50,
    },
    "Winger — Inverted": {
        "Shots/90": 1.60, "SoT/90": 1.60, "Dribbles/90": 1.70,
        "Open Play Key Passes/90": 1.40, "Chances Created/90": 1.50,
        "Conversion Rate": 1.60, "Progressive Passes/90": 1.30,
    },
    "ST — Poacher": {
        "Shots/90": 1.60, "SoT/90": 1.70, "Conversion Rate": 1.70,
        "Goals / 90": 1.90, "xG/90": 1.70, "xG/Shot": 1.40,
    },
    "ST — Target Man": {
        "Headers won/90": 1.80, "Header Win Rate": 1.70, "Aerial Duels Attempted/90": 1.60,
        "Shots/90": 1.30, "SoT/90": 1.20, "Key Passes/90": 1.10,
    },
}

# =========================
# ======= APP STATE =======
# =========================
if "custom_metrics" not in st.session_state:
    st.session_state["custom_metrics"] = {}
if "custom_arches" not in st.session_state:
    st.session_state["custom_arches"] = {}
if "arch_weights" not in st.session_state:
    st.session_state["arch_weights"] = {}

# =========================
# ========= UI ============
# =========================
st.title("FM Analytics")

st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader("Upload FM HTML export (or CSV)", type=["html","htm","csv"])

# convenience downloads
st.sidebar.header("Extras")
for fname, label in [("leagues filter.fmf", "Download ‘leagues filter.fmf’"),
                     ("player search.fmf", "Download ‘player search.fmf’")]:
    fpath = os.path.join(APP_DIR, fname)
    if os.path.isfile(fpath):
        with open(fpath, "rb") as fh:
            st.sidebar.download_button(label, data=fh.read(), file_name=fname, mime="application/octet-stream",
                                       key=f"dl_{re.sub(r'\\W+','_',fname)}")
    else:
        st.sidebar.caption(f"*{fname} not found in working dir.*")

@st.cache_data(show_spinner=True)
def parse_and_cache(name: str, raw: bytes) -> Tuple[pd.DataFrame, str]:
    if name.lower().endswith((".html",".htm")):
        df = read_fm_html(io.BytesIO(raw))
    else:
        df = pd.read_csv(io.BytesIO(raw))
        df = _sanitize_df(df)
    df = apply_hard_remap(df)

    # position tokens
    pos_col = find_col(df, ["Pos","Position"])
    if pos_col is None:
        df["__pos_tokens"] = [[] for _ in range(len(df))]
    else:
        df["__pos_tokens"] = df[pos_col].apply(expand_positions)

    # tidy dtypes
    for c in df.select_dtypes(include="float").columns:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float").round(2)
    for c in df.select_dtypes(include="integer").columns:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")

    cache_id = f"{hash((name, df.shape, tuple(df.columns)))}"
    return df, cache_id

df = pd.DataFrame(); cache_id = ""
if uploaded is not None:
    df, cache_id = parse_and_cache(uploaded.name, uploaded.read())

if df.empty:
    st.info("Upload an FM export to begin.")
    st.stop()

# ======== Sidebar: Minimum Minutes ========
if "Minutes" in df.columns:
    max_min = int(np.nanmax(pd.to_numeric(df["Minutes"], errors="coerce").fillna(0)))
    min_minutes = st.sidebar.slider("Minimum Minutes", 0, max_min, 0, step=90,
                                    help="Filter players everywhere by played minutes.")
    df_work = df[pd.to_numeric(df["Minutes"], errors="coerce").fillna(0) >= min_minutes].copy()
else:
    st.sidebar.caption("*No Minutes column found — minutes filter disabled.*")
    min_minutes = 0
    df_work = df.copy()

_CURRENT_DF = df_work

# ==== baseline positions (atomic unique only) ====
all_tokens = sorted(set(
    t for lst in df_work["__pos_tokens"] for t in lst
    if isinstance(t, str) and "," not in t
))
name_col = find_col(df_work, ["Name"]) or "Name"
pos_col  = find_col(df_work, ["Pos","Position"]) or "Pos"
player_list = sorted(df_work[name_col].dropna().astype(str).unique().tolist())
player = st.sidebar.selectbox("Player", player_list)

player_row = df_work[df_work[name_col] == player].iloc[0]
player_tokens = player_row["__pos_tokens"] if "__pos_tokens" in player_row else []

st.sidebar.header("Percentile Baseline")
baseline_tokens = st.sidebar.multiselect(
    "Positions to compare against",
    options=all_tokens,
    default=(player_tokens if player_tokens else all_tokens),
    help="Percentiles are computed only vs these positions. Tokens are exploded atomic positions."
)

def filter_by_tokens(df_: pd.DataFrame, tokens: List[str]) -> pd.DataFrame:
    if not tokens:
        return df_
    tokset = set(tokens)
    mask = df_["__pos_tokens"].apply(lambda lst: bool(tokset.intersection(lst)))
    out = df_.loc[mask]
    return out if not out.empty else df_

BASELINE_DF = filter_by_tokens(df_work, baseline_tokens)

st.sidebar.write("---")
mode = st.sidebar.radio(
    "Mode",
    ["Pizza", "Archetypes", "Percentile Bars", "Distribution", "Stat Scatter", "Role Scatter",
     "Top 10 — Roles", "Top 10 — Stats", "Player Finder", "PCA Map", "Build: Metrics & Archetypes", "Table"],
    index=0
)

numeric_cols = [c for c in df_work.columns if pd.api.types.is_numeric_dtype(df_work[c])]
display_cols = [c for c in df_work.columns if c not in {name_col, "Club", "League", "Pos", "__pos_tokens"}]

# =========================
# ====== ROLE SCORES ======
# =========================
def all_archetypes_dict() -> Dict[str, List[str]]:
    arch_all = dict(ARCHETYPES)
    arch_all.update({f"(Custom) {k}": v.get("stats", []) for k, v in st.session_state["custom_arches"].items()})
    for k in list(arch_all.keys()):
        arch_all[k] = [s for s in arch_all[k] if s in df_work.columns and pd.api.types.is_numeric_dtype(df_work[s])]
        if not arch_all[k]:
            del arch_all[k]
    return arch_all

def default_weights_for(arch_name: str, stats: List[str]) -> Dict[str, float]:
    base = DEFAULT_ARCHETYPE_WEIGHTS.get(arch_name, {})
    return {s: float(base.get(s, 1.0)) for s in stats}

def get_arch_weights(arch_name: str, stats: List[str]) -> Dict[str, float]:
    saved = st.session_state["arch_weights"].get(arch_name, {})
    if not saved:
        saved = {}
    w = default_weights_for(arch_name, stats)
    w.update({s: float(saved.get(s, w.get(s, 1.0))) for s in stats})
    w = {s: float(w.get(s, 1.0)) for s in stats}
    return w

def set_arch_weights(arch_name: str, new_weights: Dict[str, float]) -> None:
    st.session_state["arch_weights"][arch_name] = {k: float(v) for k, v in new_weights.items()}

def role_scores_for_archetype(arch_name: str, base_df: pd.DataFrame) -> pd.Series:
    """
    Role Score (0–100) = weighted composite of per-stat percentiles (0–100).
    Steps:
      1) For each stat in the archetype, compute percentile vs baseline (respect less-is-better).
      2) Take a weighted average per player (weights renormalised for missing stats).
    """
    arch_all = all_archetypes_dict()
    stats = arch_all.get(arch_name, [])
    if not stats:
        return pd.Series([np.nan]*len(df_work), index=df_work.index)

    weights = get_arch_weights(arch_name, stats)
    w = np.array([float(weights.get(s, 1.0)) for s in stats], dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)

    pct_mat = []
    for s in stats:
        pct_s = column_percentiles(df_work[s], base_df[s], is_less_better(s))
        pct_mat.append(pct_s.to_numpy())
    P = np.vstack(pct_mat).T  # [n_players, n_stats], NaNs allowed

    valid = np.isfinite(P).astype(float)
    w_row = valid * w
    denom = w_row.sum(axis=1)
    numer = np.nansum(P * w, axis=1)
    composite = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)  # 0–100
    return pd.Series(composite, index=df_work.index).clip(0, 100)

# =========================
# ========= MODES =========
# =========================
def unique_key(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"

# ---- Pizza ----
if mode == "Pizza":
    st.subheader("Custom Pizza")
    default_stats = [x for x in ["Shots/90","SoT/90","Expected Goals","Expected Assists/90","Dribbles/90","Open Play Key Passes/90"] if x in numeric_cols]
    stats_pick = st.multiselect("Choose stats (ordered)", options=numeric_cols, default=default_stats)
    fig = plotly_pizza(player_row, stats_pick, BASELINE_DF)
    fig.update_layout(title=f"{player} — Custom Pizza")
    _plotly_polar_black(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)

# ---- Archetypes ----
elif mode == "Archetypes":
    st.subheader("Archetype Pizza (with per-stat weights)")
    arch_all = all_archetypes_dict()
    if not arch_all:
        st.warning("No archetypes with valid stats available.")
    else:
        arch_name = st.selectbox("Archetype", list(arch_all.keys()))
        arch_stats = arch_all.get(arch_name, [])
        if len(arch_stats) < 1:
            st.warning("No valid stats for this archetype in the current dataset.")
        else:
            with st.expander("Weights (per stat for this archetype)", expanded=True):
                w = get_arch_weights(arch_name, arch_stats)
                cols = st.columns(min(4, len(arch_stats)))
                new_w = {}
                for i, s in enumerate(arch_stats):
                    with cols[i % len(cols)]:
                        new_w[s] = st.number_input(f"{s}", min_value=0.0, value=float(w.get(s, 1.0)), step=0.1, key=f"w_{arch_name}_{s}")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    if st.button("Save weights", key=f"savew_{arch_name}"):
                        set_arch_weights(arch_name, new_w); st.success("Saved weights.")
                with c2:
                    if st.button("Equal weights", key=f"eqw_{arch_name}"):
                        set_arch_weights(arch_name, {s:1.0 for s in arch_stats}); st.success("Set equal weights.")
                with c3:
                    if st.button("Reset to defaults", key=f"defaultw_{arch_name}"):
                        set_arch_weights(arch_name, default_weights_for(arch_name, arch_stats)); st.success("Reset to defaults.")
                with c4:
                    if st.button("Export weights (JSON)", key=f"expw_{arch_name}"):
                        payload = {"archetype": arch_name, "weights": get_arch_weights(arch_name, arch_stats)}
                        st.download_button("Download", data=json.dumps(payload, indent=2).encode("utf-8"),
                                           file_name=f"weights_{re.sub(r'\\W+','_',arch_name)}.json",
                                           mime="application/json", key=unique_key("dl_w_json"))
            with st.expander("Import weights JSON", expanded=False):
                up = st.file_uploader("Upload JSON", type=["json"], key=f"impw_{arch_name}")
                if up is not None:
                    try:
                        data = json.loads(up.read().decode("utf-8"))
                        if isinstance(data, dict) and "weights" in data:
                            w_in = {k: float(v) for k, v in data["weights"].items()}
                            w_in = {s: w_in.get(s, 1.0) for s in arch_stats}
                            set_arch_weights(arch_name, w_in); st.success("Imported weights.")
                        else:
                            st.error("JSON missing 'weights' field.")
                    except Exception as e:
                        st.error(f"Import failed: {e}")

            apply_weights = st.checkbox("Apply weights to pizza slices (visual scaling)", value=False)
            sort_by_w    = st.checkbox("Sort slices by weight (desc)", value=False)

            fig = plotly_pizza(player_row, arch_stats, BASELINE_DF,
                               weights=get_arch_weights(arch_name, arch_stats),
                               apply_weights_to_slices=apply_weights,
                               sort_by_weight=sort_by_w)
            fig.update_layout(title=f"{player} — {arch_name}")
            _plotly_polar_black(fig)
            st.plotly_chart(fig, use_container_width=True, theme=None)

# ---- Percentile Bars ----
elif mode == "Percentile Bars":
    st.subheader("Percentile Ladder")
    default_stats = [x for x in ["Shots/90","SoT/90","Expected Goals","Expected Assists/90","Dribbles/90","Open Play Key Passes/90"] if x in numeric_cols]
    stats_pick = st.multiselect("Choose stats", options=numeric_cols, default=default_stats, key="ladder_stats")
    rows = compute_percentiles_for(player_row, stats_pick, BASELINE_DF)
    rows = [r for r in rows if not np.isnan(r[1])]
    if not rows:
        st.warning("Pick at least one valid numeric stat.")
    else:
        rows.sort(key=lambda x: x[1], reverse=True)
        labels = [r[0] for r in rows]; pcts = [r[1] for r in rows]; vals = [r[2] for r in rows]
        df_plot = pd.DataFrame({"stat": labels, "percentile": pcts, "value": vals})
        fig = go.Figure(go.Bar(
            x=df_plot["percentile"], y=df_plot["stat"], orientation="h",
            text=[f"{p:.0f}%" for p in df_plot["percentile"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Percentile: %{x:.1f}<br>Value: %{customdata}<extra></extra>",
            customdata=df_plot["value"]
        ))
        fig.update_layout(
            template="simple_white",
            xaxis=dict(range=[0,100], title="Percentile"),
            yaxis=dict(autorange="reversed"),
            shapes=[dict(type="line", x0=50, x1=50, y0=-0.5, y1=len(labels)-0.5, line=dict(width=1, dash="dot", color="gray"))],
            title=f"{player} — Percentile Ladder",
            paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
            font=dict(family=FONT_FAMILY)
        ); _plotly_axes_black(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

# ---- Distribution ----
elif mode == "Distribution":
    st.subheader("Distribution Explorer")
    stat = st.selectbox("Stat", options=numeric_cols, index=0, key="dist_stat")
    series = pd.to_numeric(BASELINE_DF[stat], errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
    val = pd.to_numeric(player_row.get(stat), errors="coerce")
    if series.empty:
        st.warning("No data for this stat in the baseline.")
    else:
        df_hist = pd.DataFrame({stat: series})
        fig = px.histogram(df_hist, x=stat, nbins=25, opacity=0.9)
        if pd.notna(val):
            fig.add_shape(type="line", x0=float(val), x1=float(val), y0=0, y1=1, xref="x", yref="paper",
                          line=dict(color="#D70232", width=2))
            fig.add_annotation(x=float(val), y=1, yref="paper", showarrow=False, xanchor="right",
                               text=f"{player}: {float(val):.2f}", font=dict(family=FONT_FAMILY, color="#D70232"))
        fig.update_layout(template="simple_white", title=f"{stat} — Baseline Distribution",
                          xaxis_title=stat, yaxis_title="Count",
                          paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
                          font=dict(family=FONT_FAMILY)); _plotly_axes_black(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Stat Scatter ----------
elif mode == "Stat Scatter":
    st.subheader("Stat Scatter (X vs Y)")
    colx, coly = st.columns(2)
    with colx: xcol = st.selectbox("X", options=numeric_cols, index=0, key="scat_x")
    with coly: ycol = st.selectbox("Y", options=numeric_cols, index=1, key="scat_y")

    show_cloud = st.checkbox("Show all points", value=False)
    age_ok = "Age" in BASELINE_DF.columns
    u23 = st.checkbox("Highlight U23", value=False, disabled=not age_ok)
    u21 = st.checkbox("Highlight U21", value=False, disabled=not age_ok)
    hi_name = st.selectbox("Highlight player (type to search)", options=["(none)"] + sorted(BASELINE_DF[name_col].astype(str).unique().tolist()), index=0)

    X = pd.to_numeric(BASELINE_DF[xcol], errors="coerce")
    Y = pd.to_numeric(BASELINE_DF[ycol], errors="coerce")
    mask_xy = X.notna() & Y.notna()
    cols_to_pull = [name_col, xcol, ycol]
    if "Club" in BASELINE_DF.columns: cols_to_pull.append("Club")
    if pos_col in BASELINE_DF.columns: cols_to_pull.append(pos_col)
    df_sc = BASELINE_DF.loc[mask_xy, cols_to_pull].copy()
    df_sc.rename(columns={pos_col: "Pos"}, inplace=True, errors="ignore")
    for c in ["Club","Pos"]:
        if c not in df_sc.columns: df_sc[c] = ""

    fig = go.Figure()

    if show_cloud:
        sc = px.scatter(df_sc, x=xcol, y=ycol, opacity=0.35, hover_name=name_col,
                        hover_data={name_col:False, "Club":True, "Pos":True, xcol:":.2f", ycol:":.2f"})
        for tr in sc.data: fig.add_trace(tr)

    if age_ok:
        ages = pd.to_numeric(BASELINE_DF["Age"], errors="coerce")
        if u23:
            cohort = df_sc[df_sc[name_col].isin(BASELINE_DF.loc[ages < 23, name_col])]
            fig.add_trace(go.Scatter(
                x=cohort[xcol], y=cohort[ycol], mode="markers",
                marker=dict(size=10, color="rgba(26,120,207,0.85)", line=dict(width=1, color="black")),
                name="U23", hovertext=cohort[name_col], hoverinfo="text"
            ))
        if u21:
            cohort = df_sc[df_sc[name_col].isin(BASELINE_DF.loc[ages < 21, name_col])]
            fig.add_trace(go.Scatter(
                x=cohort[xcol], y=cohort[ycol], mode="markers",
                marker=dict(size=10, color="rgba(199, 43, 98, 0.85)", line=dict(width=1, color="black")),
                name="U21", hovertext=cohort[name_col], hoverinfo="text"
            ))

    if hi_name != "(none)":
        prow = df_sc[df_sc[name_col] == hi_name]
        if not prow.empty:
            fig.add_trace(go.Scatter(
                x=prow[xcol], y=prow[ycol], mode="markers+text",
                marker=dict(size=18, color="#D70232", line=dict(width=1, color="black")),
                text=[hi_name], textposition="middle right",
                hoverinfo="skip", showlegend=False, name="Highlight"
            ))

    fig.update_layout(template="simple_white", title=f"{ycol} vs {xcol}",
                      xaxis_title=xcol, yaxis_title=ycol,
                      paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
                      font=dict(family=FONT_FAMILY))
    _plotly_axes_black(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Role Scatter ----------
elif mode == "Role Scatter":
    st.subheader("Role Scatter (weighted composite of percentiles, 0–100)")
    arch_all = all_archetypes_dict()
    if not arch_all:
        st.warning("No archetypes with valid stats available.")
    else:
        colx, coly = st.columns(2)
        with colx: ax = st.selectbox("X archetype", list(arch_all.keys()), key="role_x")
        with coly: ay = st.selectbox("Y archetype", list(arch_all.keys()), key="role_y", index=min(1,len(arch_all)-1))

        show_cloud = st.checkbox("Show all points", value=False, key="role_cloud")
        age_ok = "Age" in BASELINE_DF.columns
        u23 = st.checkbox("Highlight U23", value=False, disabled=not age_ok, key="role_u23")
        u21 = st.checkbox("Highlight U21", value=False, disabled=not age_ok, key="role_u21")
        hi_name = st.selectbox("Highlight player (type to search)",
                               options=["(none)"] + sorted(BASELINE_DF[name_col].astype(str).unique().tolist()),
                               index=0, key="role_hi")

        # Compute scores across full df_work, then restrict to BASELINE_DF players
        score_x = role_scores_for_archetype(ax, BASELINE_DF).reindex(BASELINE_DF.index)
        score_y = role_scores_for_archetype(ay, BASELINE_DF).reindex(BASELINE_DF.index)

        df_rs = BASELINE_DF[[name_col]].copy()
        if "Club" in BASELINE_DF.columns: df_rs["Club"] = BASELINE_DF["Club"]
        if pos_col in BASELINE_DF.columns: df_rs["Pos"] = BASELINE_DF[pos_col]
        df_rs[ax] = score_x; df_rs[ay] = score_y
        df_rs = df_rs[df_rs[ax].notna() & df_rs[ay].notna()]

        fig = go.Figure()

        if show_cloud:
            sc = px.scatter(df_rs, x=ax, y=ay, opacity=0.35, hover_name=name_col,
                            hover_data={name_col:False,"Club":True,"Pos":True,ax:":.1f",ay:":.1f"})
            for tr in sc.data: fig.add_trace(tr)

        if age_ok:
            ages = pd.to_numeric(BASELINE_DF["Age"], errors="coerce")
            if u23:
                cohort = df_rs[df_rs[name_col].isin(BASELINE_DF.loc[ages < 23, name_col])]
                fig.add_trace(go.Scatter(x=cohort[ax], y=cohort[ay], mode="markers",
                                         marker=dict(size=10, color="rgba(26,120,207,0.85)", line=dict(width=1, color="black")),
                                         name="U23", hovertext=cohort[name_col], hoverinfo="text"))
            if u21:
                cohort = df_rs[df_rs[name_col].isin(BASELINE_DF.loc[ages < 21, name_col])]
                fig.add_trace(go.Scatter(x=cohort[ax], y=cohort[ay], mode="markers",
                                         marker=dict(size=10, color="rgba(199, 43, 98, 0.85)", line=dict(width=1, color="black")),
                                         name="U21", hovertext=cohort[name_col], hoverinfo="text"))

        if hi_name != "(none)":
            prow = df_rs[df_rs[name_col] == hi_name]
            if not prow.empty:
                fig.add_trace(go.Scatter(
                    x=prow[ax], y=prow[ay], mode="markers+text",
                    marker=dict(size=18, color="#D70232", line=dict(width=1, color="black")),
                    text=[hi_name], textposition="middle right",
                    hoverinfo="skip", showlegend=False, name="Highlight"
                ))

        fig.update_layout(template="simple_white",
                          title=f"Role Scores — {ay} vs {ax}",
                          xaxis_title=ax, yaxis_title=ay,
                          xaxis=dict(range=[0,100]), yaxis=dict(range=[0,100]),
                          paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
                          font=dict(family=FONT_FAMILY))
        _plotly_axes_black(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Top 10 — Roles ----------
elif mode == "Top 10 — Roles":
    st.subheader("Top 10 by Role Score (weighted composite, 0–100)")
    arch_all = all_archetypes_dict()
    if not arch_all:
        st.info("No archetypes with valid stats available.")
    else:
        role = st.selectbox("Role", list(arch_all.keys()))
        scores = role_scores_for_archetype(role, BASELINE_DF).round(1)
        tbl = df_work[[name_col]].copy()
        if "Club" in df_work.columns: tbl["Club"] = df_work["Club"]
        if pos_col in df_work.columns: tbl["Pos"] = df_work[pos_col]
        tbl["Score"] = scores
        top = tbl.dropna(subset=["Score"]).sort_values("Score", ascending=False).head(10)

        fig = go.Figure(go.Bar(
            x=top["Score"], y=top[name_col], orientation="h",
            hovertemplate="<b>%{y}</b><br>Role Score: %{x:.1f}<br>Club: %{customdata[0]}<extra></extra>",
            customdata=np.stack([top.get("Club", pd.Series([""]*len(top)))], axis=1)
        ))
        fig.update_layout(
            title=f"Top 10 — {role}",
            xaxis=dict(title="Role Score (0–100)", range=[0, 100]),
            yaxis=dict(autorange="reversed"),
            template="simple_white",
            paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
            font=dict(family=FONT_FAMILY)
        )
        _plotly_axes_black(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Top 10 — Stats ----------
elif mode == "Top 10 — Stats":
    st.subheader("Top 10 by Stat")
    stat = st.selectbox("Stat", options=numeric_cols)
    rank_type = st.radio("Rank by", ["Percentile (vs baseline)", "Raw Value"], horizontal=True)

    tbl = df_work[[name_col]].copy()
    if "Club" in df_work.columns: tbl["Club"] = df_work["Club"]
    if pos_col in df_work.columns: tbl["Pos"] = df_work[pos_col]

    if rank_type.startswith("Percentile"):
        series = column_percentiles(df_work[stat], BASELINE_DF[stat], is_less_better(stat)).round(1)
        tbl["Value"] = series
        top = tbl.dropna(subset=["Value"]).sort_values("Value", ascending=False).head(10)
        x_title = "Percentile (0–100)"; x_range = [0, 100]
        hover = "<b>%{y}</b><br>Percentile: %{x:.1f}<br>Club: %{customdata[0]}<extra></extra>"
    else:
        series = pd.to_numeric(df_work[stat], errors="coerce")
        tbl["Value"] = series
        asc = is_less_better(stat)
        top = tbl.dropna(subset=["Value"]).sort_values("Value", ascending=asc).head(10)
        x_title = f"{stat} ({'lower is better' if asc else 'higher is better'})"
        x_range = None
        hover = "<b>%{y}</b><br>Value: %{x}<br>Club: %{customdata[0]}<extra></extra>"

    fig = go.Figure(go.Bar(
        x=top["Value"], y=top[name_col], orientation="h",
        hovertemplate=hover,
        customdata=np.stack([top.get("Club", pd.Series([""]*len(top)))], axis=1)
    ))
    fig.update_layout(
        title=f"Top 10 — {stat}",
        xaxis=dict(title=x_title, range=x_range),
        yaxis=dict(autorange="reversed"),
        template="simple_white",
        paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
        font=dict(family=FONT_FAMILY)
    )
    _plotly_axes_black(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Player Finder ----------
elif mode == "Player Finder":
    st.subheader("Player Finder (values or percentiles)")
    mode_type = st.radio("Use", ["Values", "Percentiles"], horizontal=True)
    st.caption("Percentiles are computed vs your selected baseline positions above.")

    max_rows = 6
    ncrit = st.slider("Number of criteria", 1, max_rows, 3)
    crits = []
    for i in range(ncrit):
        c1, c2, c3 = st.columns([4,2,4])
        with c1: stat = st.selectbox(f"Stat {i+1}", options=numeric_cols, key=f"pf_stat_{i}")
        with c2: op = st.selectbox(f"Comp {i+1}", options=[">=", "<="], key=f"pf_op_{i}")
        with c3:
            if mode_type == "Values":
                thr = st.number_input(f"Threshold {i+1}", value=0.0, step=0.1, key=f"pf_thr_{i}")
            else:
                thr = st.number_input(f"Percentile {i+1}", value=70.0, min_value=0.0, max_value=100.0, step=1.0, key=f"pf_thr_{i}")
        crits.append((stat, op, float(thr)))

    if st.button("Apply filters", type="primary"):
        out = df_work.copy()
        mask_all = pd.Series(True, index=out.index)

        for (stat, op, thr) in crits:
            if mode_type == "Values":
                series = pd.to_numeric(out[stat], errors="coerce")
                mask = series >= thr if op == ">=" else series <= thr
                mask = mask.fillna(False)
            else:
                pct_col = column_percentiles(out[stat], BASELINE_DF[stat], is_less_better(stat))
                mask = pct_col >= thr if op == ">=" else pct_col <= thr
                mask = mask.fillna(False)
                out[f"{stat} (pct)"] = pct_col.round(1)
            mask_all &= mask

        res = out.loc[mask_all].copy()
        keep_cols = [name_col]
        for c in ["Club", pos_col, "Minutes", "Avg Rating"]:
            if c in res.columns and c not in keep_cols:
                keep_cols.append(c)
        for (stat, _, _) in crits:
            if stat not in keep_cols: keep_cols.append(stat)
            pct_name = f"{stat} (pct)"
            if pct_name in res.columns and pct_name not in keep_cols: keep_cols.append(pct_name)

        res = res[keep_cols].sort_values(by=[keep_cols[0]])
        st.success(f"Found {len(res):,} players matching.")
        st.dataframe(res, use_container_width=True)

        csv = res.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="player_finder_results.csv", mime="text/csv", key=unique_key("dl_pf"))

# ---------- PCA Map ----------
elif mode == "PCA Map":
    st.subheader("PCA Role Map")
    default_stats = [x for x in ["Passes Attempted/90","Pass Completion%","Progressive Passes/90","Key Passes/90",
                                 "Dribbles/90","Shots/90","SoT/90","Interceptions/90","Tackles/90"] if x in numeric_cols]
    stats_pick = st.multiselect("Metrics to embed", options=numeric_cols, default=default_stats, key="pca_stats")

    if len(stats_pick) < 2:
        st.info("Pick two or more metrics.")
    else:
        X = BASELINE_DF[stats_pick].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
        mask = X.notna().all(axis=1)
        X = X.loc[mask]
        names = BASELINE_DF.loc[mask, name_col].astype(str).reset_index(drop=True)
        clubs = BASELINE_DF.loc[mask, "Club"].astype(str).reset_index(drop=True) if "Club" in BASELINE_DF.columns else pd.Series([""]*len(names))
        poss  = BASELINE_DF.loc[mask, pos_col].astype(str).reset_index(drop=True) if pos_col in BASELINE_DF.columns else pd.Series([""]*len(names))

        if X.shape[0] < 3:
            st.warning("Not enough complete rows for PCA.")
        else:
            mu = X.mean(axis=0); sd = X.std(axis=0).replace(0, 1)
            Z = (X - mu) / sd
            U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
            PC = U[:, :2] * S[:2]
            pca_df = pd.DataFrame({"PC1": PC[:,0], "PC2": PC[:,1], name_col: names, "Club": clubs, "Pos": poss})

            fig = px.scatter(pca_df, x="PC1", y="PC2", hover_name=name_col,
                             hover_data={name_col:False, "Club":True, "Pos":True, "PC1":":.2f", "PC2":":.2f"},
                             opacity=0.6)
            prow = pca_df[pca_df[name_col] == player]
            if not prow.empty:
                fig.add_trace(go.Scatter(x=prow["PC1"], y=prow["PC2"], mode="markers+text",
                                         marker=dict(size=18, color="#D70232", line=dict(width=1, color="black")),
                                         text=[player], textposition="middle right",
                                         hoverinfo="skip", showlegend=False))
            fig.update_layout(template="simple_white", title="PCA Role Map",
                              paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
                              font=dict(family=FONT_FAMILY))
            _plotly_axes_black(fig)
            st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Build: Metrics & Archetypes ----------
elif mode == "Build: Metrics & Archetypes":
    st.subheader("Create metrics")
    with st.expander("Add a custom metric", expanded=True):
        colA, colOp, colB = st.columns([3,1,3])
        with colA: a = st.selectbox("A", options=[None]+display_cols, index=0, key="cm_a")
        with colOp: op = st.selectbox("Op", options=["+","-","*","/"], index=2, key="cm_op")
        with colB: b = st.selectbox("B", options=[None]+display_cols, index=0, key="cm_b")
        mname = st.text_input("Metric name", "")
        color = st.color_picker("Color", "#1A78CF")
        lib = st.checkbox("Less is better", value=False)
        if st.button("Add metric", type="primary", key="btn_add_metric"):
            if mname and a and b and op:
                try:
                    sa = pd.to_numeric(df_work[a], errors="coerce")
                    sb = pd.to_numeric(df_work[b], errors="coerce")
                    if op == "+": s = sa + sb
                    elif op == "-": s = sa - sb
                    elif op == "*": s = sa * sb
                    elif op == "/": s = sa / sb.replace(0, np.nan)
                    s = s.replace([np.inf, -np.inf], np.nan).round(2)
                    df_work[mname] = s
                    st.session_state["custom_metrics"][mname] = {"a":a,"op":op,"b":b,"color":color,"lib":lib}
                    set_less_is_better(mname, lib)
                    st.success(f"Added metric '{mname}'.")
                except Exception as e:
                    st.error(f"Failed to add metric: {e}")
            else:
                st.warning("Please fill A, Op, B and a name.")

    st.markdown("**Existing custom metrics:**")
    if st.session_state["custom_metrics"]:
        st.dataframe(pd.DataFrame(st.session_state["custom_metrics"]).T)
    else:
        st.caption("No custom metrics yet.")

    st.subheader("Create archetypes")
    with st.expander("Add a custom archetype", expanded=False):
        aname = st.text_input("Archetype name", key="arch_name")
        astats = st.multiselect("Stats for this archetype", options=display_cols, key="arch_stats")
        if st.button("Add archetype", key="btn_add_arch"):
            if aname and astats:
                st.session_state["custom_arches"][aname] = {"stats": astats}
                st.success(f"Added archetype '{aname}'.")
            else:
                st.warning("Pick a name and at least one stat.")

    # weights manager
    st.subheader("Archetype Weights (per stat)")
    arch_all = all_archetypes_dict()
    if arch_all:
        w_arch = st.selectbox("Select archetype", list(arch_all.keys()), key="aw_select")
        w_stats = arch_all[w_arch]
        w = get_arch_weights(w_arch, w_stats)
        cols = st.columns(min(4, len(w_stats)))
        new_w = {}
        for i, s in enumerate(w_stats):
            with cols[i % len(cols)]:
                new_w[s] = st.number_input(f"{s}", min_value=0.0, value=float(w.get(s, 1.0)), step=0.1, key=f"aw_{w_arch}_{s}")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Save weights", key=f"aw_save_{w_arch}"):
                set_arch_weights(w_arch, new_w); st.success("Saved.")
        with c2:
            if st.button("Set equal weights", key=f"aw_eq_{w_arch}"):
                set_arch_weights(w_arch, {s:1.0 for s in w_stats}); st.success("Reset to equal weights.")
        with c3:
            if st.button("Reset to defaults", key=f"aw_def_{w_arch}"):
                set_arch_weights(w_arch, default_weights_for(w_arch, w_stats)); st.success("Reset to defaults.")
        with c4:
            if st.button("Export all weights", key=f"aw_exp_all"):
                payload = {"weights": st.session_state["arch_weights"]}
                st.download_button("Download JSON", data=json.dumps(payload, indent=2).encode("utf-8"),
                                   file_name="all_archetype_weights.json", mime="application/json", key=unique_key("dl_allw"))
        with st.expander("Import all weights JSON", expanded=False):
            up = st.file_uploader("Upload JSON", type=["json"], key="aw_imp_all")
            if up is not None:
                try:
                    data = json.loads(up.read().decode("utf-8"))
                    if isinstance(data, dict) and "weights" in data and isinstance(data["weights"], dict):
                        cleaned = {}
                        for arch, wmap in data["weights"].items():
                            if arch in arch_all and isinstance(wmap, dict):
                                cleaned[arch] = {s: float(wmap.get(s, 1.0)) for s in arch_all[arch]}
                        st.session_state["arch_weights"].update(cleaned)
                        st.success("Imported weights.")
                    else:
                        st.error("JSON missing 'weights' object.")
                except Exception as e:
                    st.error(f"Import failed: {e}")

# ---------- Table ----------
elif mode == "Table":
    st.subheader("Cleaned data (floats rounded to 2 dp)")
    st.dataframe(df_work.drop(columns=["__pos_tokens"]), use_container_width=True)
    st.caption(f"Rows: {len(df_work):,} • Columns: {df_work.drop(columns=['__pos_tokens']).shape[1]}")
# =========================
# ========= END ===========
# =========================
