# fmApp.py — FM Analytics (pos-aware role scores, Plotly, pretty cards)
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

# Try to load mplsoccer for Matplotlib pizza
_HAS_MPLSOCCER = True
try:
    from mplsoccer import PyPizza
except Exception:
    _HAS_MPLSOCCER = False

# =========================
# ======== CONFIG =========
# =========================
st.set_page_config(page_title="FM Analytics", layout="wide")

# --- Minimal UI CSS (fonts + cards + plotly hover) ---
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Gabarito:wght@400;700&display=swap');
      :root, body, .stApp { --app-font: 'Gabarito', 'DejaVu Sans', Arial, sans-serif; }
      .stApp * { font-family: var(--app-font) !important; }

      /* Card grid for Player Finder results */
      .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 12px;
      }
      .pcard {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,.08);
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,.06);
      }
      .pcard h4 {
        margin: 0 0 4px 0;
        font-size: 1.05rem;
        line-height: 1.15;
        color: #111;
      }
      .pcard .sub {
        color: #333;
        font-size: 0.92rem;
        margin-bottom: 6px;
      }
      .pill {
        display: inline-block;
        background: #f1ffcd;
        border: 1px solid #cfd9a7;
        color: #222;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 6px;
        margin-top: 4px;
      }
      .kv { font-size: 0.85rem; color:#000; }
      .kv b { color:#000; }

      /* Streamlit tweaks */
      .metric-ct { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
    </style>
    """,
    unsafe_allow_html=True
)

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

# ---------- Import Guide (smaller images) ----------
GUIDE_IMG_WIDTH = 560  # px — tweak smaller/larger to taste

def _guide_img(n: int):
    """Show tutorial_n.png if present, at a fixed smaller width (no deprecation)."""
    import os
    p = os.path.join(APP_DIR, f"tutorial_{n}.png")
    if os.path.isfile(p):
        st.image(p, width=GUIDE_IMG_WIDTH)
    else:
        st.caption(f"*tutorial_{n}.png not found in working directory.*")

def render_import_guide():
    st.header("Import Guide")

    # --- VERBATIM TEXT BLOCKS + IMAGES ---
    st.markdown(r"""
Download the Player Search.fmf view, and put this into Documents\Football Manager 2024\Views 

Create the folder if it doesn't exist.

In order to then narrow down the players, I have a filter called leagues filter, Download this if you just want some basic leagues to filter by. Put this in Documents\Football Manager 2024\Filters

Create the folder if it doesn't exist.

With the game open, navigate to scouting, and then to player search. From here, click overview, hover over custom, and click import.
""")
    _guide_img(1)

    st.markdown(r"""
With your view now set up, to use the league filters, click "New/Edit search", navigate to the bottom left corner, click the cog icon, and then click manage filters. When in here, click the import button, and select the filter.
""")
    _guide_img(2)
    _guide_img(3)
    _guide_img(4)

    st.markdown(r"""
With it loaded in, you will be brought back to the manage filters screen. Select your desired filter and then click the Ok button.

  
""")
    _guide_img(5)

    st.markdown(r"""
The leagues may appear blank when you first load the filter in, don't worry! the filter will be working. If they appear blank, simply click out of the edit search menu by clicking Ok, and click back onto it. 

It is important to note that if you want the most accurate dataset, you should untick the "interested" buttons  

And you should also go back into edit search, click Exclude, and make sure that (your club)'s players are unticked 

Now you have your dataset ready, to export the data you will need to click the top row of the players. Then with the top row selected press Control(command for mac) + A at the same time, wait for a second, if you can move your mouse on screen and nothing is highlighting or changing then it's just taking a second to process, don't click anything. 

  
""")
    _guide_img(6)
    _guide_img(7)
    _guide_img(9)

    st.markdown(r"""
If things are highlighting after you've clicked Control + A, then just click the back arrow, and go back into the mode and try again.

  
""")
    _guide_img(8)

    st.markdown(r"""
Hopefully you've been able to highlight every player. 

Now click Control/Command + P, which should bring up this menu.

""")
    _guide_img(10)

    st.markdown(r"""
Select web page, and save it somewhere you can easily access! (I made a Transfers folder in my Football Manager 2024 Documents)


With your file exported. Come back to the site and click upload. Find your .html file and click open.

It will now read your file, in my experience it takes around 25 seconds to import around 4,000 players.  Times may vary.
""")

# ---- Money parsing (e.g., "£2.5M - £3.2M", "$800K", "€1.2B") -> numeric £m-ish ----
_MONEY_RE = re.compile(r"([£$€])?\s*([0-9]*\.?[0-9]+)\s*([KkMmBb])?")
def _scalar_from_token(n: float, tok: str | None) -> float:
    if not tok: return n
    t = tok.upper()
    if t == "K": return n / 1_000_000.0
    if t == "M": return n
    if t == "B": return n * 1000.0
    return n

def money_to_millions(x: str | float | int) -> float | None:
    """Return mid-value in £millions (symbol ignored; treat units consistently)."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    s = str(x).strip()
    if not s: return None
    # ranges like "£2.5M - £3.2M"
    parts = re.split(r"\s*[-–]\s*", s)
    vals = []
    for p in parts:
        m = _MONEY_RE.search(p)
        if not m: continue
        num = float(m.group(2))
        unit = m.group(3)
        vals.append(_scalar_from_token(num, unit))
    if not vals:
        # maybe plain number
        try:
            return float(s) / 1_000_000.0
        except Exception:
            return None
    if len(vals) == 1:
        return vals[0]
    return float(np.mean(vals))

def fig_to_png_bytes(fig, dpi: int = 300) -> bytes:
    """Safely convert a Matplotlib figure to PNG bytes (for Streamlit downloads)."""
    from io import BytesIO
    buf = BytesIO()
    try:
        # draw canvas if backend needs it
        if hasattr(fig, "canvas") and hasattr(fig.canvas, "draw"):
            fig.canvas.draw()
    except Exception:
        pass
    face = fig.get_facecolor() if hasattr(fig, "get_facecolor") else None
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=face)
    buf.seek(0)
    return buf.getvalue()

def money_min_max_millions(s: str) -> Tuple[float | None, float | None]:
    parts = re.split(r"\s*[-–]\s*", str(s).strip())
    nums = []
    for p in parts:
        m = _MONEY_RE.search(p)
        if not m: continue
        num = float(m.group(2)); unit = m.group(3)
        nums.append(_scalar_from_token(num, unit))
    if not nums:
        v = money_to_millions(s)
        return (v, v)
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (float(min(nums)), float(max(nums)))

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

def set_defaults_less_is_better():
    set_less_is_better("Mistakes Leading to Goal", True)
    set_less_is_better("Conceded/90", True)
    set_less_is_better("Red Cards", True)
    set_less_is_better("Yellow Cards", True)
    set_less_is_better("Offsides", True)
    # value columns (cheaper is "better" if used as percentile)
    set_less_is_better("Transfer Value £m (min)", True)
    set_less_is_better("Transfer Value £m (max)", True)
    set_less_is_better("Transfer Value £m (mid)", True)
    # defensive positives
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

    # per90 via minutes
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

    # Transfer Value parsing -> £m fields
    tv_col = find_col(df, ["Transfer Value"])
    if tv_col:
        mins_ = []; maxs_ = []; mids_ = []
        for v in df[tv_col].astype(str).tolist():
            mn, mx = money_min_max_millions(v)
            mins_.append(mn if mn is not None else np.nan)
            maxs_.append(mx if mx is not None else np.nan)
            mids_.append(np.mean([mn, mx]) if (mn is not None and mx is not None) else (mn if mn is not None else (mx if mx is not None else np.nan)))
        df["Transfer Value £m (min)"] = np.array(mins_, dtype=float).round(3)
        df["Transfer Value £m (max)"] = np.array(maxs_, dtype=float).round(3)
        df["Transfer Value £m (mid)"] = np.array(mids_, dtype=float).round(3)

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

# Archetype -> positional tokens baseline (pos-aware role scores)
ARCH_BASELINE_TOKENS: Dict[str, List[str]] = {
    # Goalkeepers
    "GK — Shot Stopper": ["GK"],
    "GK — Sweeper Keeper": ["GK"],
    # Centre-backs
    "CB — Stopper": ["D (C)"],
    "CB — Ball Playing": ["D (C)"],
    # Full-backs / Wing-backs
    "FB — Overlapping": ["D (R)","D (L)","WB (R)","WB (L)"],
    "FB — Inverted":    ["D (R)","D (L)","WB (R)","WB (L)"],
    # Defensive Midfield
    "DM — Ball Winner": ["DM"],
    "DM — Deep-Lying Playmaker": ["DM"],
    # Central Midfield
    "CM — Box to Box": ["M (C)"],
    "CM — Progresser": ["M (C)"],
    # Attacking Midfield
    "AM — Classic 10": ["AM (C)"],
    "AM — Shadow Striker": ["AM (C)"],
    # Wingers
    "Winger — Classic":  ["AM (R)","AM (L)","M (R)","M (L)"],
    "Winger — Inverted": ["AM (R)","AM (L)","M (R)","M (L)"],
    # Strikers
    "ST — Poacher": ["ST","ST (C)"],
    "ST — Target Man": ["ST","ST (C)"],
}

# =========================
# ======= PLOTTING ========
# =========================
def get_contrast_text_color(hex_color):
    r, g, b = mcolors.hex2color(hex_color)
    brightness = (r*299 + g*587 + b*114) * 255 / 1000
    return "#000000" if brightness > 140 else "#F2F2F2"

# --- Matplotlib pizza (optional) ---
def mpl_pizza(player_row: pd.Series, stat_cols: List[str], title: str,
              light: bool = True, base_df: pd.DataFrame | None = None):
    if not _HAS_MPLSOCCER:
        st.error("mplsoccer is not installed. Install with: pip install mplsoccer")
        return None
    if player_row is None or not stat_cols:
        st.info("Pick a player and at least 1 stat."); return None

    base = base_df if base_df is not None and not base_df.empty else _CURRENT_DF

    pcts, raw_vals = [], []
    for s in stat_cols:
        if s not in base.columns:
            pcts.append(0.0); raw_vals.append(np.nan); continue
        series = pd.to_numeric(base[s], errors="coerce").replace([np.inf, -np.inf], np.nan)
        v = pd.to_numeric(player_row.get(s), errors="coerce")
        raw_vals.append(None if pd.isna(v) else float(v))
        if series.dropna().empty or pd.isna(v):
            pcts.append(0.0); continue
        if is_less_better(s): asc_series, asc_v = -series, -v
        else:                 asc_series, asc_v =  series,  v
        arr = asc_series.dropna().to_numpy(); arr.sort()
        lo = np.searchsorted(arr, asc_v, side="left")
        hi = np.searchsorted(arr, asc_v, side="right")
        pct = ((lo + hi)/2) / arr.size * 100.0
        pcts.append(float(np.clip(pct, 0.0, 100.0)))

    slice_colors = ["#2E4374", "#1A78CF", "#D70232", "#FF9300", "#44C3A1",
                    "#CA228D", "#E1C340", "#7575A9", "#9DDFD3"] * 6
    slice_colors = slice_colors[:len(stat_cols)]
    text_colors  = [get_contrast_text_color(c) for c in slice_colors]
    params_disp  = [s for s in stat_cols]

    bg = POSTER_BG if light else "#222222"
    param_color = "#000000" if light else "#fffff0"
    value_txt_color = "#000000" if light else "#fffff0"

    baker = PyPizza(
        params=params_disp,
        background_color=bg,
        straight_line_color="#000000",
        straight_line_lw=.3,
        last_circle_color="#000000",
        last_circle_lw=1,
        other_circle_lw=0,
        inner_circle_size=0.30
    )

    def _fmt_val(x):
        if x is None or pd.isna(x): return None
        if float(x).is_integer(): return int(round(float(x)))
        return round(float(x), 2)

    fig, ax = baker.make_pizza(
        pcts,
        alt_text_values=[_fmt_val(v) for v in raw_vals],
        figsize=(9.4, 9.8),
        color_blank_space="same",
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=slice_colors,
        blank_alpha=0.40,
        kwargs_slices=dict(edgecolor="#000000", zorder=2, linewidth=1),
        kwargs_params=dict(color=param_color, fontsize=12, fontproperties=font_normal, va="center"),
        kwargs_values=dict(
            color=value_txt_color,
            fontsize=11,
            fontproperties=font_normal,
            zorder=3,
            bbox=dict(edgecolor="#000000", facecolor="cornflowerblue",
                      boxstyle=f"round,pad=0.16", lw=1)
        ),
    )
    ax.set_position([0.06, 0.07, 0.88, 0.77])

    name = player_row.get("Name", "Player")
    club = player_row.get("Club", "")
    pos  = player_row.get("Pos", "")
    age  = player_row.get("Age", np.nan)
    mins = player_row.get("Minutes", np.nan)
    apps = player_row.get("Appearances", np.nan)

    def _fmt_intish(x):
        try:
            f = float(x)
            if np.isnan(f): return ""
            return f"{int(round(f)):,}"
        except Exception:
            return str(x)

    age_str  = "" if pd.isna(age) else f"{int(round(float(age)))}"
    mins_str = _fmt_intish(mins)
    apps_str = _fmt_intish(apps)

    fig.text(0.5, 0.988, str(name), ha='center', va='top', fontsize=22,
             color=param_color, fontproperties=font_bold)
    sub1 = " • ".join([x for x in [str(pos) if pos else "", f"Age {age_str}" if age_str else "", str(club) if club else ""] if x])
    if sub1:
        fig.text(0.5, 0.954, sub1, ha='center', va='top', fontsize=14,
                 color=param_color, fontproperties=font_normal)
    sub2 = " • ".join([x for x in [f"Minutes {mins_str}" if mins_str else "", f"Apps {apps_str}" if apps_str else ""] if x])
    if sub2:
        fig.text(0.5, 0.93, sub2, ha='center', va='top', fontsize=12,
                 color=param_color, fontproperties=font_normal)
    if title:
        fig.text(0.5, 0.91, f"{title}", ha='center', va='top', fontsize=12,
                 color=param_color, fontproperties=font_normal)
    return fig

# --- Plotly pizza ---
def plotly_pizza(player_row: pd.Series, stat_cols: List[str], base_df: pd.DataFrame) -> go.Figure:
    base = base_df if base_df is not None and not base_df.empty else _CURRENT_DF
    pts = compute_percentiles_for(player_row, stat_cols, base)
    stats = []; pcts = []; vals = []
    for s, p, v in pts:
        if not np.isnan(p):
            stats.append(s); pcts.append(float(p)); vals.append(v)
    if not stats:
        return go.Figure()

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
        customdata=np.array([stats, pcts, vals], dtype=object).T,
        hovertemplate="<b>%{customdata[0]}</b><br>Percentile: %{customdata[1]:.1f}"
                      "<br>Value: %{customdata[2]}<extra></extra>"
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

# ---- Default weights per archetype ----
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

# --- helper: resolve possible keys for a given archetype name
def _resolve_weight_key(arch_name: str) -> list[str]:
    """Try both '(Custom) Name' and 'Name' so older/saved keys still work."""
    keys = [arch_name]
    if arch_name.startswith("(Custom) "):
        keys.append(arch_name.replace("(Custom) ", "", 1))
    else:
        keys.append(f"(Custom) {arch_name}")
    return keys

def get_arch_weights(arch_name: str, stats: list[str]) -> dict[str, float]:
    """
    Returns per-stat weights for this archetype:
    1. Start from 1.0 for every stat
    2. Overlay built-in defaults (if any)
    3. Overlay saved user weights (supports both '(Custom) Name' and 'Name' keys)
    """
    # built-in defaults (empty for customs)
    base_default = DEFAULT_ARCHETYPE_WEIGHTS.get(arch_name, {})

    # user-saved weights (look under flexible keys)
    saved = {}
    for k in _resolve_weight_key(arch_name):
        if k in st.session_state["arch_weights"]:
            saved = st.session_state["arch_weights"][k]
            break

    # merge precedence: 1.0 -> built-in default -> saved
    w = {s: 1.0 for s in stats}
    for s in stats:
        if s in base_default:
            w[s] = float(base_default[s])
    for s in stats:
        if s in saved:
            w[s] = float(saved[s])
    return w


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

# If nothing uploaded yet, show Import Guide as its own sidebar mode/page
if uploaded is None:
    st.sidebar.header("Mode")
    st.sidebar.radio("Mode", ["Import Guide"], index=0)
    render_import_guide()
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
    # only keep stats present & numeric
    keep = {}
    for k, lst in arch_all.items():
        numeric_stats = [s for s in lst if s in df_work.columns and pd.api.types.is_numeric_dtype(df_work[s])]
        if numeric_stats:
            keep[k] = numeric_stats
    return keep

def default_weights_for(arch_name: str, stats: List[str]) -> Dict[str, float]:
    base = DEFAULT_ARCHETYPE_WEIGHTS.get(arch_name, {})
    return {s: float(base.get(s, 1.0)) for s in stats}

def get_arch_weights(arch_name: str, stats: List[str]) -> Dict[str, float]:
    saved = st.session_state["arch_weights"].get(arch_name, {})
    w = default_weights_for(arch_name, stats)
    w.update({s: float(saved.get(s, w.get(s, 1.0))) for s in stats})
    return {s: float(w.get(s, 1.0)) for s in stats}

def role_baseline_df(arch_name: str) -> pd.DataFrame:
    """Return position-restricted baseline for a given archetype; fallback to user baseline."""
    toks = ARCH_BASELINE_TOKENS.get(arch_name, None)
    if toks:
        df_role = filter_by_tokens(df_work, toks)
        if not df_role.empty:
            return df_role
    return BASELINE_DF

def role_scores_for_archetype(arch_name: str) -> pd.Series:
    """
    Role Score (0–100) = weighted composite of per-stat percentiles (0–100),
    computed vs archetype-specific positional baseline.
    """
    arch_all = all_archetypes_dict()
    stats = arch_all.get(arch_name, [])
    if not stats:
        return pd.Series([np.nan]*len(df_work), index=df_work.index)

    base_df_role = role_baseline_df(arch_name)

    weights = get_arch_weights(arch_name, stats)
    w = np.array([float(weights.get(s, 1.0)) for s in stats], dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)

    pct_mat = []
    for s in stats:
        pct_s = column_percentiles(df_work[s], base_df_role[s], is_less_better(s))
        pct_mat.append(pct_s.to_numpy())
    P = np.vstack(pct_mat).T  # [n_players, n_stats]

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

# ---- Pizza (Plotly ↔ Matplotlib) ----
if mode == "Pizza":
    st.subheader("Custom Pizza")
    renderer = st.selectbox("Renderer", ["Plotly", "Matplotlib"], index=0,
                            help="Choose Plotly for interactivity or Matplotlib (mplsoccer) for publication-style.")
    default_stats = [x for x in ["Shots/90","SoT/90","Expected Goals","Expected Assists/90","Dribbles/90","Open Play Key Passes/90"] if x in numeric_cols]
    stats_pick = st.multiselect("Choose stats (ordered)", options=numeric_cols, default=default_stats)

    if renderer == "Plotly":
        fig = plotly_pizza(player_row, stats_pick, BASELINE_DF)
        fig.update_layout(title=f"{player} — Custom Pizza")
        _plotly_polar_black(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        style = st.selectbox("Matplotlib Style", ["Light", "Dark"], index=0)
        if not _HAS_MPLSOCCER:
            st.error("mplsoccer is not installed. Install with: pip install mplsoccer")
        else:
            fig = mpl_pizza(player_row, stats_pick, title="Custom Pizza", light=(style=="Light"), base_df=BASELINE_DF)
            if fig is not None:
                st.pyplot(fig, clear_figure=False)
                st.download_button("Download pizza (PNG)", data=fig_to_png_bytes(fig),
                                   file_name=f"pizza_{str(player).replace(' ','_')}.png",
                                   mime="image/png", key=unique_key("dl_pizza"))

# ---- Archetypes (permanent weights pane, no dropdowns/expanders) ----
elif mode == "Archetypes":
    st.subheader("Archetype Pizza (per-stat weights; role scores are pos-aware)")
    arch_all = all_archetypes_dict()
    if not arch_all:
        st.warning("No archetypes with valid stats available.")
    else:
        arch_name = st.selectbox("Archetype", list(arch_all.keys()))
        arch_stats = arch_all.get(arch_name, [])
        if len(arch_stats) < 1:
            st.warning("No valid stats for this archetype in the current dataset.")
        else:
            st.markdown("**Weights (per stat for this archetype)**")
            w = get_arch_weights(arch_name, arch_stats)
            cols = st.columns(min(4, len(arch_stats)))
            new_w = {}
            for i, s in enumerate(arch_stats):
                with cols[i % len(cols)]:
                    new_w[s] = st.number_input(f"{s}", min_value=0.0, value=float(w.get(s, 1.0)), step=0.1, key=f"w_{arch_name}_{s}")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Save weights", key=f"savew_{arch_name}"):
                    st.session_state["arch_weights"][arch_name] = {k: float(v) for k, v in new_w.items()}
                    st.success("Saved weights.")
            with c2:
                if st.button("Equal weights", key=f"eqw_{arch_name}"):
                    st.session_state["arch_weights"][arch_name] = {s:1.0 for s in arch_stats}
                    st.success("Set equal weights.")
            with c3:
                if st.button("Reset to defaults", key=f"defaultw_{arch_name}"):
                    st.session_state["arch_weights"][arch_name] = default_weights_for(arch_name, arch_stats)
                    st.success("Reset to defaults.")

            renderer_arch = st.selectbox("Renderer", ["Plotly", "Matplotlib"], index=0, key="arch_renderer")
            if renderer_arch == "Plotly":
                # pizza vs archetype's position baseline
                role_base = role_baseline_df(arch_name)
                fig = plotly_pizza(player_row, arch_stats, role_base)
                fig.update_layout(title=f"{player} — {arch_name} (pos baseline)")
                _plotly_polar_black(fig)
                st.plotly_chart(fig, use_container_width=True, theme=None)
            else:
                style = st.selectbox("Matplotlib Style", ["Light", "Dark"], index=0, key="arch_style2")
                if not _HAS_MPLSOCCER:
                    st.error("mplsoccer is not installed. Install with: pip install mplsoccer")
                else:
                    fig = mpl_pizza(player_row, arch_stats, title=f"{arch_name} (pos baseline)", light=(style=="Light"),
                                    base_df=role_baseline_df(arch_name))
                    if fig is not None:
                        st.pyplot(fig, clear_figure=False)
                        st.download_button("Download pizza (PNG)", data=fig_to_png_bytes(fig),
                                           file_name=f"pizza_{str(player).replace(' ','_')}_{re.sub(r'\\W+','_',arch_name)}.png",
                                           mime="image/png", key=unique_key("dl_archpizza"))

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
# ---------- Role Scatter ----------
elif mode == "Role Scatter":
    st.subheader("Role Scatter (pos-aware; weighted composite of percentiles, 0–100)")
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

        # --- Compute scores for ALL players (df_work), then display only players in BASELINE_DF (sidebar filter)
        score_x_all = role_scores_for_archetype(ax)  # index = df_work.index
        score_y_all = role_scores_for_archetype(ay)

        # Display frame is strictly the sidebar-filtered cohort:
        df_disp = BASELINE_DF[[name_col]].copy()
        if "Club" in BASELINE_DF.columns: df_disp["Club"] = BASELINE_DF["Club"]
        if pos_col in BASELINE_DF.columns: df_disp["Pos"] = BASELINE_DF[pos_col]

        # Reindex role scores to the displayed cohort (do NOT filter by archetype tokens!)
        df_disp[ax] = score_x_all.reindex(df_disp.index)
        df_disp[ay] = score_y_all.reindex(df_disp.index)
        df_disp = df_disp[df_disp[ax].notna() & df_disp[ay].notna()]

        fig = go.Figure()

        if show_cloud:
            sc = px.scatter(df_disp, x=ax, y=ay, opacity=0.35, hover_name=name_col,
                            hover_data={name_col:False,"Club":True,"Pos":True,ax:":.1f",ay:":.1f"})
            for tr in sc.data: fig.add_trace(tr)

        if age_ok:
            ages = pd.to_numeric(BASELINE_DF["Age"], errors="coerce")
            if u23:
                cohort_ix = BASELINE_DF.index[ages < 23]
                cohort = df_disp.loc[df_disp.index.intersection(cohort_ix)]
                fig.add_trace(go.Scatter(x=cohort[ax], y=cohort[ay], mode="markers",
                                         marker=dict(size=10, color="rgba(26,120,207,0.85)", line=dict(width=1, color="black")),
                                         name="U23", hovertext=cohort[name_col], hoverinfo="text"))
            if u21:
                cohort_ix = BASELINE_DF.index[ages < 21]
                cohort = df_disp.loc[df_disp.index.intersection(cohort_ix)]
                fig.add_trace(go.Scatter(x=cohort[ax], y=cohort[ay], mode="markers",
                                         marker=dict(size=10, color="rgba(199, 43, 98, 0.85)", line=dict(width=1, color="black")),
                                         name="U21", hovertext=cohort[name_col], hoverinfo="text"))

        if hi_name != "(none)":
            prow = df_disp[df_disp[name_col] == hi_name]
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
# ---------- Top 10 — Roles ----------
elif mode == "Top 10 — Roles":
    st.subheader("Top 10 by Role Score (pos-aware, weighted composite, 0–100)")
    arch_all = all_archetypes_dict()
    if not arch_all:
        st.info("No archetypes with valid stats available.")
    else:
        role = st.selectbox("Role", list(arch_all.keys()))
        # Compute scores for everyone, then slice to the sidebar selection only:
        scores_all = role_scores_for_archetype(role).round(1)
        scores = scores_all.reindex(BASELINE_DF.index)

        tbl = BASELINE_DF[[name_col]].copy()
        if "Club" in BASELINE_DF.columns: tbl["Club"] = BASELINE_DF["Club"]
        if pos_col in BASELINE_DF.columns: tbl["Pos"] = BASELINE_DF[pos_col]
        tbl["Score"] = scores

        top = tbl.dropna(subset=["Score"]).sort_values("Score", ascending=False).head(10)

        fig = go.Figure(go.Bar(
            x=top["Score"], y=top[name_col], orientation="h",
            hovertemplate="<b>%{y}</b><br>Role Score: %{x:.1f}<br>Club: %{customdata[0]}<extra></extra>",
            customdata=np.stack([top.get("Club", pd.Series([""]*len(top)))], axis=1)
        ))
        fig.update_layout(
            title=f"Top 10 — {role} (computed vs role baseline, shown for sidebar selection)",
            xaxis=dict(title="Role Score (0–100)", range=[0, 100]),
            yaxis=dict(autorange="reversed"),
            template="simple_white",
            paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
            font=dict(family=FONT_FAMILY)
        )
        _plotly_axes_black(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)


# ---------- Player Finder (values/percentiles; fancy cards; supports Transfer Value £m) ----------
elif mode == "Player Finder":
    st.subheader("Player Finder (values or percentiles)")
    st.caption("Tip: Transfer Value fields are in **£ millions**: ‘Transfer Value £m (min/max/mid)’.")

    mode_type = st.radio("Use", ["Values", "Percentiles"], horizontal=True)

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

        # Build “cards” HTML
        cards = []
        for _, r in res.iterrows():
            nm   = str(r.get(name_col, ""))
            club = str(r.get("Club", "")) if "Club" in res.columns else ""
            pos  = str(r.get(pos_col, "")) if pos_col in res.columns else ""
            age  = r.get("Age", "")
            mins = r.get("Minutes", "")
            rat  = r.get("Avg Rating", "")
            tvm  = r.get("Transfer Value £m (mid)", np.nan)
            tvtxt = f"£{tvm:.2f}m" if pd.notna(tvm) else (str(r.get("Transfer Value", "")) if "Transfer Value" in res.columns else "—")

            # show criteria values
            crit_txts = []
            for (s, _, _) in crits:
                val = r.get(s, np.nan)
                if pd.notna(val):
                    crit_txts.append(f"<span class='pill'>{s}: <b>{val}</b></span>")
                pct_name = f"{s} (pct)"
                if pct_name in res.columns and pd.notna(r.get(pct_name, np.nan)):
                    crit_txts.append(f"<span class='pill'>{s} pct: <b>{r.get(pct_name):.0f}</b></span>")

            html = f"""
              <div class="pcard">
                <h4>{nm}</h4>
                <div class="sub">{club} &nbsp;•&nbsp; {pos}</div>
                <div class="kv"><b>Age:</b> {age} &nbsp; <b>Min:</b> {mins} &nbsp; <b>Rating:</b> {rat if pd.notna(rat) else '—'}</div>
                <div class="kv" style="margin-top:4px;"><b>Transfer Value:</b> {tvtxt}</div>
                <div class="metric-ct">{''.join(crit_txts)}</div>
              </div>
            """
            cards.append(html)

        st.markdown(f"""<div class="card-grid">{''.join(cards) if cards else '<em>No matches</em>'}</div>""", unsafe_allow_html=True)

        # CSV download
        st.download_button(
            "Download CSV",
            data=res.to_csv(index=False).encode("utf-8"),
            file_name="player_finder_results.csv",
            mime="text/csv",
            key=unique_key("dl_pf")
        )

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

# ---------- Build: Metrics & Archetypes (no expanders; always visible) ----------
elif mode == "Build: Metrics & Archetypes":
    st.subheader("Create a custom metric")
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

    st.subheader("Create a custom archetype")
    aname = st.text_input("Archetype name", key="arch_name")
    astats = st.multiselect("Stats for this archetype", options=display_cols, key="arch_stats")

    # NEW: optional initial weights UI for selected stats
    custom_init_weights = {}
    if astats:
        st.markdown("**Initial weights (per stat)**")
        cols_w = st.columns(min(4, len(astats)))
        for i, s in enumerate(astats):
            with cols_w[i % len(cols_w)]:
                custom_init_weights[s] = st.number_input(
                    f"{s}",
                    min_value=0.0,
                    value=1.0,
                    step=0.1,
                    key=f"arch_w_{s}"
                )

    if st.button("Add archetype", key="btn_add_arch"):
        if aname and astats:
            # save the archetype
            st.session_state["custom_arches"][aname] = {"stats": astats}
            # NEW: persist the weights specifically for this custom archetype
            if custom_init_weights:
                st.session_state["arch_weights"][f"(Custom) {aname}"] = {
                    s: float(custom_init_weights.get(s, 1.0)) for s in astats
                }
            st.success(f"Added archetype '{aname}' with weights.")
        else:
            st.warning("Pick a name and at least one stat.")


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
                st.session_state["arch_weights"][w_arch] = {k: float(v) for k, v in new_w.items()}
                st.success("Saved.")
        with c2:
            if st.button("Set equal weights", key=f"aw_eq_{w_arch}"):
                st.session_state["arch_weights"][w_arch] = {s:1.0 for s in w_stats}
                st.success("Reset to equal weights.")
        with c3:
            if st.button("Reset to defaults", key=f"aw_def_{w_arch}"):
                st.session_state["arch_weights"][w_arch] = default_weights_for(w_arch, w_stats)
                st.success("Reset to defaults.")
        with c4:
            if st.button("Export all weights", key=f"aw_exp_all"):
                payload = {"weights": st.session_state["arch_weights"]}
                st.download_button("Download JSON", data=json.dumps(payload, indent=2).encode("utf-8"),
                                   file_name="all_archetype_weights.json", mime="application/json", key=unique_key("dl_allw"))

        st.markdown("**Import all weights JSON**")
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
