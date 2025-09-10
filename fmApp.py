import os, io, re, json, copy
from io import BytesIO
from uuid import uuid4
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---- Optional Matplotlib (only needed for mpl pizza) ----
_HAS_MPL = True
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
except Exception:
    mpl = None
    plt = None
    fm = None
    _HAS_MPL = False

from typing import TYPE_CHECKING, Any

# Safe Figure type for hints even if Matplotlib isn't installed
if TYPE_CHECKING:
    try:
        from matplotlib.figure import Figure  # real type at type-check time
    except Exception:
        class Figure:  # fallback stub
            ...
else:
    try:
        from matplotlib.figure import Figure  # type: ignore
    except Exception:
        Figure = Any  # runtime fallback when matplotlib missin

# ---- Optional mplsoccer (depends on Matplotlib) ----
_HAS_MPLSOCCER = True
try:
    from mplsoccer import PyPizza
except Exception:
    _HAS_MPLSOCCER = False


# =========================
# ======== CONFIG =========
# =========================
st.set_page_config(page_title="FM Analytics", layout="wide")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(APP_DIR, "fonts")
GABARITO_REG = os.path.join(FONT_DIR, "Gabarito-Regular.ttf")
GABARITO_BOLD = os.path.join(FONT_DIR, "Gabarito-Bold.ttf")

def _fontprops_or_fallback(ttf_path: str, fallback_family: str = "DejaVu Sans"):
    if not _HAS_MPL or fm is None:
        return None
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

if _HAS_MPL:
    mpl.rcParams["font.family"] = ["Gabarito", "DejaVu Sans", "Arial", "sans-serif"]

# App style
POSTER_BG = "#f1ffcd"
FONT_FAMILY = "Gabarito, DejaVu Sans, Arial, sans-serif"
GUIDE_IMG_WIDTH = 560  # px — smaller guide images

# Global CSS: fonts + minor UI polish
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Gabarito:wght@400;700&display=swap');
:root, body, .stApp { --app-font: 'Gabarito', 'DejaVu Sans', Arial, sans-serif; }
.stApp * { font-family: var(--app-font) !important; }

/* Card grid for Player Finder */
.card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
.pcard { background: #fff; border: 1px solid #000; border-radius: 14px; padding: 12px 12px 10px; }
.pcard h4 { margin: 0 0 4px 0; font-size: 1.0rem; line-height: 1.2; color: #000; }
.pcard .sub { color: #333; font-size: .86rem; margin-bottom: 6px; }
.pcard .kv { color: #111; font-size: .88rem; }
.pcard .pill { display: inline-block; margin: 4px 6px 0 0; padding: 2px 6px; border: 1px solid #000; border-radius: 999px; font-size: .78rem; color: #000; background: #f7f7f7; }
</style>
""", unsafe_allow_html=True)


# =========================
# ======= UTILITIES =======
# =========================
def unique_key(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"

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
    if hasattr(v, "shape") and len(getattr(v, "shape", [])) >= 2 and v.shape[1] == 1:
        return v.iloc[:, 0]
    if hasattr(v, "shape") and len(getattr(v, "shape", [])) >= 2 and v.shape[1] > 1:
        scores = []
        for i in range(v.shape[1]):
            s = pd.to_numeric(_clean_num(v.iloc[:, i]), errors="coerce")
            scores.append(int(s.notna().sum()))
        best_i = int(np.argmax(scores))
        return v.iloc[:, best_i]
    return v

def find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cols = {re.sub(r"\s+", "", str(c)).lower(): str(c) for c in df.columns}
    for nm in names:
        key = re.sub(r"\s+", "", str(nm)).lower()
        if key in cols:
            return cols[key]
    return None

# Percentile direction
LESS_IS_BETTER: Dict[str, bool] = {}
def set_less_is_better(stat: str, flag: bool) -> None:
    LESS_IS_BETTER[str(stat)] = bool(flag)
def is_less_better(stat: str) -> bool:
    return bool(LESS_IS_BETTER.get(str(stat), False))

# Pre-set common directions
for s in ["Mistakes Leading to Goal","Conceded/90","Red Cards","Yellow Cards","Offsides"]:
    set_less_is_better(s, True)
for s in ["Shots Blocked/90","Blocks/90","Interceptions/90","Clearances/90"]:
    set_less_is_better(s, False)

def _plotly_axes_black(fig: go.Figure):
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black",
                     tickfont=dict(color="black"), titlefont=dict(color="black"),
                     gridcolor="rgba(0,0,0,0.12)")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black",
                     tickfont=dict(color="black"), titlefont=dict(color="black"),
                     gridcolor="rgba(0,0,0,0.12)")

def fig_to_png_bytes(fig, dpi: int = 300) -> bytes:
    """Safely convert a Matplotlib figure to PNG bytes (for Streamlit downloads)."""
    buf = BytesIO()
    try:
        if hasattr(fig, "canvas") and hasattr(fig.canvas, "draw"):
            fig.canvas.draw()
    except Exception:
        pass
    face = fig.get_facecolor() if hasattr(fig, "get_facecolor") else None
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=face)
    buf.seek(0)
    return buf.getvalue()


# =========================
# ====== HTML PARSING =====
# =========================
def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for c in list(out.columns):
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
    # drop obvious index column
    if df.shape[1] >= 2:
        c0 = df.columns[0]
        as_num = pd.to_numeric(df[c0], errors="coerce")
        if as_num.notna().mean() > 0.9 and as_num.dropna().between(0, 999999).mean() > 0.8:
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
    "Clr/90":"Clearances/90","Clear":"Clearances","CCC":"Chances Created","Ch C/90":"Chances Created/90","Blk/90":"Blocks/90","Blk":"Blocks", "Aer A/90":"Aerial Duels Attempted/90"
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

def apply_hard_remap(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in list(df.columns):
        df[c] = _series_for(df, c)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # ---- rename to canonical where we have a hard mapping ----
    df.columns = [RENAME_MAP.get(c, c) for c in df.columns]

    # ---- Distance: strip trailing "km"/"KM" from cell values, then normalise name ----
    # Look for any distance-like column names
    dist_cols_exact = [c for c in df.columns if c in {
        "Distance Covered (KM)", "Distance Covered", "Distance"
    }]
    dist_cols_loose = [c for c in df.columns if re.search(r"\bdistance\b", c, flags=re.I)]

    # Clean cell values on all candidates first (before numeric coercion)
    for c in set(dist_cols_exact or dist_cols_loose):
        try:
            df[c] = (
                df[c].astype(str)
                     .str.replace(r"\s*[kK][mM]\s*$", "", regex=True)  # strip trailing "km"
                     .str.strip()
            )
        except Exception:
            pass

    # Canonicalise the distance column name so /90 logic always picks it up
    if "Distance Covered (KM)" not in df.columns:
        # Prefer a more specific header if present
        if "Distance Covered" in df.columns:
            df.rename(columns={"Distance Covered": "Distance Covered (KM)"}, inplace=True)
        elif "Distance" in df.columns:
            df.rename(columns={"Distance": "Distance Covered (KM)"}, inplace=True)
        else:
            # Fall back to any loose distance-like column
            for c in dist_cols_loose:
                if c != "Distance Covered (KM)" and "/90" not in c:
                    df.rename(columns={c: "Distance Covered (KM)"}, inplace=True)
                    break

    # ---- try to make columns numeric where appropriate ----
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = _maybe_numeric(df[c])

    # ---- minutes denominator (for /90) ----
    min_name = find_col(df, ["Minutes", "Mins", "Min", "Time Played"])
    denom = None
    if min_name:
        mins = pd.to_numeric(df[min_name], errors="coerce")
        denom = (mins / 90.0).replace(0, np.nan)

    if denom is not None:
        for src, tgt in CREATE_PER90_FROM_TOTAL.items():
            if src in df.columns and tgt not in df.columns:
                s = pd.to_numeric(df[src], errors="coerce")
                df[tgt] = (s / denom).round(2)

        # Distance per 90 (works now that name is canonical + values cleaned)
        if "Distance Covered (KM)" in df.columns and "Distance Covered (KM)/90" not in df.columns:
            s = pd.to_numeric(df["Distance Covered (KM)"], errors="coerce")
            df["Distance Covered (KM)/90"] = (s / denom).round(2)

    # ---- xG/Shot (computed fresh) ----
    xg_col = find_col(df, ["Expected Goals", "xG"])
    shots_col = find_col(df, ["Shots"])
    if xg_col and shots_col:
        xg = pd.to_numeric(df[xg_col], errors="coerce")
        sh = pd.to_numeric(df[shots_col], errors="coerce").replace(0, np.nan)
        df["xG/Shot"] = (xg / sh).round(3)

    # ---- round all float cols to 2 dp ----
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(2)
    return df



# =========================
# ===== POS TOKENISER =====
# =========================
def expand_positions(pos_str: str | float) -> List[str]:
    """'D (RLC), DM, M (C)' → ['D (R)','D (L)','D (C)','DM','M (C)', raw]"""
    if pd.isna(pos_str):
        return []
    s = str(pos_str)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    tokens = []
    for p in parts:
        m = re.match(r"^([A-Z]{1,3})\s*\(([^)]+)\)$", p.strip())
        if m:
            base = m.group(1).upper()
            ins = re.sub(r"[^A-Z]", "", m.group(2).upper())
            if ins:
                for ch in ins:
                    tokens.append(f"{base} ({ch})")
            else:
                tokens.append(f"{base}")
        else:
            tokens.append(p)
    tokens.append(s)
    return sorted(set(tokens))

# Allowed tokens for sidebar selector (and display order)
ALLOWED_TOKENS = [
    "GK", "D (R)", "D (C)", "D (L)", "DM", "WB (L)", "WB (R)",
    "M (C)", "M (R)", "M (L)", "AM (C)", "AM (L)", "AM (R)", "ST (C)"
]
_BASE_TO_CENTER = {"ST": "ST (C)", "M": "M (C)", "AM": "AM (C)", "D": "D (C)"}

def _norm_token(tok: str) -> str:
    t = str(tok).strip().upper()
    m = re.match(r"^([A-Z]{1,3})\s*\(([RLC])\)$", t)
    if m:
        return f"{m.group(1)} ({m.group(2)})"
    if t in _BASE_TO_CENTER:
        return _BASE_TO_CENTER[t]
    return t

def filter_by_tokens(df_: pd.DataFrame, tokens: List[str]) -> pd.DataFrame:
    if not tokens:
        return df_
    tokset = set(tokens)
    def _has_tok(lst):
        if not isinstance(lst, list):
            return False
        normed = {_norm_token(t) for t in lst if isinstance(t, str)}
        return bool(tokset.intersection(normed))
    mask = df_["__pos_tokens"].apply(_has_tok)
    out = df_.loc[mask]
    return out if not out.empty else df_


# =========================
# ===== Import Guide ======
# =========================
def _guide_img(n: int):
    """Show tutorial_n.png at a fixed width."""
    p = os.path.join(APP_DIR, f"tutorial_{n}.png")
    if os.path.isfile(p):
        st.image(p, width=GUIDE_IMG_WIDTH)
    else:
        st.caption(f"*tutorial_{n}.png not found in working directory.*")

def render_import_guide():
    st.header("Import Guide")

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


# =========================
# === UNIFIED ROLE BOOK ===
# =========================
# Each role now has:
#   - "baseline": list of positional tokens to compare against (built-ins)
#   - "weights":  { "<Stat Name>": <weight>, ... }  # keys ARE the stats; no separate list
ROLE_BOOK_BUILTIN: Dict[str, Dict[str, object]] = {
    # ---- Goalkeepers ----
    "GK — Shot Stopper": {
        "baseline": ["GK"],
        "weights": {
            "Save %": 2.0, "Expected Save %": 1.0, "Expected Goals Prevented/90": 1.7,
            "Saves Held": 1.2, "Saves Parried": 0.8, "Saves Tipped": 0.6,
            "Conceded/90": 1.2, "Clean Sheets/90": 0.9
        },
    },
    "GK — Sweeper Keeper": {
        "baseline": ["GK"],
        "weights": {
            "Passes Attempted/90": 1.2, "Passes Completed/90": 0.9, "Pass Completion%": 0.9,
            "Progressive Passes/90": 1.6, "Expected Goals Prevented/90": 1.1,
            "Save %": 1.1, "Saves Held": 0.7, "Saves Parried": 0.5
        },
    },

    # ---- Centre-backs ----
    "CB — Stopper": {
        "baseline": ["D (C)"],
        "weights": {
            "Tackles/90": 1.5, "Tackles Won/90": 1.7, "Tackle Ratio": 1.2,
            "Interceptions/90": 1.3, "Blocks/90": 1.1, "Shots Blocked/90": 1.0,
            "Clearances/90": 1.0, "Headers won/90": 1.1, "Header Win Rate": 1.1
        },
    },
    "CB — Ball Playing": {
        "baseline": ["D (C)"],
        "weights": {
            "Passes Attempted/90": 1.4, "Passes Completed/90": 1.2, "Pass Completion%": 1.1,
            "Progressive Passes/90": 1.7, "Key Passes/90": 1.2, "Interceptions/90": 1.0,
            "Tackles/90": 0.9, "Chances Created/90": 1.0
        },
    },

    # ---- Full-backs / Wing-backs ----
    "FB — Overlapping": {
        "baseline": ["D (R)","D (L)","WB (R)","WB (L)"],
        "weights": {
            "Crosses Attempted/90": 1.4, "Crosses Completed/90": 1.6, "Cross Completion Ratio": 1.2,
            "Open Play Crosses Attempted/90": 1.5, "Open Play Crosses Completed/90": 1.6,
            "Open Play Key Passes/90": 1.4, "Key Passes/90": 1.2, "Dribbles/90": 1.2,
            "Tackles/90": 1.0, "Interceptions/90": 1.0
        },
    },
    "FB — Inverted": {
        "baseline": ["D (R)","D (L)","WB (R)","WB (L)"],
        "weights": {
            "Passes Attempted/90": 1.3, "Passes Completed/90": 1.3, "Pass Completion%": 1.4,
            "Progressive Passes/90": 1.7, "Open Play Key Passes/90": 1.4,
            "Interceptions/90": 1.1, "Tackles/90": 1.0, "Chances Created/90": 1.1
        },
    },

    # ---- Defensive Midfield ----
    "DM — Ball Winner": {
        "baseline": ["DM"],
        "weights": {
            "Tackles/90": 1.6, "Tackles Won/90": 1.7, "Tackle Ratio": 1.3,
            "Interceptions/90": 1.6, "Blocks/90": 1.2, "Shots Blocked/90": 1.2,
            "Possession Won/90": 1.5, "Pressures Completed/90": 1.2
        },
    },
    "DM — Deep-Lying Playmaker": {
        "baseline": ["DM"],
        "weights": {
            "Passes Attempted/90": 1.6, "Passes Completed/90": 1.3, "Pass Completion%": 1.3,
            "Progressive Passes/90": 1.7, "Open Play Key Passes/90": 1.3,
            "Key Passes/90": 1.1, "Interceptions/90": 1.0
        },
    },

    # ---- Central Midfield ----
    "CM — Box to Box": {
        "baseline": ["M (C)"],
        "weights": {
            "Progressive Passes/90": 1.4, "Open Play Key Passes/90": 1.2, "Dribbles/90": 1.2,
            "Pressures Completed/90": 1.1, "Tackles/90": 1.2, "Interceptions/90": 1.2,
            "Shots/90": 1.1, "SoT/90": 1.1
        },
    },
    "CM — Progresser": {
        "baseline": ["M (C)"],
        "weights": {
            "Progressive Passes/90": 1.7, "Passes Completed/90": 1.3, "Passes Attempted/90": 1.4,
            "Open Play Key Passes/90": 1.5, "Key Passes/90": 1.3, "Dribbles/90": 1.2,
            "Chances Created/90": 1.5
        },
    },

    # ---- Attacking Midfield ----
    "AM — Classic 10": {
        "baseline": ["AM (C)"],
        "weights": {
            "Open Play Key Passes/90": 1.7, "Key Passes/90": 1.6, "Chances Created/90": 1.7,
            "Assists/90": 1.5, "Progressive Passes/90": 1.3, "Dribbles/90": 1.2,
            "Shots/90": 1.0
        },
    },
    "AM — Shadow Striker": {
        "baseline": ["AM (C)"],
        "weights": {
            "Shots/90": 1.5, "SoT/90": 1.5, "Dribbles/90": 1.1,
            "Chances Created/90": 1.1, "Key Passes/90": 1.1,
            "Conversion Rate": 1.6, "Goals / 90": 1.8
        },
    },

    # ---- Wingers ----
    "Winger — Classic": {
        "baseline": ["AM (R)","AM (L)","M (R)","M (L)"],
        "weights": {
            "Crosses Attempted/90": 1.4, "Crosses Completed/90": 1.6, "Cross Completion Ratio": 1.3,
            "Open Play Crosses Attempted/90": 1.5, "Open Play Crosses Completed/90": 1.7,
            "Open Play Key Passes/90": 1.4, "Dribbles/90": 1.6, "Assists/90": 1.5
        },
    },
    "Winger — Inverted": {
        "baseline": ["AM (R)","AM (L)","M (R)","M (L)"],
        "weights": {
            "Shots/90": 1.6, "SoT/90": 1.6, "Dribbles/90": 1.6,
            "Open Play Key Passes/90": 1.3, "Chances Created/90": 1.4,
            "Conversion Rate": 1.6, "Progressive Passes/90": 1.2
        },
    },

    # ---- Strikers ----
    "ST — Poacher": {
        "baseline": ["ST","ST (C)"],
        "weights": {
            "Shots/90": 1.6, "SoT/90": 1.7, "Conversion Rate": 1.7,
            "Goals / 90": 1.9, "xG/90": 1.7, "xG/Shot": 1.4
        },
    },
    "ST — Target Man": {
        "baseline": ["ST","ST (C)"],
        "weights": {
            "Headers won/90": 1.8, "Header Win Rate": 1.7, "Aerial Duels Attempted/90": 1.6,
            "Shots/90": 1.2, "SoT/90": 1.1, "Key Passes/90": 1.0
        },
    },
}

# =========================
# ===== App STATE =========
# =========================
if "custom_metrics" not in st.session_state:
    st.session_state["custom_metrics"] = {}

if "role_book_custom" not in st.session_state:
    # role -> {"baseline": None (uses sidebar), "weights": {"Stat": weight, ...}}
    st.session_state["role_book_custom"] = {}

if "saved_filters" not in st.session_state:
    st.session_state["saved_filters"] = {}

# Default baseline toggle + active role
st.session_state.setdefault("pct_baseline_mode", "Role baseline")
st.session_state.setdefault("active_role", None)


# =========================
# ==== Percentiles util ===
# =========================
def column_percentiles(series_all: pd.Series, series_baseline: pd.Series, less_is_better: bool) -> pd.Series:
    """Return percentiles (0–100) of each row in series_all vs the distribution in series_baseline."""
    a = pd.to_numeric(series_all, errors="coerce").astype(float)
    b = pd.to_numeric(series_baseline, errors="coerce").astype(float)
    mask = b.notna()
    arr = b[mask].to_numpy()
    out = pd.Series(np.nan, index=a.index, dtype=float)
    if arr.size == 0:
        return out
    if less_is_better:
        arr = -arr
        a_vals = -a
    else:
        a_vals = a
    arr_sorted = np.sort(arr)
    # vectorised rank
    xv = a_vals.to_numpy()
    lo = np.searchsorted(arr_sorted, xv, side="left")
    hi = np.searchsorted(arr_sorted, xv, side="right")
    pct = ((lo + hi) / 2) / arr_sorted.size * 100.0
    out[:] = pct
    return out.clip(0, 100)


# =========================
# ==== Role book helpers ==
# =========================
# =========================
# ===== ROLE HELPERS  =====
# =========================
def role_book_all() -> Dict[str, Dict[str, Any]]:
    """
    Unified role book: built-ins + user overrides + user custom.
    Each entry is { "weights": {stat: weight, ...}, "baseline": [tokens]|None }.
    """
    # Built-ins must be in the unified shape already
    built = {}
    for k, v in ROLE_BOOK_BUILTIN.items():
        if isinstance(v, dict) and "weights" in v:
            built[k] = {"weights": {str(s): float(w) for s, w in v["weights"].items()}, "baseline": v.get("baseline")}
        elif isinstance(v, dict):  # legacy dict of stat->weight
            built[k] = {"weights": {str(s): float(w) for s, w in v.items()}, "baseline": None}
        elif isinstance(v, list):  # legacy list of stats -> equal weights
            built[k] = {"weights": {str(s): 1.0 for s in v}, "baseline": None}
    # Custom / overrides
    custom = st.session_state.get("role_book_custom", {})
    comb = dict(built)
    for k, v in (custom or {}).items():
        if isinstance(v, dict) and "weights" in v:
            comb[k] = {"weights": {str(s): float(w) for s, w in v["weights"].items()}, "baseline": v.get("baseline")}
    return comb


def role_weights_for(role_name: Optional[str]) -> Dict[str, float]:
    """
    Always return a dict of weights; empty dict if role missing or invalid.
    """
    if not role_name:
        return {}
    book = role_book_all()
    cfg = book.get(role_name)
    if not cfg:
        return {}
    w = cfg.get("weights", {})
    if isinstance(w, dict):
        # Do not filter by columns here; callers can intersect with df columns
        return {str(k): float(v) for k, v in w.items()}
    if isinstance(w, list):
        return {str(s): 1.0 for s in w}
    return {}


def role_baseline_tokens(role_name: Optional[str]) -> List[str]:
    """
    Return the role's fixed baseline tokens (if any). Otherwise [].
    """
    if not role_name:
        return []
    book = role_book_all()
    cfg = book.get(role_name) or {}
    toks = cfg.get("baseline")
    if isinstance(toks, list):
        return [_norm_token(t) for t in toks if isinstance(t, str)]
    return []


def pick_baseline_df_for(role_name: Optional[str]) -> pd.DataFrame:
    """
    Choose which dataframe to use as percentile baseline based on global toggle.
    """
    mode = st.session_state.get("pct_baseline_mode", "Role baseline")
    if mode == "Whole dataset":
        return df_work
    if mode == "Sidebar positions":
        return BASELINE_DF if not BASELINE_DF.empty else df_work
    # Role baseline
    toks = role_baseline_tokens(role_name)
    if toks:
        return filter_by_tokens(df_work, toks)
    return BASELINE_DF if not BASELINE_DF.empty else df_work


def role_scores_for_role(role_name: str) -> pd.Series:
    """
    Weighted sum of per-stat percentiles (0..100), normalized by weight sum.
    Never raises; returns NaNs if no usable stats.
    """
    weights = role_weights_for(role_name)
    # intersect with present numeric columns
    usable = {s: w for s, w in weights.items() if s in df_work.columns and pd.api.types.is_numeric_dtype(df_work[s])}
    if not usable:
        return pd.Series(np.nan, index=df_work.index)

    base_df = pick_baseline_df_for(role_name)
    # per-stat percentiles
    pct_cols = {}
    for s, w in usable.items():
        pct_cols[s] = column_percentiles(df_work[s], base_df[s], is_less_better(s))

    # weighted sum / weight_sum
    W = sum(abs(float(w)) for w in usable.values()) or 1.0
    out = None
    for s, w in usable.items():
        col = pct_cols[s].astype(float)
        contrib = col * float(w) / W
        out = contrib if out is None else (out + contrib)

    return out.clip(0, 100)


###############

def _plotly_axes_black(fig):
    """
    Make all axis/label text black and grids subtle.
    Works across Plotly versions (uses title_font; falls back to title.font).
    """
    # Global font/legend to black
    fig.update_layout(
        font=dict(color="black"),
        legend=dict(font=dict(color="black"))
    )

    # X axis
    try:
        fig.update_xaxes(
            showline=True, linewidth=1, linecolor="black",
            tickfont=dict(color="black"),
            title_font=dict(color="black"),   # modern Plotly
            gridcolor="rgba(0,0,0,0.12)"
        )
    except Exception:
        # Older Plotly fallback
        fig.update_xaxes(
            showline=True, linewidth=1, linecolor="black",
            tickfont=dict(color="black"),
            title=dict(font=dict(color="black")),
            gridcolor="rgba(0,0,0,0.12)"
        )

    # Y axis
    try:
        fig.update_yaxes(
            showline=True, linewidth=1, linecolor="black",
            tickfont=dict(color="black"),
            title_font=dict(color="black"),
            gridcolor="rgba(0,0,0,0.12)"
        )
    except Exception:
        fig.update_yaxes(
            showline=True, linewidth=1, linecolor="black",
            tickfont=dict(color="black"),
            title=dict(font=dict(color="black")),
            gridcolor="rgba(0,0,0,0.12)"
        )

    # Keep hover label readable
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_color="black"))



# =========================
# ======= PIZZA (MPL) =====
# =========================
def mpl_pizza(player_row: pd.Series, stat_cols: List[str], title: str,
              light: bool = True, base_df: Optional[pd.DataFrame] = None) -> Optional[Figure]:

    if not _HAS_MPLSOCCER:
        st.error("Matplotlib / mplsoccer not available.")
        return None
    if player_row is None or not stat_cols:
        st.info("Pick a player and at least 1 stat.")
        return None

    base = base_df if base_df is not None and not base_df.empty else df_work

    pcts, raw_vals = [], []
    for s in stat_cols:
        if s not in base.columns or s not in df_work.columns:
            pcts.append(0.0); raw_vals.append(np.nan); continue
        series = pd.to_numeric(base[s], errors="coerce").replace([np.inf, -np.inf], np.nan)
        v = pd.to_numeric(player_row.get(s), errors="coerce")
        raw_vals.append(None if pd.isna(v) else float(v))

        if series.dropna().empty or pd.isna(v):
            pcts.append(0.0); continue

        if is_less_better(s):
            asc_series, asc_v = -series, -v
        else:
            asc_series, asc_v = series, v

        arr = asc_series.dropna().to_numpy()
        arr.sort()
        lo = np.searchsorted(arr, asc_v, side="left")
        hi = np.searchsorted(arr, asc_v, side="right")
        pct = ((lo + hi)/2) / arr.size * 100.0
        pcts.append(float(np.clip(pct, 0.0, 100.0)))

    slice_colors = ["#2E4374", "#1A78CF", "#D70232", "#FF9300", "#44C3A1",
                    "#CA228D", "#E1C340", "#7575A9", "#9DDFD3"] * 6
    slice_colors = slice_colors[:len(stat_cols)]
    text_colors  = ["#000000"] * len(slice_colors)  # ensure black text for readability
    params_disp  = [s for s in stat_cols]

    bg = POSTER_BG if light else "#222222"
    param_color = "#222222" if light else "#fffff0"
    value_txt_color = "#222222" if light else "#fffff0"

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
        if float(x).is_integer():
            return int(round(float(x)))
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

    # Header lines
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


# =========================
# ========= UI ============
# ========= UI =========

st.title("FM Analytics")

# ---- Upload in sidebar ----
# ---- Sidebar: Upload + helper downloads (no expanders, no duplicates) ----
st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader(
    "Upload FM HTML export (or CSV)",
    type=["html", "htm", "csv"],
    key="upload_main",  # unique key so we don't collide
)

st.sidebar.header("Downloads (FM helper files)")
for fname, label in [
    ("player search.fmf", "Download Player Search.fmf"),
    ("leagues filter.fmf", "Download leagues filter.fmf"),
]:
    fpath = os.path.join(APP_DIR, fname)
    if os.path.isfile(fpath):
        with open(fpath, "rb") as fh:
            st.sidebar.download_button(
                label,
                data=fh.read(),
                file_name=fname,
                mime="application/octet-stream",
                key=unique_key(f"dl_{re.sub(r'\\W+','_', fname)}"),
                help="Save into Documents\\Football Manager 2024\\Views (Player Search.fmf) or \\Filters (leagues filter.fmf).",
            )
    else:
        st.sidebar.caption(f"*{fname} not found in working directory.*")

@st.cache_data(show_spinner=True)
def parse_and_cache(name: str, raw: bytes) -> Tuple[pd.DataFrame, str]:
    if name.lower().endswith((".html", ".htm")):
        dfp = read_fm_html(io.BytesIO(raw))
    else:
        dfp = pd.read_csv(io.BytesIO(raw))
        dfp = _sanitize_df(dfp)
    dfp = apply_hard_remap(dfp)

    # build position tokens
    pos_col0 = find_col(dfp, ["Pos", "Position"])
    if pos_col0 is None:
        dfp["__pos_tokens"] = [[] for _ in range(len(dfp))]
    else:
        dfp["__pos_tokens"] = dfp[pos_col0].apply(expand_positions)

    # downcast + ensure floats are 2dp (already rounded in remap)
    for c in dfp.select_dtypes(include="float").columns:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce", downcast="float").round(2)
    for c in dfp.select_dtypes(include="integer").columns:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce", downcast="integer")

    cache_id = f"{hash((name, dfp.shape, tuple(dfp.columns)))}"
    return dfp, cache_id

# Parse on demand
df = pd.DataFrame()
cache_id = ""
if uploaded is not None:
    try:
        df, cache_id = parse_and_cache(uploaded.name, uploaded.read())
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# ---- If no data yet, show the Import Guide (plain, no expander) and stop ----
if df.empty:
    st.title("Import Guide")
    render_import_guide()
    st.stop()


# Make a copy we can mutate during session (e.g., custom metrics)
df_work = df.copy()
name_col = find_col(df_work, ["Name"]) or "Name"
pos_col  = find_col(df_work, ["Pos","Position"]) or "Pos"

# ==== TOP MODE SELECTOR (in main area) ====
MODES = [
    "Pizza & Archetypes",    # matplotlib pizza only
    "Percentile Bars",
    "Distribution",
    "Stat Scatter",
    "Role Scatter",
    "Top 10 — Roles",
    "Top 10 — Stats",
    "Player Finder",
    "Build: Metrics & Archetypes",
    "Table",
]
# Stable, single-click mode selector
if "mode" not in st.session_state:
    st.session_state["mode"] = MODES[0]
mode = st.radio("Mode", MODES, horizontal=True, key="mode")  # no index arg
st.write("")  # tiny spacer


# ==== SIDEBAR FILTERS (minutes, player search, role, baseline, positions) ====
st.sidebar.header("Filters")


# Minimum minutes slider
_min_col = find_col(df_work, ["Minutes","Mins","Min","Time Played"])
_mins_series = pd.to_numeric(df_work.get(_min_col, pd.Series(np.nan, index=df_work.index)), errors="coerce") if _min_col else pd.Series(np.nan, index=df_work.index)
min_possible = int(np.nanmin([0, _mins_series.min()])) if _min_col and _mins_series.notna().any() else 0
max_possible = int(np.nanmax([0, _mins_series.max()])) if _min_col and _mins_series.notna().any() else 0
default_minutes = min(900, max_possible) if max_possible > 0 else 0
min_minutes = st.sidebar.slider("Minimum Minutes", min_possible, max_possible, value=default_minutes, step=90)
if _min_col:
    df_work = df_work[pd.to_numeric(df_work[_min_col], errors="coerce").fillna(0) >= min_minutes].reset_index(drop=True)

# Re-derive name/pos columns after filtering
name_col = find_col(df_work, ["Name"]) or name_col
pos_col  = find_col(df_work, ["Pos","Position"]) or pos_col

# Build allowed/normalised tokens from df_work
present_norm = set()
for lst in df_work["__pos_tokens"]:
    if isinstance(lst, list):
        for t in lst:
            if isinstance(t, str):
                present_norm.add(_norm_token(t))
options_tokens = [tok for tok in ALLOWED_TOKENS if tok in present_norm] or ALLOWED_TOKENS

# Name search + dropdown (also search by transfer value string)
st.sidebar.subheader("Player")
_tv_field   = find_col(df_work, ["Transfer Value £m (mid)","Transfer Value"])  # optional
query = st.sidebar.text_input("Search (name or value)", "")
if query:
    q = query.strip().lower()
    def _match_row(r):
        nm = str(r.get(name_col, "")).lower()
        tv = str(r.get(_tv_field, "")).lower() if _tv_field and _tv_field in r else ""
        return (q in nm) or (q in tv)
    name_options = sorted(df_work.loc[df_work.apply(_match_row, axis=1), name_col].astype(str).unique().tolist())
else:
    name_options = sorted(df_work[name_col].dropna().astype(str).unique().tolist())
if not name_options:
    st.warning("No players match the current minutes filter.")
    st.stop()
player = st.sidebar.selectbox("Select player", name_options)
player_row = df_work[df_work[name_col] == player].iloc[0]

# Global active role (archetype) — used across modes
st.sidebar.subheader("Active Role")
_book_all = role_book_all()
if _book_all:
    active_role_default = list(_book_all.keys())[0]
    st.session_state["active_role"] = st.sidebar.selectbox(
        "Archetype / Role", list(_book_all.keys()),
        index=list(_book_all.keys()).index(st.session_state.get("active_role", active_role_default)) if st.session_state.get("active_role") in _book_all else 0
    )
else:
    st.sidebar.caption("No roles available.")
    st.session_state["active_role"] = None

# Percentiles baseline mode (affects pizzas, bars, role scores, etc.)
st.sidebar.subheader("Percentile Baseline")
# Default once
if "pct_baseline_mode" not in st.session_state:
    st.session_state["pct_baseline_mode"] = "Role baseline"

# Let Streamlit own the value; no index juggling, no manual assignment
st.sidebar.radio(
    "Compute percentiles against",
    ["Role baseline", "Sidebar positions", "Whole dataset"],
    key="pct_baseline_mode",
)


# Sidebar positional cohort for "Sidebar positions" mode
st.sidebar.subheader("Positions cohort")
raw_player_tokens = player_row["__pos_tokens"] if "__pos_tokens" in player_row else []
player_tokens_norm = sorted({_norm_token(t) for t in raw_player_tokens if isinstance(t, str)})
default_sel = [t for t in player_tokens_norm if t in options_tokens] or options_tokens
baseline_tokens = st.sidebar.multiselect(
    "Positions to include",
    options=options_tokens,
    default=default_sel,
    help="Used when 'Sidebar positions' baseline is selected."
)
BASELINE_DF = filter_by_tokens(df_work, baseline_tokens)

# =========================
# ========= MODES =========
# =========================

# ---------- Pizza & Archetypes (Matplotlib-only; unified role book) ----------
# ---------- Pizza & Archetypes (Matplotlib-only; unified role book) ----------
if mode == "Pizza & Archetypes":
    st.subheader("Pizza & Archetypes (Matplotlib only)")
    if not _HAS_MPLSOCCER:
        st.error(
            "Matplotlib / mplsoccer not available on this server. "
            "Install `matplotlib` and `mplsoccer`, or deploy with those packages."
        )
    else:
        choice = st.radio("What would you like to build?", ["Archetype pizza", "Custom pizza"], horizontal=True)

        # We no longer render any separate header/HTML outside the figure.
        # All title/subtitle lines are drawn INSIDE the Matplotlib figure by `mpl_pizza()`.

        if choice == "Archetype pizza":
            book = role_book_all()
            if not book:
                st.warning("No roles available.")
            else:
                roles_list = list(book.keys())
                default_idx = roles_list.index(st.session_state.get("active_role")) if st.session_state.get("active_role") in roles_list else 0
                role_name = st.selectbox("Role", roles_list, index=default_idx, key="pz_role")

                # Gather the numeric stats present in the dataset for this role (by weight keys)
                weights = role_weights_for(role_name) or {}
                stats = [s for s in weights.keys() if s in df_work.columns and pd.api.types.is_numeric_dtype(df_work[s])]
                if not stats:
                    st.info("This role has no valid stats in the current dataset.")
                else:
                    base_df = pick_baseline_df_for(role_name)
                    # Build subtitle that states the comparison baseline clearly
                    toks = role_baseline_tokens(role_name)
                    base_note = (
                        ", ".join(toks) if st.session_state.get("pct_baseline_mode") == "Role baseline" and toks
                        else (", ".join(baseline_tokens) if st.session_state.get("pct_baseline_mode") == "Sidebar positions"
                              else "Whole dataset")
                    )
                    subtitle = f"{role_name}, Percentiles Compared against {base_note}"

                    # Draw the pizza (all text is part of the figure; no extra markdown outside)
                    fig = mpl_pizza(player_row, stats, title=subtitle, light=True, base_df=base_df)
                    if fig is not None:
                        st.pyplot(fig, clear_figure=False)
                        st.download_button(
                            "Download pizza (PNG)",
                            data=fig_to_png_bytes(fig),
                            file_name=f"pizza_{str(player).replace(' ','_')}_{re.sub(r'\\W+','_',role_name)}.png",
                            mime="image/png",
                            key=unique_key("dl_pizza_role")
                        )

        else:  # Custom pizza
            all_cols = [c for c in df_work.columns if c not in {name_col, "Club", "League", "Pos", "__pos_tokens"}]
            default_stats = [x for x in [
                "Shots/90", "SoT/90", "Expected Goals", "Expected Assists/90", "Dribbles/90", "Open Play Key Passes/90"
            ] if x in all_cols]
            stats_pick = st.multiselect("Choose stats (ordered)", options=all_cols, default=default_stats, key="pz_custom_stats")
            stats_pick = [s for s in stats_pick if pd.api.types.is_numeric_dtype(df_work[s])]

            if stats_pick:
                base_df = pick_baseline_df_for(None)
                base_note = (
                    ", ".join(baseline_tokens) if st.session_state.get("pct_baseline_mode") == "Sidebar positions"
                    else ("Whole dataset" if st.session_state.get("pct_baseline_mode") == "Whole dataset" else ", ".join(baseline_tokens))
                )
                subtitle = f"Custom Pizza, Percentiles Compared against {base_note}"

                fig = mpl_pizza(player_row, stats_pick, title=subtitle, light=True, base_df=base_df)
                if fig is not None:
                    st.pyplot(fig, clear_figure=False)
                    st.download_button(
                        "Download pizza (PNG)",
                        data=fig_to_png_bytes(fig),
                        file_name=f"pizza_{str(player).replace(' ','_')}.png",
                        mime="image/png",
                        key=unique_key("dl_pizza_custom")
                    )
            else:
                st.info("Pick at least one numeric stat.")

# ---------- Percentile Bars (toggle: Archetype / Custom; dynamic colors; header) ----------
elif mode == "Percentile Bars":
    st.subheader("Percentile Bars")

    def _fmt_val(val) -> str:
        if pd.isna(val): return "—"
        try:
            x = float(val)
            return f"{int(round(x)):,}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}"
        except Exception:
            s = str(val)
            return s if len(s) <= 32 else s[:29] + "…"

    # Player header
    mins_col = find_col(df_work, ["Minutes","Mins","Min","Time Played"])
    apps_col = find_col(df_work, ["Appearances","Apps"])
    def _fmt_intish(v):
        try:
            f = float(pd.to_numeric(v, errors="coerce"))
            return f"{int(round(f)):,}"
        except Exception:
            return ""
    header_bits = [
        str(player_row.get(pos_col, "")) if pos_col in df_work.columns else "",
        str(player_row.get("Club", "")) if "Club" in df_work.columns else "",
        f"Minutes {_fmt_intish(player_row.get(mins_col,''))}" if mins_col else "",
        f"Apps {_fmt_intish(player_row.get(apps_col,''))}" if apps_col else "",
    ]
    header_line = f"{player_row.get(name_col,'')}"
    hb = [x for x in header_bits if x]
    if hb: header_line += " — " + " • ".join(hb)

    src = st.radio("Build from", ["Archetype", "Custom"], horizontal=True, key="pb_src")

    if src == "Archetype":
        book = role_book_all()
        if not book:
            st.info("No roles available."); st.stop()
        roles = list(book.keys())
        default_idx = roles.index(st.session_state.get("active_role")) if st.session_state.get("active_role") in roles else 0
        role_name = st.selectbox("Role", roles, index=default_idx, key="pb_role")
        weights = role_weights_for(role_name) or {}
        stats = [s for s in weights.keys() if s in df_work.columns and pd.api.types.is_numeric_dtype(df_work[s])]
        if not stats:
            st.info("This role has no valid numeric stats in the current dataset."); st.stop()
        base_df = pick_baseline_df_for(role_name)
        toks = role_baseline_tokens(role_name)
        base_note = ", ".join(toks) if st.session_state.get("pct_baseline_mode")=="Role baseline" and toks else (
            ", ".join(baseline_tokens) if st.session_state.get("pct_baseline_mode")=="Sidebar positions" else "Whole dataset"
        )
        subtitle_line = f"{role_name} — Percentiles Compared against {base_note}"
    else:
        all_cols = [c for c in df_work.columns if c not in {name_col, "Club", "League", "Pos", "__pos_tokens"}]
        default_stats = [x for x in [
            "Shots/90","SoT/90","Expected Goals","Expected Assists/90","Dribbles/90","Open Play Key Passes/90"
        ] if x in all_cols]
        stats = st.multiselect("Stats", options=all_cols, default=default_stats, key="pb_stats_custom")
        stats = [s for s in stats if pd.api.types.is_numeric_dtype(df_work[s])]
        if not stats:
            st.info("Pick at least one numeric stat."); st.stop()
        base_df = pick_baseline_df_for(None)
        base_note = ", ".join(baseline_tokens) if st.session_state.get("pct_baseline_mode")=="Sidebar positions" else (
            "Whole dataset" if st.session_state.get("pct_baseline_mode")=="Whole dataset" else ", ".join(baseline_tokens)
        )
        subtitle_line = f"Custom — Percentiles Compared against {base_note}"

    # Percentiles + raw values for selected player
    pcts, raw_vals_fmt = [], []
    for s in stats:
        pct_col = column_percentiles(df_work[s], base_df[s], is_less_better(s))
        pct_val = float(pct_col.get(player_row.name, np.nan))
        pcts.append(pct_val)
        raw_vals_fmt.append(_fmt_val(df_work.loc[player_row.name, s]))

    data = pd.DataFrame({"Stat": stats, "Percentile": np.clip(pcts, 0, 100), "Value": raw_vals_fmt}).sort_values("Percentile", ascending=False)

    # Dynamic red->green color by percentile + hover shows actual stat value
    fig = go.Figure(go.Bar(
        x=data["Percentile"],
        y=data["Stat"],
        orientation="h",
        customdata=np.stack([data["Value"]], axis=1),
        hovertemplate="<b>%{y}</b><br>Value: %{customdata[0]}<br>Percentile: %{x:.1f}<extra></extra>",
        marker=dict(
            color=data["Percentile"],
            cmin=0, cmax=100,
            colorscale="RdYlGn",
            line=dict(color="black", width=1)
        ),
    ))
    fig.update_layout(
        title=f"{header_line}<br><sup>{subtitle_line}</sup>",
        xaxis=dict(title="Percentile (0–100)", range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        template="simple_white",
        paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
        font=dict(family=FONT_FAMILY),
        hoverlabel=dict(bgcolor="white", font_color="black"),
    )
    _plotly_axes_black(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Distribution ----------
elif mode == "Distribution":
    st.subheader("Distribution")
    numeric_cols = [c for c in df_work.columns if pd.api.types.is_numeric_dtype(df_work[c])]
    # include numeric-like objects too
    for c in df_work.columns:
        if c not in numeric_cols and df_work[c].dtype == object:
            if pd.to_numeric(df_work[c], errors="coerce").notna().mean() > 0.55:
                numeric_cols.append(c)
    if not numeric_cols:
        st.info("No numeric columns available.")
    else:
        stat = st.selectbox("Stat", sorted(set(numeric_cols)))
        base_df = pick_baseline_df_for(st.session_state.get("active_role") if st.session_state.get("pct_baseline_mode")=="Role baseline" else None)
        series = pd.to_numeric(base_df[stat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        fig = px.histogram(series, nbins=40)
        fig.update_traces(marker_line_color="black", marker_line_width=1, hovertemplate="Count: %{y}<br>Bin: %{x}<extra></extra>")

        # Player marker: bold red line + annotation
        try:
            pval = float(pd.to_numeric(df_work.loc[player_row.name, stat], errors="coerce"))
            fig.add_vline(x=pval, line_color="#D70232", line_width=3)
            fig.add_annotation(x=pval, y=1.05, xref="x", yref="paper",
                               text=f"{player_row[name_col]} ({pval:.2f})",
                               showarrow=True, arrowhead=2, arrowcolor="#D70232",
                               bgcolor="white", bordercolor="#D70232")
        except Exception:
            pass

        fig.update_layout(
            title=f"Distribution of {stat} — Baseline: {'Role' if st.session_state.get('pct_baseline_mode')=='Role baseline' else ('Sidebar' if st.session_state.get('pct_baseline_mode')=='Sidebar positions' else 'Whole dataset')}",
            template="simple_white",
            paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
            font=dict(family=FONT_FAMILY),
            hoverlabel=dict(bgcolor="white", font_color="black"),
        )
        _plotly_axes_black(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Stat Scatter (always show points; highlight search) ----------
elif mode == "Stat Scatter":
    st.subheader("Stat Scatter")
    numeric_cols = [c for c in df_work.columns if pd.api.types.is_numeric_dtype(df_work[c])]
    for c in df_work.columns:
        if c not in numeric_cols and df_work[c].dtype == object:
            if pd.to_numeric(df_work[c], errors="coerce").notna().mean() > 0.55:
                numeric_cols.append(c)
    if not numeric_cols:
        st.info("No numeric columns found.")
    else:
        colx, coly = st.columns(2)
        with colx:
            xstat = st.selectbox("X stat", sorted(set(numeric_cols)))
        with coly:
            ystat = st.selectbox("Y stat", sorted(set(numeric_cols)), index=min(1, len(numeric_cols)-1))

        age_ok = "Age" in BASELINE_DF.columns
        u23 = st.checkbox("Highlight U23", value=False, disabled=not age_ok, key="stat_u23")
        u21 = st.checkbox("Highlight U21", value=False, disabled=not age_ok, key="stat_u21")
        hi_name = st.text_input("Highlight player (type to search)", "", key="stat_hi_q")

        df_disp = BASELINE_DF[[name_col]].copy()
        if "Club" in BASELINE_DF.columns: df_disp["Club"] = BASELINE_DF["Club"]
        if pos_col in BASELINE_DF.columns: df_disp["Pos"] = BASELINE_DF[pos_col]
        df_disp[xstat] = pd.to_numeric(BASELINE_DF[xstat], errors="coerce")
        df_disp[ystat] = pd.to_numeric(BASELINE_DF[ystat], errors="coerce")
        df_disp = df_disp.dropna(subset=[xstat, ystat])

        # Consistent hover on base layer
        hover_tmpl = "<b>%{hovertext}</b><br>Club: %{customdata[0]}<br>Pos: %{customdata[1]}<br>" \
                     + f"{xstat}: %{{x:.2f}}<br>{ystat}: %{{y:.2f}}<extra></extra>"
        fig = px.scatter(df_disp, x=xstat, y=ystat, hover_name=name_col,
                         hover_data={name_col:False, "Club":True, "Pos":True, xstat:":.2f", ystat:":.2f"},
                         opacity=0.9)
        for tr in fig.data:
            tr.marker.update(size=8, line=dict(width=1, color="black"))
            tr.update(hovertemplate=hover_tmpl)

        # overlays for age (visual only; keep base hover intact)
        if age_ok:
            ages = pd.to_numeric(BASELINE_DF["Age"], errors="coerce")
            if u23:
                cohort_ix = BASELINE_DF.index[ages < 23]
                cohort = df_disp.loc[df_disp.index.intersection(cohort_ix)]
                fig.add_trace(go.Scatter(
                    x=cohort[xstat], y=cohort[ystat], mode="markers",
                    marker=dict(size=14, color="rgba(0,135,68,0.25)", line=dict(width=2, color="rgba(0,135,68,0.9)")),
                    name="U23", hoverinfo="skip", showlegend=True
                ))
            if u21:
                cohort_ix = BASELINE_DF.index[ages < 21]
                cohort = df_disp.loc[df_disp.index.intersection(cohort_ix)]
                fig.add_trace(go.Scatter(
                    x=cohort[xstat], y=cohort[ystat], mode="markers",
                    marker=dict(size=16, color="rgba(88,24,173,0.25)", line=dict(width=2, color="rgba(88,24,173,0.95)")),
                    name="U21", hoverinfo="skip", showlegend=True
                ))

        # explicit player highlight by search
        if hi_name:
            prow = df_disp[df_disp[name_col].str.contains(hi_name, case=False, na=False)]
            if not prow.empty:
                prow = prow.iloc[[0]]
                fig.add_trace(go.Scatter(
                    x=prow[xstat], y=prow[ystat], mode="markers+text",
                    marker=dict(size=18, color="#D70232", line=dict(width=1, color="black")),
                    text=prow[name_col], textposition="middle right",
                    hoverinfo="skip", showlegend=False, name="Highlight"
                ))

        fig.update_layout(
            title=f"{ystat} vs {xstat}",
            template="simple_white",
            paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
            font=dict(family=FONT_FAMILY),
            hoverlabel=dict(bgcolor="white", font_color="black"),
        )
        _plotly_axes_black(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Role Scatter (safe role scoring) ----------
elif mode == "Role Scatter":
    st.subheader("Role Scatter (pos-aware; weighted composite of percentiles, 0–100)")

    # Safe local scorer in case older helper returns None
    def _safe_role_scores(role_name: str) -> pd.Series:
        weights = role_weights_for(role_name) or {}
        usable = {s: w for s, w in weights.items() if s in df_work.columns and pd.api.types.is_numeric_dtype(df_work[s])}
        if not usable:
            return pd.Series(np.nan, index=df_work.index)
        base_df = pick_baseline_df_for(role_name)
        W = sum(abs(float(w)) for w in usable.values()) or 1.0
        out = None
        for s, w in usable.items():
            pct = column_percentiles(df_work[s], base_df[s], is_less_better(s)).astype(float)
            contrib = pct * float(w) / W
            out = contrib if out is None else (out + contrib)
        return out.clip(0, 100)

    book = role_book_all()
    if not book:
        st.warning("No roles available.")
    else:
        names = list(book.keys())
        colx, coly = st.columns(2)
        default_x = names.index(st.session_state.get("active_role")) if st.session_state.get("active_role") in names else 0
        with colx: ax = st.selectbox("X role", names, index=default_x, key="role_x")
        with coly: ay = st.selectbox("Y role", names, index=min(1, len(names)-1), key="role_y")

        age_ok = "Age" in BASELINE_DF.columns
        u23 = st.checkbox("Highlight U23", value=False, disabled=not age_ok, key="role_u23")
        u21 = st.checkbox("Highlight U21", value=False, disabled=not age_ok, key="role_u21")
        hi_name = st.text_input("Highlight player (type to search)", "", key="role_hi_q")

        score_x_all = _safe_role_scores(ax)
        score_y_all = _safe_role_scores(ay)

        df_disp = BASELINE_DF[[name_col]].copy()
        if "Club" in BASELINE_DF.columns: df_disp["Club"] = BASELINE_DF["Club"]
        if pos_col in BASELINE_DF.columns: df_disp["Pos"] = BASELINE_DF[pos_col]
        df_disp[ax] = score_x_all.reindex(df_disp.index)
        df_disp[ay] = score_y_all.reindex(df_disp.index)
        df_disp = df_disp[df_disp[ax].notna() & df_disp[ay].notna()]

        fig = px.scatter(df_disp, x=ax, y=ay, hover_name=name_col,
                         hover_data={name_col:False, "Club":True, "Pos":True, ax:":.1f", ay:":.1f"},
                         opacity=0.95)
        for tr in fig.data:
            tr.marker.update(size=8, line=dict(width=1, color="black"))

        if age_ok:
            ages = pd.to_numeric(BASELINE_DF["Age"], errors="coerce")
            if u23:
                cohort_ix = BASELINE_DF.index[ages < 23]
                cohort = df_disp.loc[df_disp.index.intersection(cohort_ix)]
                fig.add_trace(go.Scatter(x=cohort[ax], y=cohort[ay], mode="markers",
                                         marker=dict(size=14, color="rgba(0,135,68,0.25)", line=dict(width=2, color="rgba(0,135,68,0.9)")),
                                         name="U23", hoverinfo="skip", showlegend=True))
            if u21:
                cohort_ix = BASELINE_DF.index[ages < 21]
                cohort = df_disp.loc[df_disp.index.intersection(cohort_ix)]
                fig.add_trace(go.Scatter(x=cohort[ax], y=cohort[ay], mode="markers",
                                         marker=dict(size=16, color="rgba(88,24,173,0.25)", line=dict(width=2, color="rgba(88,24,173,0.95)")),
                                         name="U21", hoverinfo="skip", showlegend=True))

        if hi_name:
            prow = df_disp[df_disp[name_col].str.contains(hi_name, case=False, na=False)]
            if not prow.empty:
                prow = prow.iloc[[0]]
                fig.add_trace(go.Scatter(
                    x=prow[ax], y=prow[ay], mode="markers+text",
                    marker=dict(size=18, color="#D70232", line=dict(width=1, color="black")),
                    text=prow[name_col], textposition="middle right",
                    hoverinfo="skip", showlegend=False, name="Highlight"
                ))

        fig.update_layout(
            title=f"Role Scores — {ay} vs {ax}",
            xaxis=dict(title=ax, range=[0,100]),
            yaxis=dict(title=ay, range=[0,100]),
            template="simple_white",
            paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
            font=dict(family=FONT_FAMILY),
            hoverlabel=dict(bgcolor="white", font_color="black"),
        )
        _plotly_axes_black(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Top 10 — Roles (dynamic color) ----------
elif mode == "Top 10 — Roles":
    st.subheader("Top 10 by Role Score (0–100)")
    def _safe_role_scores(role_name: str) -> pd.Series:
        weights = role_weights_for(role_name) or {}
        usable = {s: w for s, w in weights.items() if s in df_work.columns and pd.api.types.is_numeric_dtype(df_work[s])}
        if not usable:
            return pd.Series(np.nan, index=df_work.index)
        base_df = pick_baseline_df_for(role_name)
        W = sum(abs(float(w)) for w in usable.values()) or 1.0
        out = None
        for s, w in usable.items():
            pct = column_percentiles(df_work[s], base_df[s], is_less_better(s)).astype(float)
            contrib = pct * float(w) / W
            out = contrib if out is None else (out + contrib)
        return out.clip(0, 100)

    book = role_book_all()
    if not book:
        st.info("No roles available.")
    else:
        roles = list(book.keys())
        ridx = roles.index(st.session_state.get("active_role")) if st.session_state.get("active_role") in roles else 0
        role = st.selectbox("Role", roles, index=ridx)
        scores_all = _safe_role_scores(role).round(1)
        scores = scores_all.reindex(BASELINE_DF.index)

        tbl = BASELINE_DF[[name_col]].copy()
        if "Club" in BASELINE_DF.columns: tbl["Club"] = BASELINE_DF["Club"]
        if pos_col in BASELINE_DF.columns: tbl["Pos"] = BASELINE_DF[pos_col]
        tbl["Score"] = scores

        top = tbl.dropna(subset=["Score"]).sort_values("Score", ascending=False).head(10)

        fig = go.Figure(go.Bar(
            x=top["Score"], y=top[name_col], orientation="h",
            customdata=np.stack([top.get("Club", pd.Series([""]*len(top)))], axis=1),
            hovertemplate="<b>%{y}</b><br>Role Score: %{x:.1f}<br>Club: %{customdata[0]}<extra></extra>",
            marker=dict(color=top["Score"], cmin=0, cmax=100, colorscale="RdYlGn", line=dict(color="black", width=1))
        ))
        fig.update_layout(
            title=f"Top 10 — {role}",
            xaxis=dict(title="Role Score (0–100)", range=[0, 100]),
            yaxis=dict(autorange="reversed"),
            template="simple_white",
            paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
            font=dict(family=FONT_FAMILY),
            hoverlabel=dict(bgcolor="white", font_color="black"),
        )
        _plotly_axes_black(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Build: Metrics & Archetypes (custom metrics + custom/override roles) ----------
elif mode == "Build: Metrics & Archetypes":
    st.subheader("Create a custom metric")

    # Column choices
    numeric_like = [c for c in df_work.columns if pd.api.types.is_numeric_dtype(df_work[c])]
    for c in df_work.columns:
        if c not in numeric_like and df_work[c].dtype == object:
            if pd.to_numeric(df_work[c], errors="coerce").notna().mean() > 0.55:
                numeric_like.append(c)
    numeric_like = sorted(set(numeric_like))

    colA, colOp, colB = st.columns([3,1,3])
    with colA:
        a = st.selectbox("A", options=[None] + numeric_like, index=0, key="cm_a")
    with colOp:
        op = st.selectbox("Op", options=["+","-","*","/"], index=2, key="cm_op")
    with colB:
        b = st.selectbox("B", options=[None] + numeric_like, index=0, key="cm_b")

    mname = st.text_input("Metric name (new column)", "")
    lib = st.checkbox("Less is better for this metric", value=False, help="Affects percentile direction")

    c1, c2 = st.columns([1,1])
    with c1:
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
                    set_less_is_better(mname, lib)
                    st.success(f"Added metric '{mname}'.")
                except Exception as e:
                    st.error(f"Failed to add metric: {e}")
            else:
                st.warning("Please fill A, Op, B and a name.")
    with c2:
        if st.button("Remove metric column", key="btn_rm_metric"):
            if mname and mname in df_work.columns:
                df_work.drop(columns=[mname], inplace=True, errors="ignore")
                st.success(f"Removed column '{mname}'.")
            else:
                st.info("Enter an existing column name to remove.")

    st.markdown("---")
    st.subheader("Create / Edit Custom Role (unified: stat → weight)")

    # Custom role builder
    role_name_new = st.text_input("Role name (new or existing)")
    pick_stats = st.multiselect("Stats for this role", options=numeric_like, help="Choose stats; assign a weight to each.")

    weights_map = {}
    if pick_stats:
        cols_w = st.columns(min(4, len(pick_stats)))
        for i, s in enumerate(pick_stats):
            with cols_w[i % len(cols_w)]:
                weights_map[s] = st.number_input(f"{s}", min_value=0.0, value=1.0, step=0.1, key=f"rbw_{s}")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Save Custom Role", key="btn_save_custom_role"):
            if role_name_new and weights_map:
                st.session_state["role_book_custom"][role_name_new] = {
                    "baseline": None,  # uses sidebar cohort unless you override below
                    "weights": {k: float(v) for k, v in weights_map.items()},
                }
                st.success(f"Saved custom role '{role_name_new}'.")
            else:
                st.warning("Provide a role name and at least one stat.")
    with c2:
        if st.button("Delete Custom Role", key="btn_del_custom_role"):
            if role_name_new and role_name_new in st.session_state["role_book_custom"]:
                st.session_state["role_book_custom"].pop(role_name_new, None)
                st.success(f"Deleted custom role '{role_name_new}'.")
            else:
                st.info("Enter an existing custom role name to delete.")
    with c3:
        st.caption("Tip: This saves in session. Export/import below to persist.")

    # Override a built-in role (edit weights; keep built-in baseline)
    st.markdown("---")
    st.subheader("Override a Built-in Role (edit weights)")

    built_names = [k for k in ROLE_BOOK_BUILTIN.keys()]
    if built_names:
        target = st.selectbox("Built-in role to override", built_names, key="override_pick")
        base_cfg = ROLE_BOOK_BUILTIN.get(target, {})
        base_weights = dict(base_cfg.get("weights", {}))
        if not base_weights:
            st.info("This role has no weight map; nothing to override.")
        else:
            cols = st.columns(min(4, len(base_weights)))
            new_w = {}
            for i, (stat, w) in enumerate(sorted(base_weights.items())):
                with cols[i % len(cols)]:
                    new_w[stat] = st.number_input(stat, min_value=0.0, value=float(w), step=0.1, key=f"ov_{target}_{stat}")
            if st.button("Save override", key="btn_save_override"):
                st.session_state["role_book_custom"][target] = {
                    "baseline": base_cfg.get("baseline", None),
                    "weights": {k: float(v) for k, v in new_w.items()},
                }
                st.success(f"Saved override for '{target}'.")
    else:
        st.caption("No built-in roles found.")

    # Export / Import all custom roles
    st.markdown("---")
    st.subheader("Export / Import Custom Roles")

    exp_payload = {"roles": st.session_state["role_book_custom"]}
    st.download_button(
        "Download custom roles JSON",
        data=json.dumps(exp_payload, indent=2).encode("utf-8"),
        file_name="custom_roles.json",
        mime="application/json",
        key=unique_key("dl_custom_roles")
    )

    up = st.file_uploader("Upload custom roles JSON", type=["json"], key="ul_custom_roles")
    if up is not None:
        try:
            data = json.loads(up.read().decode("utf-8"))
            if isinstance(data, dict) and "roles" in data and isinstance(data["roles"], dict):
                # coerce shapes
                cleaned = {}
                for rname, cfg in data["roles"].items():
                    if not isinstance(cfg, dict): continue
                    w = cfg.get("weights", {})
                    if isinstance(w, dict) and w:
                        cleaned[rname] = {
                            "baseline": cfg.get("baseline", None),
                            "weights": {str(s): float(wv) for s, wv in w.items()},
                        }
                st.session_state["role_book_custom"].update(cleaned)
                st.success(f"Imported {len(cleaned)} custom role(s).")
            else:
                st.error("JSON must have a top-level 'roles' object.")
        except Exception as e:
            st.error(f"Import failed: {e}")


# ---------- Top 10 — Stats (percentile leaders; dynamic color; hover shows value 2dp) ----------
elif mode == "Top 10 — Stats":
    st.subheader("Top 10 — Stats (by percentile)")
    numeric_cols = [c for c in df_work.columns if pd.api.types.is_numeric_dtype(df_work[c])]
    for c in df_work.columns:
        if c not in numeric_cols and df_work[c].dtype == object:
            if pd.to_numeric(df_work[c], errors="coerce").notna().mean() > 0.55:
                numeric_cols.append(c)
    if not numeric_cols:
        st.info("No numeric columns available.")
    else:
        stat = st.selectbox("Stat", sorted(set(numeric_cols)))
        base_df = pick_baseline_df_for(None)
        pct_all = column_percentiles(df_work[stat], base_df[stat], is_less_better(stat)).round(1)
        pct = pct_all.reindex(BASELINE_DF.index)

        tbl = BASELINE_DF[[name_col]].copy()
        if "Club" in BASELINE_DF.columns: tbl["Club"] = BASELINE_DF["Club"]
        if pos_col in BASELINE_DF.columns: tbl["Pos"] = BASELINE_DF[pos_col]
        tbl["Value"] = pd.to_numeric(BASELINE_DF[stat], errors="coerce")
        tbl["Percentile"] = pct

        top = tbl.dropna(subset=["Percentile"]).sort_values("Percentile", ascending=False).head(10)

        customdata = np.stack([top.get("Club", pd.Series([""]*len(top))), top["Value"]], axis=1)

        fig = go.Figure(go.Bar(
            x=top["Percentile"], y=top[name_col], orientation="h",
            customdata=customdata,
            hovertemplate="<b>%{y}</b><br>Value: %{customdata[1]:.2f}<br>Percentile: %{x:.1f}<br>Club: %{customdata[0]}<extra></extra>",
            marker=dict(color=top["Percentile"], cmin=0, cmax=100, colorscale="RdYlGn", line=dict(color="black", width=1))
        ))
        fig.update_layout(
            title=f"Top 10 — {stat} (by percentile)",
            xaxis=dict(title="Percentile (0–100)", range=[0, 100]),
            yaxis=dict(autorange="reversed"),
            template="simple_white",
            paper_bgcolor=POSTER_BG, plot_bgcolor=POSTER_BG,
            font=dict(family=FONT_FAMILY),
            hoverlabel=dict(bgcolor="white", font_color="black"),
        )
        _plotly_axes_black(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Player Finder (rich filters + pills; no Text filter) ----------
# ---------- Player Finder (values, percentiles, categories, positions) ----------
# ---------- Player Finder (values, percentiles, categories, positions) ----------
# ---------- Player Finder (rich filters + pills; no Text filter) ----------
# ---------- Player Finder (filters; cards show pills from your chosen filter stats) ----------
elif mode == "Player Finder":
    if "saved_filters" not in st.session_state:
        st.session_state["saved_filters"] = {}

    st.subheader("Player Finder (values, percentiles, categories, positions)")

    def _fmt_num(x):
        try:
            if pd.isna(x): return "—"
            xf = float(x)
            if abs(xf - round(xf)) < 1e-9:
                return f"{int(round(xf)):,}"
            return f"{xf:.2f}"
        except Exception:
            return str(x)

    def _unique_vals(col: str, maxn: int = 200) -> list:
        try:
            return (
                df_work[col]
                .dropna()
                .astype(str)
                .value_counts()
                .nlargest(maxn)
                .index.astype(str)
                .tolist()
            )
        except Exception:
            return []

    def _ensure_percentile(colname: str) -> pd.Series:
        return column_percentiles(df_work[colname], BASELINE_DF[colname], is_less_better(colname)).round(1)

    # Presets row (save / export / import)
    st.markdown("#### Presets")
    pcol1, pcol2, pcol3, pcol4 = st.columns([3,3,2,2])
    with pcol1:
        preset_name = st.text_input("Save current filters as", key="pf_preset_name")
    with pcol2:
        if st.button("Save preset", key="pf_save_preset"):
            if preset_name:
                st.session_state["saved_filters"][preset_name] = {
                    "combine": st.session_state.get("pf_combine", "ALL"),
                    "max_show": st.session_state.get("pf_max_show", 100),
                    "rows": st.session_state.get("pf_rows_payload", []),
                }
                st.success(f"Saved preset “{preset_name}”.")
            else:
                st.warning("Give your preset a name.")
    with pcol3:
        if st.session_state["saved_filters"]:
            exp = json.dumps(st.session_state["saved_filters"], indent=2).encode("utf-8")
            st.download_button("Export presets", data=exp, file_name="player_finder_presets.json",
                               mime="application/json", key=unique_key("dl_pf_presets"))
    with pcol4:
        imp = st.file_uploader("Import presets", type=["json"], key="pf_import_presets")
        if imp is not None:
            try:
                data = json.loads(imp.read().decode("utf-8"))
                if isinstance(data, dict):
                    st.session_state["saved_filters"].update(data)
                    st.success(f"Imported {len(data)} preset(s).")
                else:
                    st.error("JSON root must be an object (dict).")
            except Exception as e:
                st.error(f"Import failed: {e}")

    if st.session_state["saved_filters"]:
        lcol1, lcol2, lcol3 = st.columns([4,2,2])
        with lcol1:
            chosen_preset = st.selectbox("Load preset", list(st.session_state["saved_filters"].keys()), key="pf_load_choice")
        with lcol2:
            use_loaded = st.checkbox("Use loaded preset for Apply", value=False, key="pf_use_loaded")
        with lcol3:
            if st.button("Delete preset", key="pf_del_preset"):
                if chosen_preset in st.session_state["saved_filters"]:
                    st.session_state["saved_filters"].pop(chosen_preset, None)
                    st.success("Deleted preset.")

    st.write("---")

    # Builder
    st.markdown("#### Build filters")
    combine_logic = st.radio("Combine filters with", ["ALL", "ANY"], horizontal=True, key="pf_combine")
    max_show = st.number_input("Max results to show", min_value=10, max_value=10000, value=100, step=10, key="pf_max_show")

    # Column lists for builders
    numeric_cols_all = [c for c in df_work.columns if pd.api.types.is_numeric_dtype(df_work[c])]
    for c in df_work.columns:
        if c not in numeric_cols_all and df_work[c].dtype == object:
            s_clean = pd.to_numeric(df_work[c], errors="coerce")
            if (s_clean.notna().mean() > 0.55):
                numeric_cols_all.append(c)
    numeric_cols_all = sorted(set(numeric_cols_all))

    text_cols_all = sorted([c for c in df_work.columns if df_work[c].dtype == object and c not in {"__pos_tokens"}])
    category_cols_suggested = [c for c in text_cols_all if df_work[c].nunique(dropna=True) <= 80]

    max_rows = 10
    ncrit = st.slider("Number of rows", 1, max_rows, 3, key="pf_nrows")
    rows_payload = []

    for i in range(ncrit):
        st.markdown(f"**Row {i+1}**")
        t1, t2, t3, t4 = st.columns([2.3, 3.2, 2.2, 4.5])

        with t1:
            kind = st.selectbox(
                "Type",
                ["Numeric (value)", "Numeric (percentile)", "Category", "Position includes"],
                key=f"pf_kind_{i}"
            )

        if kind == "Position includes":
            with t2:
                toks = st.multiselect("Tokens", options=ALLOWED_TOKENS, key=f"pf_pos_tokens_{i}")
            rows_payload.append({"kind": "pos_includes", "tokens": toks})
            st.write("")
            continue

        if kind == "Category":
            with t2:
                col = st.selectbox("Column", options=category_cols_suggested or text_cols_all, key=f"pf_cat_col_{i}")
            with t3:
                op = st.selectbox("Op", ["in", "not in"], key=f"pf_cat_op_{i}")
            with t4:
                opts = _unique_vals(col, 200)
                vals = st.multiselect("Values", options=opts, key=f"pf_cat_vals_{i}")
            rows_payload.append({"kind":"category","col":col,"op":op,"vals":vals})
            continue

        if kind == "Numeric (percentile)":
            with t2:
                col = st.selectbox("Column", options=numeric_cols_all, key=f"pf_pct_col_{i}")
            with t3:
                op = st.selectbox("Op", [">=", "<=", "between"], key=f"pf_pct_op_{i}")
            with t4:
                if op == "between":
                    c1, c2 = st.columns(2)
                    with c1: lo = st.number_input("Pctl min", min_value=0.0, max_value=100.0, value=60.0, step=1.0, key=f"pf_pct_lo_{i}")
                    with c2: hi = st.number_input("Pctl max", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key=f"pf_pct_hi_{i}")
                    thr = [float(lo), float(hi)]
                else:
                    thr = st.number_input("Threshold (0–100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key=f"pf_pct_thr_{i}")
            rows_payload.append({"kind":"num_pct","col":col,"op":op,"thr":thr})
        else:
            with t2:
                col = st.selectbox("Column", options=numeric_cols_all, key=f"pf_num_col_{i}")
            with t3:
                op = st.selectbox("Op", [">=", "<=", "==", "between"], key=f"pf_num_op_{i}")
            with t4:
                if op == "between":
                    c1, c2 = st.columns(2)
                    with c1: lo = st.number_input("Min", value=0.0, step=0.1, key=f"pf_num_lo_{i}")
                    with c2: hi = st.number_input("Max", value=0.0, step=0.1, key=f"pf_num_hi_{i}")
                    thr = [float(lo), float(hi)]
                else:
                    thr = st.number_input("Threshold", value=0.0, step=0.1, key=f"pf_num_thr_{i}")
            rows_payload.append({"kind":"num_val","col":col,"op":op,"thr":thr})

    st.session_state["pf_rows_payload"] = rows_payload

    st.write("---")
    apply_cols = st.columns([1,1,5,3])
    with apply_cols[0]:
        apply_now = st.button("Apply filters", type="primary")
    with apply_cols[1]:
        reset_sel = st.button("Reset selection")
    with apply_cols[3]:
        st.caption("Tip: Combine with the sidebar **positions** and **minimum minutes** filter for cleaner cohorts.")

    if reset_sel:
        st.experimental_rerun()

    # Build pills automatically from user-chosen numeric filter columns (order preserved, max 3)
    def _derive_pill_fields(payload: list) -> list:
        fields = []
        for r in payload:
            if r.get("kind") in ("num_val", "num_pct"):
                c = r.get("col")
                if c and c not in fields:
                    fields.append(c)
            if len(fields) >= 3:
                break
        # fallback if none found
        if not fields:
            candidates = ["Goals / 90", "Assists/90", "Expected Goals/90", "Shots/90", "SoT/90"]
            fields = [c for c in candidates if c in df_work.columns][:3]
        return fields

    def _mask_from_filters(payload: list) -> pd.Series:
        if not payload:
            return pd.Series(True, index=df_work.index)
        masks = []
        for flt in payload:
            kind = flt.get("kind")
            if kind == "pos_includes":
                toks = set(flt.get("tokens", []))
                if not toks:
                    masks.append(pd.Series(True, index=df_work.index)); continue
                def _has_tok(lst):
                    if not isinstance(lst, list): return False
                    normed = {_norm_token(t) for t in lst if isinstance(t, str)}
                    return bool(toks.intersection(normed))
                m = df_work["__pos_tokens"].apply(_has_tok)
                masks.append(m.fillna(False)); continue
            if kind == "category":
                col = flt.get("col"); op = flt.get("op", "in")
                vals = set(map(str, flt.get("vals", [])))
                ser = df_work[col].astype(str)
                m = ser.isin(vals)
                if op == "not in": m = ~m
                masks.append(m.fillna(False)); continue
            if kind == "num_pct":
                col = flt.get("col"); op = flt.get("op"); thr = flt.get("thr")
                pct = _ensure_percentile(col)
                if op == ">=": m = pct >= float(thr)
                elif op == "<=": m = pct <= float(thr)
                else:
                    lo, hi = float(thr[0]), float(thr[1])
                    if lo > hi: lo, hi = hi, lo
                    m = pct.between(lo, hi, inclusive="both")
                masks.append(m.fillna(False)); continue
            if kind == "num_val":
                col = flt.get("col"); op = flt.get("op"); thr = flt.get("thr")
                ser = pd.to_numeric(df_work[col], errors="coerce")
                if op == ">=": m = ser >= float(thr)
                elif op == "<=": m = ser <= float(thr)
                elif op == "==": m = ser == float(thr)
                else:
                    lo, hi = float(thr[0]), float(thr[1])
                    if lo > hi: lo, hi = hi, lo
                    m = ser.between(lo, hi, inclusive="both")
                masks.append(m.fillna(False)); continue
        if not masks:
            return pd.Series(True, index=df_work.index)
        out = masks[0].copy()
        if combine_logic == "ALL":
            for m in masks[1:]: out &= m
        else:
            for m in masks[1:]: out |= m
        return out

    # Apply either current builder or loaded preset
    apply_loaded = st.session_state.get("pf_use_loaded") and st.session_state.get("pf_load_choice")
    if apply_now or apply_loaded:
        payload_to_apply = rows_payload
        if apply_loaded:
            preset_key = st.session_state.get("pf_load_choice")
            preset = st.session_state["saved_filters"].get(preset_key, {})
            payload_to_apply = preset.get("rows", [])
            # restore combine / max_show if present
            combine_logic = preset.get("combine", combine_logic)
            max_show = int(preset.get("max_show", max_show))

        mask_all = _mask_from_filters(payload_to_apply)
        res = df_work.loc[mask_all].copy()

        # Decide which fields to show as pills (from filters)
        pill_fields = _derive_pill_fields(payload_to_apply)

        # Card styles
        st.markdown("""
<style>
.card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
.pcard { background: #fff; border: 1px solid #000; border-radius: 14px; padding: 12px 12px 10px; }
.pcard h4 { margin: 0 0 4px 0; font-size: 1.0rem; line-height: 1.2; color: #000; }
.pcard .sub { color: #333; font-size: .86rem; margin-bottom: 6px; }
.pcard .kv { color: #111; font-size: .88rem; }
.pcard .pill { display: inline-block; margin: 4px 6px 0 0; padding: 2px 6px; border: 1px solid #000; border-radius: 999px; font-size: .78rem; color: #000; background: #f7f7f7; }
</style>
""", unsafe_allow_html=True)

        cards_html = []
        for _, r in res.iterrows():
            nm   = str(r.get(name_col, ""))
            club = str(r.get("Club", "")) if "Club" in res.columns else ""
            pos  = str(r.get(pos_col, "")) if pos_col in res.columns else ""
            age  = _fmt_num(r.get("Age", np.nan))
            mins = _fmt_num(r.get("Minutes", np.nan) if "Minutes" in res.columns else r.get("Mins", np.nan))
            rat  = _fmt_num(r.get("Avg Rating", np.nan))
            tvm  = r.get("Transfer Value £m (mid)", np.nan)
            tvtxt = f"£{float(tvm):.2f}m" if pd.notna(tvm) else (str(r.get("Transfer Value", "")) if "Transfer Value" in res.columns else "—")

            # key-value items
            items = []
            if age != "—": items.append(f"<b>Age:</b> {age}")
            if mins != "—": items.append(f"<b>Min:</b> {mins}")
            if rat != "—": items.append(f"<b>Rating:</b> {rat}")

            # Pills (derived from filters)
            pill_htmls = []
            for col_name in pill_fields:
                val = r.get(col_name, np.nan)
                try:
                    if pd.isna(val):
                        txt = "—"
                    else:
                        fval = float(pd.to_numeric(val, errors="coerce"))
                        txt = f"{fval:.2f}" if abs(fval - round(fval)) >= 1e-9 else f"{int(round(fval)):,}"
                except Exception:
                    txt = str(val)
                pill_htmls.append(f"<span class='pill'>{col_name}: {txt}</span>")
            pills_str = "".join(pill_htmls)

            # SAFE builder (no implicit concat)
            parts = []
            parts.append("<div class='pcard'>")
            parts.append(f"<h4>{nm}</h4>")
            parts.append(f"<div class='sub'>{club} &nbsp;•&nbsp; {pos}</div>")
            if items:
                parts.append(f"<div class='kv'>{' &nbsp; '.join(items)}</div>")
            parts.append(f"<div class='kv' style='margin-top:4px;'><b>Transfer Value:</b> {tvtxt}</div>")
            if pills_str:
                parts.append(f"<div class='metric-ct'>{pills_str}</div>")
            parts.append("</div>")
            card_html = "".join(parts)

            cards_html.append(card_html)

        total_matches = len(cards_html)
        cards_html = cards_html[: int(max_show)]

        if cards_html:
            body = "".join(cards_html)
            body = re.sub(r">\s+<", "><", body).strip()
            html_grid = f"<div class='card-grid'>{body}</div>"
            st.markdown(html_grid, unsafe_allow_html=True)
            st.caption(f"Showing {len(cards_html):,} of {total_matches:,} matches.")
        else:
            st.info("No matches.")

        st.download_button(
            "Download CSV (all matches)",
            data=res.to_csv(index=False).encode("utf-8"),
            file_name="player_finder_results.csv",
            mime="text/csv",
            key=unique_key("dl_pf_results_csv")
        )

# ---------- Table ----------
elif mode == "Table":
    st.subheader("Cleaned data (floats rounded to 2 dp)")
    st.dataframe(df_work.drop(columns=["__pos_tokens"]), use_container_width=True)
    st.caption(f"Rows: {len(df_work):,} • Columns: {df_work.drop(columns=['__pos_tokens']).shape[1]}")
