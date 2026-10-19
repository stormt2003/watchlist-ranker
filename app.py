"""
Watchlist Ranker — Streamlit-app
--------------------------------
Live koersdata via yfinance, MA's (20w/50w/200w/200d) lokaal berekend,
score-ranglijst gesplitst in Crypto en Aandelen/ETF's, plus de
Crypto Fear & Greed-gauge.

Lokaal draaien:
    pip install -r requirements.txt
    streamlit run app.py
"""

import math

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = [
    "VWRL.AS", "AMZN", "TSLA", "MU",
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
]

DEFAULT_WEIGHTS = {"20w": 1.0, "50w": 1.5, "200w": 2.0, "200d": 1.5}

# Badge-symbolen (geen externe logo's). Onbekende tickers krijgen een monogram.
BADGES = {"BTC-USD": "₿", "ETH-USD": "Ξ", "SOL-USD": "◎", "ADA-USD": "₳"}

CRYPTO_SUFFIXES = ("-USD", "-EUR", "-USDT")

BAR_CAP_PCT = 50  # balk is "vol" bij ±50% afstand tot de MA

st.set_page_config(page_title="Watchlist Ranker", page_icon="📉", layout="centered")


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily_closes(ticker: str) -> pd.Series:
    """Dagelijkse closes (5 jaar). Ticker.history geeft altijd platte kolommen,
    in tegenstelling tot yf.download (die recent MultiIndex-kolommen teruggeeft)."""
    hist = yf.Ticker(ticker).history(period="5y", interval="1d", auto_adjust=True)
    if hist.empty:
        raise ValueError("geen data (verkeerde ticker of Yahoo onbereikbaar)")
    closes = hist["Close"].dropna()
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    return closes


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_crypto_fear_greed() -> dict:
    resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
    resp.raise_for_status()
    data = resp.json()["data"][0]
    return {"waarde": int(data["value"]), "classificatie": data["value_classification"]}


def is_crypto(ticker: str) -> bool:
    return ticker.upper().endswith(CRYPTO_SUFFIXES)


def compute_row(ticker: str, weights: dict) -> dict:
    daily = fetch_daily_closes(ticker)
    weekly = daily.resample("W").last().dropna()  # laatste bar = lopende week
    price = float(daily.iloc[-1])

    mas = {
        "20w": weekly.rolling(20).mean().iloc[-1],
        "50w": weekly.rolling(50).mean().iloc[-1],
        "200w": weekly.rolling(200).mean().iloc[-1],
        "200d": daily.rolling(200).mean().iloc[-1],
    }
    pcts = {
        k: None if pd.isna(v) else round((price - float(v)) / float(v) * 100, 2)
        for k, v in mas.items()
    }
    # Lager onder de MA = hogere score. Ontbrekende MA's tellen niet mee.
    score = sum(-p * weights[k] for k, p in pcts.items() if p is not None)

    return {
        "ticker": ticker,
        "prijs": price,
        "datum": daily.index[-1].date(),
        **{f"%_vs_{k}": p for k, p in pcts.items()},
        "score": round(score, 1),
    }


def build_ranking(tickers: list, weights: dict):
    rows, errors = [], []
    for t in tickers:
        try:
            rows.append(compute_row(t, weights))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{t}: {e}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df, errors


# ---------------------------------------------------------------------------
# UI-ONDERDELEN (inline SVG/HTML, geen externe assets)
# ---------------------------------------------------------------------------

ZONES = [  # (van, tot, naam, kleur)
    (0, 25, "Extreme Fear", "#d64545"),
    (25, 45, "Fear", "#ee8a35"),
    (45, 55, "Neutral", "#e9c440"),
    (55, 75, "Greed", "#9acd5c"),
    (75, 100, "Extreme Greed", "#2e9e58"),
]


def _polar(cx, cy, r, value):
    angle = math.radians(180 - value / 100 * 180)
    return cx + r * math.cos(angle), cy - r * math.sin(angle)


def gauge_svg(value, title: str) -> str:
    cx, cy, r, w = 110, 105, 85, 20
    gap = 0.8
    parts = []
    for lo, hi, name, color in ZONES:
        a0, a1 = lo + (gap if lo > 0 else 0), hi - (gap if hi < 100 else 0)
        x0, y0 = _polar(cx, cy, r, a0)
        x1, y1 = _polar(cx, cy, r, a1)
        parts.append(
            f'<path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 0 1 {x1:.1f},{y1:.1f}" '
            f'stroke="{color}" stroke-width="{w}" fill="none"/>'
        )
        tx, ty = _polar(cx, cy, r - w - 6, (lo + hi) / 2)
        parts.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="6.5" text-anchor="middle" '
            f'fill="currentColor" opacity="0.7">{name}</text>'
        )
    if value is not None:
        nx, ny = _polar(cx, cy, r - 4, value)
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" '
            f'stroke="currentColor" stroke-width="3" stroke-linecap="round"/>'
        )
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="currentColor"/>')
        label = f"{value}"
    else:
        label = "n.v.t."
    parts.append(
        f'<text x="{cx}" y="{cy + 22}" font-size="18" font-weight="600" '
        f'text-anchor="middle" fill="currentColor">{label}</text>'
    )
    return (
        f'<div style="text-align:center">'
        f'<div style="font-size:0.9rem;opacity:0.8">{title}</div>'
        f'<svg viewBox="0 0 220 135" style="width:100%;max-width:280px">'
        f'{"".join(parts)}</svg></div>'
    )


def badge_html(ticker: str, crypto: bool) -> str:
    sym = BADGES.get(ticker.upper(), ticker.split("-")[0].split(".")[0][:2].upper())
    bg = "#7b5cd6" if crypto else "#3a7bd5"
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:30px;height:30px;border-radius:50%;background:{bg};color:#fff;'
        f'font-weight:700;font-size:0.8rem;flex-shrink:0">{sym}</span>'
    )


def ma_bar_html(label: str, pct) -> str:
    if pct is None:
        fill, txt = "", "te weinig historie"
    else:
        width = min(abs(pct), BAR_CAP_PCT) / BAR_CAP_PCT * 50
        side = f"right:50%" if pct < 0 else "left:50%"
        color = "#d64545" if pct < 0 else "#2e9e58"
        fill = (
            f'<div style="position:absolute;top:0;bottom:0;{side};width:{width:.1f}%;'
            f'background:{color};border-radius:3px"></div>'
        )
        txt = f"{pct:+.1f}%"
    return (
        f'<div style="display:flex;align-items:center;gap:8px;font-size:0.78rem;margin:3px 0">'
        f'<span style="width:34px;opacity:0.75">{label}</span>'
        f'<div style="position:relative;flex:1;height:8px;background:rgba(128,128,128,0.18);border-radius:3px">'
        f'{fill}<div style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;'
        f'background:currentColor;opacity:0.5"></div></div>'
        f'<span style="width:64px;text-align:right;font-variant-numeric:tabular-nums">{txt}</span>'
        f"</div>"
    )


def card_html(rank: int, row: pd.Series) -> str:
    crypto = is_crypto(row["ticker"])
    bars = "".join(ma_bar_html(k, row[f"%_vs_{k}"]) for k in DEFAULT_WEIGHTS)
    return (
        f'<div style="border:1px solid rgba(128,128,128,0.25);border-radius:10px;'
        f'padding:10px 12px;margin-bottom:10px">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
        f'<span style="opacity:0.6;width:18px">{rank}</span>'
        f"{badge_html(row['ticker'], crypto)}"
        f'<div style="flex:1"><div style="font-weight:600">{row["ticker"]}</div>'
        f'<div style="font-size:0.75rem;opacity:0.65">{row["prijs"]:,.2f} op {row["datum"]}</div></div>'
        f'<div style="text-align:right"><div style="font-size:0.7rem;opacity:0.65">score</div>'
        f'<div style="font-weight:700;font-variant-numeric:tabular-nums">{row["score"]:+.1f}</div></div>'
        f"</div>{bars}</div>"
    )


# ---------------------------------------------------------------------------
# PAGINA
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Instellingen")
    tickers_text = st.text_area(
        "Tickers (één per regel)", "\n".join(DEFAULT_TICKERS), height=200
    )
    st.caption("Crypto in yfinance-formaat, bv. BTC-USD.")
    st.subheader("Gewichten")
    weights = {
        k: st.slider(f"{k} MA", 0.0, 3.0, v, 0.25) for k, v in DEFAULT_WEIGHTS.items()
    }
    if st.button("Data verversen"):
        st.cache_data.clear()

tickers = [t.strip().upper() for t in tickers_text.splitlines() if t.strip()]

st.title("Watchlist Ranker")

# Fear & Greed
col1, col2 = st.columns(2)
try:
    fng = fetch_crypto_fear_greed()
    col1.markdown(gauge_svg(fng["waarde"], "Crypto Fear & Greed"), unsafe_allow_html=True)
except Exception as e:  # noqa: BLE001
    col1.markdown(gauge_svg(None, "Crypto Fear & Greed"), unsafe_allow_html=True)
    col1.caption(f"Niet opgehaald: {e}")
col2.markdown(gauge_svg(None, "Aandelen Fear & Greed"), unsafe_allow_html=True)
col2.caption("Nog niet gekoppeld (scraping nodig).")

# Ranglijst
with st.spinner("Koersen ophalen…"):
    ranking, errors = build_ranking(tickers, weights)

if errors:
    st.warning("Niet verwerkt:\n\n" + "\n\n".join(errors))

if ranking.empty:
    st.error(
        "Geen koersdata opgehaald. Controleer de tickers, of Yahoo Finance "
        "geblokkeerd wordt door je netwerk (bijv. een bedrijfsproxy)."
    )
    st.stop()

st.caption("Hoogste score = verste onder de MA's. Balk links (rood) = onder de MA.")

for section, mask in [
    ("Crypto", ranking["ticker"].map(is_crypto)),
    ("Aandelen & ETF's", ~ranking["ticker"].map(is_crypto)),
]:
    part = ranking[mask].reset_index(drop=True)
    if part.empty:
        continue
    st.subheader(section)
    st.markdown(
        "".join(card_html(i + 1, row) for i, row in part.iterrows()),
        unsafe_allow_html=True,
    )

with st.expander("Ruwe tabel"):
    st.dataframe(ranking, width="stretch", hide_index=True)
