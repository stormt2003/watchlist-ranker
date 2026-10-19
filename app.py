"""
Watchlist Ranker — Streamlit-app
--------------------------------
Live koersdata via yfinance, MA's (20w/50w/200w/200d) lokaal berekend,
score-ranglijst gesplitst in Aandelen/ETF's (links) en Crypto (rechts),
elk met de bijbehorende Fear & Greed-gauge erboven.

Lokaal draaien:
    pip install -r requirements.txt
    streamlit run app.py
"""

import math
from datetime import timedelta

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = [
    # Aandelen & ETF's
    "VWRL.AS", "AMZN", "ACHR", "BESI.AS", "BSX", "MU", "NFLX", "MSTR", "TSLA", "PATH",
    # Crypto (Yahoo gebruikt bij naamconflicten een nummer, bv. HYPE32196-USD)
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "HYPE32196-USD",
    "NEAR-USD", "AAVE-USD", "ONDO-USD", "INJ-USD", "VIRTUAL-USD",
]

DEFAULT_WEIGHTS = {"20w": 1.0, "50w": 1.5, "200w": 2.0, "200d": 1.5}

# Badge-symbolen (geen externe logo's). Onbekende tickers krijgen een monogram.
BADGES = {"BTC-USD": "₿", "ETH-USD": "Ξ", "SOL-USD": "◎", "ADA-USD": "₳"}

CRYPTO_SUFFIXES = ("-USD", "-EUR", "-USDT")

BAR_CAP_PCT = 50  # balk is "vol" bij ±50% afstand tot de MA

st.set_page_config(page_title="Watchlist Ranker", page_icon="📉", layout="wide")


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

@st.cache_data(ttl=8, show_spinner=False)
def fetch_intraday_batch(tickers: tuple) -> dict:
    """5-minuten-data voor ALLE tickers in één verzoek naar Yahoo.
    Zo blijft auto-verversen om de 10 sec beperkt tot 1 verzoek per keer,
    in plaats van één per ticker (anders loop je snel tegen rate limits aan).
    ttl=8 zodat elke ververs-ronde van 10 sec verse data ophaalt."""
    df = yf.download(
        list(tickers), period="5d", interval="5m", group_by="ticker",
        auto_adjust=True, progress=False, threads=True,
    )
    out = {}
    for t in tickers:
        try:
            sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            s = sub["Close"].dropna()
        except KeyError:
            s = pd.Series(dtype=float)
        if not s.empty and s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        out[t] = s
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_currency(ticker: str) -> str:
    try:
        return yf.Ticker(ticker).fast_info["currency"] or ""
    except Exception:  # noqa: BLE001
        # terugval op basis van de tickernaam
        if ticker.upper().endswith(("-EUR", ".AS", ".PA", ".DE", ".BR")):
            return "EUR"
        return "USD"


def eur_pair(ticker: str) -> str:
    return ticker.rsplit("-", 1)[0] + "-EUR"


@st.cache_data(ttl=900, show_spinner=False)
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


CMC_FNG_URL = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/latest"


def _cmc_api_key():
    """API-sleutel uit st.secrets (Streamlit Cloud: Settings → Secrets)."""
    try:
        return st.secrets.get("CMC_API_KEY")
    except Exception:  # noqa: BLE001  (geen secrets-bestand aanwezig)
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cmc_fear_greed(api_key: str) -> dict:
    resp = requests.get(
        CMC_FNG_URL,
        headers={"X-CMC_PRO_API_KEY": api_key, "Accept": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    return {"waarde": int(round(float(data["value"]))),
            "classificatie": data.get("value_classification", "")}


def fetch_crypto_fear_greed_combined() -> dict:
    """Gemiddelde van alternative.me en CoinMarketCap. Valt één bron weg,
    dan wordt alleen de andere gebruikt; vallen ze allebei weg, dan een fout."""
    waardes, delen, fouten = [], [], []
    try:
        alt = fetch_crypto_fear_greed()
        waardes.append(alt["waarde"])
        delen.append(f"alternative.me {alt['waarde']}")
    except Exception as e:  # noqa: BLE001
        fouten.append(f"alternative.me: {e}")

    key = _cmc_api_key()
    if key:
        try:
            cmc = fetch_cmc_fear_greed(key)
            waardes.append(cmc["waarde"])
            delen.append(f"CoinMarketCap {cmc['waarde']}")
        except Exception as e:  # noqa: BLE001
            fouten.append(f"CoinMarketCap: {e}")
    else:
        fouten.append("CoinMarketCap: geen CMC_API_KEY in secrets")

    if not waardes:
        raise RuntimeError("; ".join(fouten))
    gemiddeld = int(round(sum(waardes) / len(waardes)))
    toelichting = ("Gemiddelde van " if len(waardes) > 1 else "Alleen ") + " en ".join(delen)
    if fouten:
        toelichting += " (" + "; ".join(fouten) + ")"
    return {"waarde": gemiddeld, "toelichting": toelichting}


CNN_FNG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CNN_HEADERS = {
    # CNN weigert verzoeken zonder browser-achtige headers
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://edition.cnn.com/markets/fear-and-greed",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_fear_greed() -> dict:
    """CNN Fear & Greed (aandelen). Geen officiële API: dit is het JSON-endpoint
    dat CNN's eigen pagina gebruikt. Kan zonder aankondiging veranderen."""
    resp = requests.get(CNN_FNG_URL, headers=CNN_HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()["fear_and_greed"]
    return {"waarde": int(round(float(data["score"]))), "classificatie": data["rating"]}


def is_crypto(ticker: str) -> bool:
    return ticker.upper().endswith(CRYPTO_SUFFIXES)


def _pct(now, ref):
    if ref is None or pd.isna(ref) or ref == 0:
        return None
    return round((now / ref - 1) * 100, 2)


def _at_or_before(series: pd.Series, ts):
    s = series[series.index <= ts]
    return float(s.iloc[-1]) if not s.empty else None


def compute_row(ticker: str, weights: dict, intraday_map: dict) -> dict:
    daily = fetch_daily_closes(ticker)
    weekly = daily.resample("W").last().dropna()  # laatste bar = lopende week
    crypto = is_crypto(ticker)

    intraday = intraday_map.get(ticker, pd.Series(dtype=float))
    currency = fetch_currency(ticker)

    if not intraday.empty:
        price = float(intraday.iloc[-1])
        last_ts = intraday.index[-1]
        last_date = last_ts.date()  # datum in tijdzone van de beurs
        updated = last_ts.tz_convert("Europe/Amsterdam").strftime("%d-%m %H:%M")
        # "Live" = laatste koers is recent (Yahoo vertraagt sommige beurzen ~15 min)
        live = pd.Timestamp.now(tz="UTC") - last_ts.tz_convert("UTC") < pd.Timedelta(minutes=45)
    else:
        price = float(daily.iloc[-1])
        last_ts, live = None, False
        last_date = daily.index[-1].date()
        updated = last_date.strftime("%d-%m")

    # 1 uur: alleen zinvol als er nu gehandeld wordt
    ch_1h = _pct(price, _at_or_before(intraday, last_ts - pd.Timedelta(hours=1))) if live else None

    # 24 uur: crypto echt 24u terug; aandelen t.o.v. slotkoers vorige handelsdag
    if crypto and last_ts is not None:
        ch_24h = _pct(price, _at_or_before(intraday, last_ts - pd.Timedelta(hours=24)))
    else:
        prev = daily[daily.index.date < last_date]
        ch_24h = _pct(price, float(prev.iloc[-1])) if not prev.empty else None

    # 7 dagen: slotkoers van (uiterlijk) een week geleden
    week_ago = daily[daily.index.date <= last_date - timedelta(days=7)]
    ch_7d = _pct(price, float(week_ago.iloc[-1])) if not week_ago.empty else None

    # Euro-prijs: echt het -EUR-handelspaar op Yahoo (bv. BTC-EUR), geen omrekening
    price_eur = None
    if crypto and currency != "EUR":
        eur_closes = intraday_map.get(eur_pair(ticker), pd.Series(dtype=float))
        if not eur_closes.empty:  # leeg = geen EUR-paar voor deze coin
            price_eur = float(eur_closes.iloc[-1])

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
        "valuta": currency,
        "bijgewerkt": updated,
        "%_1u": ch_1h,
        "%_24u": ch_24h,
        "%_7d": ch_7d,
        "prijs_eur": price_eur,
        **{f"%_vs_{k}": p for k, p in pcts.items()},
        "score": round(score, 1),
    }


def build_ranking(tickers: list, weights: dict):
    batch = list(tickers) + [
        eur_pair(t) for t in tickers if is_crypto(t) and not t.upper().endswith("-EUR")
    ]
    try:
        intraday_map = fetch_intraday_batch(tuple(batch))
    except Exception:  # noqa: BLE001
        intraday_map = {}  # val terug op dagdata

    rows, errors = [], []
    for t in tickers:
        try:
            rows.append(compute_row(t, weights, intraday_map))
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
    if pct is None or pd.isna(pct):
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


CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "GBp": "p "}


def format_price(price: float, currency: str) -> str:
    decimals = 2 if price >= 1 else 4  # kleine coins (bv. ADA) meer decimalen
    sym = CURRENCY_SYMBOLS.get(currency, f"{currency} " if currency else "")
    # Nederlandse notatie: punt voor duizendtallen, komma voor decimalen
    txt = f"{price:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{sym}{txt}"


def change_chip(label: str, pct) -> str:
    if pct is None or pd.isna(pct):
        val, color = "–", "inherit"
    else:
        val = f"{pct:+.1f}%".replace(".", ",")
        color = "#d64545" if pct < 0 else "#2e9e58"
    return (
        f'<span style="white-space:nowrap"><span style="opacity:0.6">{label}</span> '
        f'<span style="color:{color};font-weight:600;font-variant-numeric:tabular-nums">{val}</span></span>'
    )


def score_html(score: float) -> str:
    """Grote score rechts op de kaart. Positief (onder de MA's) = groen."""
    color = "#2e9e58" if score > 0 else "#d64545" if score < 0 else "inherit"
    txt = f"{score:+.1f}".replace(".", ",")
    return (
        f'<div style="text-align:right;min-width:72px">'
        f'<div style="font-size:1.7rem;font-weight:800;line-height:1.1;color:{color};'
        f'font-variant-numeric:tabular-nums">{txt}</div>'
        f'<div style="font-size:0.68rem;opacity:0.6;text-transform:uppercase;'
        f'letter-spacing:0.05em">score</div></div>'
    )


def card_html(rank: int, row: pd.Series) -> str:
    crypto = is_crypto(row["ticker"])
    bars = "".join(ma_bar_html(k, row[f"%_vs_{k}"]) for k in DEFAULT_WEIGHTS)
    chips = "".join(
        change_chip(lbl, row[col])
        for lbl, col in [("1u", "%_1u"), ("24u" if crypto else "dag", "%_24u"), ("7d", "%_7d")]
    )
    eur = row.get("prijs_eur")
    eur_txt = (
        f'{format_price(eur, "EUR")} · '
        if eur is not None and not pd.isna(eur) else ""
    )
    return (
        f'<div style="border:1px solid rgba(128,128,128,0.25);border-radius:10px;'
        f'padding:10px 12px;margin-bottom:10px">'
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<span style="opacity:0.6;width:18px">{rank}</span>'
        f"{badge_html(row['ticker'], crypto)}"
        # ticker met de prijs ernaast
        f'<div style="flex:1;min-width:0">'
        f'<div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap">'
        f'<span style="font-weight:600">{row["ticker"]}</span>'
        f'<span style="font-weight:700;font-size:1.05rem;font-variant-numeric:tabular-nums">'
        f'{format_price(row["prijs"], row["valuta"])}</span></div>'
        f'<div style="font-size:0.72rem;opacity:0.6;font-variant-numeric:tabular-nums">'
        f'{eur_txt}bijgewerkt {row["bijgewerkt"]}</div></div>'
        # score groot, helemaal rechts
        f'{score_html(row["score"])}'
        f"</div>"
        f'<div style="display:flex;gap:14px;flex-wrap:wrap;font-size:0.78rem;margin:6px 0 4px 28px">'
        f"{chips}</div>"
        f"{bars}</div>"
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
    st.subheader("Verversen")
    auto = st.toggle("Automatisch verversen", value=True)
    interval = st.select_slider(
        "Elke … seconden", options=[10, 30, 60], value=10, disabled=not auto
    )
    if st.button("Nu verversen"):
        st.cache_data.clear()

tickers = [t.strip().upper() for t in tickers_text.splitlines() if t.strip()]

st.title("Watchlist Ranker")

def render_gauge(container, fetch_fn, title: str, fout_prefix: str):
    """Fear & Greed-gauge in een kolom; bij een fout 'n.v.t.' plus melding."""
    try:
        fng = fetch_fn()
        container.markdown(gauge_svg(fng["waarde"], title), unsafe_allow_html=True)
        if fng.get("toelichting"):
            container.caption(fng["toelichting"])
    except Exception as e:  # noqa: BLE001
        container.markdown(gauge_svg(None, title), unsafe_allow_html=True)
        container.caption(f"{fout_prefix}: {e}")


def render_live(tickers: list, weights: dict):
    """Alles wat live ververst: gauges en ranglijst."""
    now = pd.Timestamp.now(tz="Europe/Amsterdam").strftime("%H:%M:%S")
    st.caption(f"Laatst ververst om {now}")

    ranking, errors = build_ranking(tickers, weights)

    if errors:
        st.warning("Niet verwerkt:\n\n" + "\n\n".join(errors))

    if ranking.empty:
        st.error(
            "Geen koersdata opgehaald. Controleer de tickers, of Yahoo Finance "
            "geblokkeerd wordt door je netwerk (bijv. een bedrijfsproxy)."
        )
    else:
        st.caption("Hoogste score = verste onder de MA's. Balk links (rood) = onder de MA. 1u toont – als de beurs dicht is. Bij aandelen is 'dag' de verandering t.o.v. de vorige slotkoers.")

    # Twee kolommen naast elkaar: aandelen links, crypto rechts,
    # elk met de eigen Fear & Greed-gauge bovenaan.
    col_stocks, col_crypto = st.columns(2, gap="large")
    secties = [
        (col_stocks, "Aandelen & ETF's", False,
         fetch_stock_fear_greed, "Aandelen Fear & Greed", "CNN niet bereikbaar"),
        (col_crypto, "Crypto", True,
         fetch_crypto_fear_greed_combined, "Crypto Fear & Greed", "Niet opgehaald"),
    ]
    for col, section, crypto, fetch_fn, gauge_title, fout_prefix in secties:
        with col:
            render_gauge(st, fetch_fn, gauge_title, fout_prefix)
            st.subheader(section)
            if ranking.empty:
                continue
            part = ranking[ranking["ticker"].map(is_crypto) == crypto].reset_index(drop=True)
            if part.empty:
                st.caption("Geen tickers in deze groep.")
                continue
            st.markdown(
                "".join(card_html(i + 1, row) for i, row in part.iterrows()),
                unsafe_allow_html=True,
            )

    if not ranking.empty:
        with st.expander("Ruwe tabel"):
            st.dataframe(ranking, width="stretch", hide_index=True)


# Fragment: alleen dit deel draait opnieuw, om de `interval` seconden
st.fragment(run_every=interval if auto else None)(render_live)(tickers, weights)
