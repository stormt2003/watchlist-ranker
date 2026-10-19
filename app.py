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

import html
import math
import re
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

# Leesbare namen. Tickers die hier niet staan krijgen automatisch de naam
# van Yahoo Finance; zet een ticker hier om die naam te overschrijven.
NAMES = {
    "VWRL.AS": "Vanguard FTSE All-World",
    "AMZN": "Amazon",
    "ACHR": "Archer Aviation",
    "BESI.AS": "BE Semiconductor (Besi)",
    "BSX": "Boston Scientific",
    "MU": "Micron Technology",
    "NFLX": "Netflix",
    "MSTR": "Strategy (MicroStrategy)",
    "TSLA": "Tesla",
    "PATH": "UiPath",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "ADA-USD": "Cardano",
    "HYPE32196-USD": "Hyperliquid",
    "NEAR-USD": "NEAR Protocol",
    "AAVE-USD": "Aave",
    "ONDO-USD": "Ondo",
    "INJ-USD": "Injective",
    "VIRTUAL-USD": "Virtuals Protocol",
}

CRYPTO_SUFFIXES = ("-USD", "-EUR", "-USDT")

# Voor- en nabeurs: alleen VS-beurzen hebben die op Yahoo.
# Handelstijden in Eastern Time, als minuten na middernacht.
US_TZ = "America/New_York"
US_PRE_START = 4 * 60          # 04:00 voorbeurs begint
US_OPEN = 9 * 60 + 30          # 09:30 reguliere handel
US_CLOSE = 16 * 60             # 16:00 nabeurs begint
US_POST_END = 20 * 60          # 20:00 nabeurs eindigt
EXT_MAX_AGE = pd.Timedelta(minutes=60)  # oudere voor-/nabeurskoers = niet meer actief

FX_TICKER = "EURUSD=X"  # wisselkoers voor coins zonder EUR-paar

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
    ttl=8 zodat elke ververs-ronde van 10 sec verse data ophaalt.
    prepost=True: ook voor- en nabeurskoersen (VS); die worden in
    compute_row weer van de reguliere handel gescheiden."""
    df = yf.download(
        list(tickers), period="5d", interval="5m", group_by="ticker",
        auto_adjust=True, prepost=True, progress=False, threads=True,
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


_SUFFIX_RE = re.compile(
    r"[,\s]+(inc\.?|incorporated|corp\.?|corporation|n\.?v\.?|plc|ltd\.?|limited|"
    r"s\.a\.?|se|ag|co\.?|company|holdings?)$",
    re.IGNORECASE,
)


def clean_name(name: str, crypto: bool) -> str:
    """'Amazon.com, Inc.' -> 'Amazon.com', 'Bitcoin USD' -> 'Bitcoin'."""
    name = " ".join((name or "").split())
    if crypto:
        name = re.sub(r"\s+(USD|EUR|USDT)$", "", name)
    prev = None
    while prev != name:  # meerdere achtervoegsels, bv. 'Holdings, Inc.'
        prev = name
        name = _SUFFIX_RE.sub("", name).strip().rstrip(",&").strip()
    return name


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_meta(ticker: str) -> dict:
    """Naam, valuta en tijdzone van de beurs, 6 uur gecachet.
    De naam komt uit Ticker.info (zwaar verzoek), dus alleen voor tickers
    die niet in NAMES staan."""
    t = ticker.upper()
    crypto = is_crypto(t)
    meta = {"naam": NAMES.get(t), "valuta": None, "tz": None}
    try:
        fi = yf.Ticker(ticker).fast_info
        meta["valuta"] = fi["currency"]
        meta["tz"] = fi["timezone"]
    except Exception:  # noqa: BLE001
        pass
    if not meta["naam"]:
        try:
            info = yf.Ticker(ticker).info or {}
            meta["naam"] = clean_name(info.get("longName") or info.get("shortName") or "", crypto)
        except Exception:  # noqa: BLE001
            pass
    # terugval op basis van de tickernaam
    if not meta["valuta"]:
        meta["valuta"] = "EUR" if t.endswith(("-EUR", ".AS", ".PA", ".DE", ".BR")) else "USD"
    if not meta["tz"] and not crypto and "." not in t:
        meta["tz"] = US_TZ  # ticker zonder beurs-achtervoegsel = VS
    meta["naam"] = meta["naam"] or ticker
    return meta


def us_session(ts: pd.Timestamp) -> str:
    """'pre', 'regulier', 'post' of 'dicht' voor een tijdstip (VS-handelstijden)."""
    t = ts.tz_convert(US_TZ)
    if t.weekday() >= 5:
        return "dicht"
    m = t.hour * 60 + t.minute
    if US_PRE_START <= m < US_OPEN:
        return "pre"
    if US_OPEN <= m < US_CLOSE:
        return "regulier"
    if US_CLOSE <= m < US_POST_END:
        return "post"
    return "dicht"


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

    meta = fetch_meta(ticker)
    currency = meta["valuta"]
    now_utc = pd.Timestamp.now(tz="UTC")

    # Bij VS-aandelen zitten voor- en nabeursbars in de intraday-data.
    # Die halen we eruit, zodat prijs, score en 1u/dag op de reguliere handel blijven.
    raw = intraday_map.get(ticker, pd.Series(dtype=float))
    us_stock = not crypto and meta["tz"] == US_TZ
    if us_stock and not raw.empty:
        local = raw.index.tz_convert(US_TZ)
        mins = local.hour * 60 + local.minute
        regulier = (local.weekday < 5) & (mins >= US_OPEN) & (mins < US_CLOSE)
        intraday, ext_bars = raw[regulier], raw[~regulier]
    else:
        intraday, ext_bars = raw, pd.Series(dtype=float)

    if not intraday.empty:
        price = float(intraday.iloc[-1])
        last_ts = intraday.index[-1]
        # datum in tijdzone van de beurs
        last_date = (last_ts.tz_convert(meta["tz"]) if meta["tz"] else last_ts).date()
        updated = last_ts.tz_convert("Europe/Amsterdam").strftime("%d-%m %H:%M")
        # "Live" = laatste koers is recent (Yahoo vertraagt sommige beurzen ~15 min)
        live = now_utc - last_ts.tz_convert("UTC") < pd.Timedelta(minutes=45)
    else:
        price = float(daily.iloc[-1])
        last_ts, live = None, False
        last_date = daily.index[-1].date()
        updated = last_date.strftime("%d-%m")

    # Beurs dicht: gebruik de officiële slotkoers (incl. slotveiling)
    # in plaats van de laatste 5-minutenbar.
    if not crypto and not live and daily.index[-1].date() == last_date:
        price = float(daily.iloc[-1])

    # Voor-/nabeurs: alleen tonen als die sessie nu loopt en de koers vers is.
    ext_sessie = ext_prijs = ext_pct = None
    if us_stock and not ext_bars.empty:
        nu_sessie = us_session(now_utc)
        ext_ts = ext_bars.index[-1]
        if (
            nu_sessie in ("pre", "post")
            and us_session(ext_ts) == nu_sessie
            and now_utc - ext_ts.tz_convert("UTC") < EXT_MAX_AGE
        ):
            ext_sessie = nu_sessie
            ext_prijs = float(ext_bars.iloc[-1])
            ext_pct = _pct(ext_prijs, price)  # t.o.v. de laatste slotkoers
            updated = ext_ts.tz_convert("Europe/Amsterdam").strftime("%d-%m %H:%M")

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

    # 30 dagen: slotkoers van (uiterlijk) 30 dagen geleden
    month_ago = daily[daily.index.date <= last_date - timedelta(days=30)]
    ch_30d = _pct(price, float(month_ago.iloc[-1])) if not month_ago.empty else None

    # Euro-prijs: bij voorkeur het echte -EUR-handelspaar op Yahoo (bv. BTC-EUR).
    # Heeft een coin geen EUR-paar, dan omrekenen met de live EUR/USD-koers.
    price_eur, eur_omgerekend = None, False
    if crypto and currency == "EUR":
        price_eur = price
    elif crypto:
        eur_closes = intraday_map.get(eur_pair(ticker), pd.Series(dtype=float))
        fx = intraday_map.get(FX_TICKER, pd.Series(dtype=float))
        if not eur_closes.empty:
            price_eur = float(eur_closes.iloc[-1])
        elif currency == "USD" and not fx.empty:
            price_eur = price / float(fx.iloc[-1])  # EURUSD=X = dollars per euro
            eur_omgerekend = True

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
        "naam": meta["naam"],
        "prijs": price,
        "ext_sessie": ext_sessie,
        "ext_prijs": ext_prijs,
        "%_ext": ext_pct,
        "valuta": currency,
        "bijgewerkt": updated,
        "%_1u": ch_1h,
        "%_24u": ch_24h,
        "%_7d": ch_7d,
        "%_30d": ch_30d,
        "prijs_eur": price_eur,
        "eur_omgerekend": eur_omgerekend,
        **{f"%_vs_{k}": p for k, p in pcts.items()},
        "score": round(score, 1),
    }


def build_ranking(tickers: list, weights: dict):
    batch = list(tickers) + [
        eur_pair(t) for t in tickers if is_crypto(t) and not t.upper().endswith("-EUR")
    ]
    if any(is_crypto(t) for t in tickers):
        batch.append(FX_TICKER)  # voor coins zonder eigen EUR-paar
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
        f'<svg viewBox="0 0 220 135" style="width:100%;max-width:220px">'
        f'{"".join(parts)}</svg></div>'
    )


def ma_bar_html(label: str, pct) -> str:
    if pct is None or pd.isna(pct):
        fill, txt = "", '<span title="te weinig historie voor dit gemiddelde" style="opacity:0.5">–</span>'
    else:
        width = min(abs(pct), BAR_CAP_PCT) / BAR_CAP_PCT * 50
        side = "right:50%" if pct < 0 else "left:50%"
        color = "#d64545" if pct < 0 else "#2e9e58"
        fill = (
            f'<div style="position:absolute;top:0;bottom:0;{side};width:{width:.1f}%;'
            f'background:{color};border-radius:2px"></div>'
        )
        txt = f"{pct:+.1f}%".replace(".", ",")
    return (
        f'<div style="display:flex;align-items:center;gap:6px;font-size:0.7rem;margin:1px 0">'
        f'<span style="width:30px;opacity:0.6">{label}</span>'
        f'<div style="position:relative;flex:1;height:5px;background:rgba(128,128,128,0.16);border-radius:2px">'
        f'{fill}<div style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;'
        f'background:currentColor;opacity:0.4"></div></div>'
        f'<span style="width:56px;text-align:right;opacity:0.85;font-variant-numeric:tabular-nums">{txt}</span>'
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


EXT_LABELS = {
    "pre": ("Voorbeurs", "Koers in de voorbeurs (04:00–09:30 New Yorkse tijd), t.o.v. de vorige slotkoers"),
    "post": ("Nabeurs", "Koers in de nabeurs (16:00–20:00 New Yorkse tijd), t.o.v. de slotkoers van vandaag"),
}


def ext_chip_html(row: pd.Series) -> str:
    """Klein, gedempt label naast de prijs, alleen als voor-/nabeurs actief is."""
    sessie = row.get("ext_sessie")
    if not isinstance(sessie, str) or sessie not in EXT_LABELS:
        return ""
    label, uitleg = EXT_LABELS[sessie]
    pct = row.get("%_ext")
    if pct is None or pd.isna(pct):
        pct_html = ""
    else:
        color = "#d64545" if pct < 0 else "#2e9e58"
        pct_html = (
            f'<span style="color:{color};font-weight:600">'
            f'{f"{pct:+.2f}%".replace(".", ",")}</span>'
        )
    return (
        f'<span title="{uitleg}" style="display:inline-flex;align-items:baseline;gap:5px;'
        f'padding:1px 8px;border-radius:999px;background:rgba(128,128,128,0.14);'
        f'font-size:0.74rem;font-variant-numeric:tabular-nums;white-space:nowrap">'
        f'<span style="opacity:0.65">{label}</span>'
        f'<span style="opacity:0.85">{format_price(row["ext_prijs"], row["valuta"])}</span>'
        f"{pct_html}</span>"
    )


def title_html(row: pd.Series, size: str = "0.95rem") -> str:
    """Naam als hoofdtekst, de ticker klein en gedempt erachter."""
    naam = html.escape(str(row.get("naam") or row["ticker"]))
    return (
        f'<div style="display:flex;align-items:baseline;gap:6px;min-width:0">'
        f'<span style="font-weight:650;font-size:{size};white-space:nowrap;overflow:hidden;'
        f'text-overflow:ellipsis">{naam}</span>'
        f'<span style="font-size:0.7rem;opacity:0.5;white-space:nowrap">{row["ticker"]}</span></div>'
    )


def score_html(score: float) -> str:
    """Score rechts op de kaart. Positief (onder de MA's) = groen."""
    color = "#2e9e58" if score > 0 else "#d64545" if score < 0 else "inherit"
    txt = f"{score:+.1f}".replace(".", ",")
    return (
        f'<div style="text-align:right;flex-shrink:0;line-height:1">'
        f'<div style="font-size:1.3rem;font-weight:800;color:{color};'
        f'font-variant-numeric:tabular-nums">{txt}</div>'
        f'<div style="font-size:0.62rem;opacity:0.5;margin-top:2px">score</div></div>'
    )


def card_html(rank: int, row: pd.Series) -> str:
    crypto = is_crypto(row["ticker"])
    bars = "".join(ma_bar_html(k, row[f"%_vs_{k}"]) for k in DEFAULT_WEIGHTS)
    chips = "".join(
        change_chip(lbl, row[col])
        for lbl, col in [("1u", "%_1u"), ("24u" if crypto else "dag", "%_24u"), ("7d", "%_7d")]
    )
    eur = row.get("prijs_eur")
    eur_txt = ""
    if crypto and row["valuta"] != "EUR" and eur is not None and not pd.isna(eur):
        approx = "≈ " if row.get("eur_omgerekend") else ""
        eur_txt = (
            f'<span style="font-size:0.8rem;opacity:0.6;font-variant-numeric:tabular-nums">'
            f'{approx}{format_price(eur, "EUR")}</span>'
        )
    return (
        f'<div style="border:1px solid rgba(128,128,128,0.22);border-radius:8px;'
        f'padding:7px 10px 6px;margin-bottom:6px">'
        # kop: rang, naam (ticker klein), score rechts
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<span style="opacity:0.45;font-size:0.75rem;width:14px;flex-shrink:0">{rank}</span>'
        f'<div style="flex:1;min-width:0">{title_html(row)}'
        # prijs met euro-prijs en eventueel voor-/nabeurs ernaast
        f'<div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-top:1px">'
        f'<span style="font-weight:700;font-size:0.92rem;font-variant-numeric:tabular-nums">'
        f'{format_price(row["prijs"], row["valuta"])}</span>{eur_txt}{ext_chip_html(row)}</div></div>'
        f'{score_html(row["score"])}'
        f"</div>"
        # wijzigingen + tijdstip op één regel
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:baseline;font-size:0.72rem;'
        f'margin:4px 0 3px 22px">{chips}'
        f'<span style="margin-left:auto;opacity:0.45;font-variant-numeric:tabular-nums">'
        f'{row["bijgewerkt"]}</span></div>'
        f'<div style="margin-left:22px">{bars}</div></div>'
    )


MOVER_PERIODES = {"Dag": "%_24u", "7d": "%_7d", "30d": "%_30d"}
MOVERS_TOP = 3


def mover_row_html(rank: int, row: pd.Series, pct: float) -> str:
    """Compacte regel: rang, naam (ticker klein), prijs en percentage rechts."""
    color = "#2e9e58" if pct >= 0 else "#d64545"
    pct_txt = f"{pct:+.2f}%".replace(".", ",")
    return (
        f'<div style="display:flex;align-items:center;gap:8px;padding:5px 9px;'
        f'border:1px solid rgba(128,128,128,0.18);border-radius:7px;margin-bottom:4px">'
        f'<span style="opacity:0.45;font-size:0.75rem;width:14px;flex-shrink:0">{rank}</span>'
        f'<div style="flex:1;min-width:0">{title_html(row, "0.88rem")}</div>'
        f'<span style="opacity:0.7;font-size:0.8rem;white-space:nowrap;font-variant-numeric:tabular-nums">'
        f'{format_price(row["prijs"], row["valuta"])}</span>'
        f'<span style="font-weight:800;font-size:0.95rem;min-width:64px;text-align:right;color:{color};'
        f'font-variant-numeric:tabular-nums">{pct_txt}</span></div>'
    )


def render_mover_list(titel: str, part: pd.DataFrame, col: str, leeg: str, alles: bool):
    """Top 3, of met 'alles' de hele lijst als één doorlopende rij kaartjes."""
    st.markdown(f"**{titel}**")
    if part.empty:
        st.caption(leeg)
        return
    rows = [mover_row_html(i + 1, r, r[col]) for i, (_, r) in enumerate(part.iterrows())]
    zichtbaar = rows if alles else rows[:MOVERS_TOP]
    st.markdown("".join(zichtbaar), unsafe_allow_html=True)
    if not alles and len(rows) > MOVERS_TOP:
        st.caption(f"+ {len(rows) - MOVERS_TOP} meer")


def render_movers(ranking: pd.DataFrame):
    st.subheader("Hardste stijgers & dalers")
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        periode = st.radio(
            "Periode", list(MOVER_PERIODES), horizontal=True,
            key="mover_periode", label_visibility="collapsed",
        )
    with c2:
        # Een schakelaar i.p.v. een uitklapper: de lijst loopt zo gewoon door,
        # en de stand blijft bewaard bij het automatisch verversen.
        alles = st.toggle("Toon alle", key="mover_alles")
    col = MOVER_PERIODES[periode]
    if periode == "Dag":
        st.caption("Crypto: laatste 24 uur. Aandelen: t.o.v. de slotkoers van de vorige handelsdag.")
    data = ranking[["ticker", "naam", "prijs", "valuta", col]].dropna(subset=[col])
    stijgers = data[data[col] > 0].sort_values(col, ascending=False)
    dalers = data[data[col] < 0].sort_values(col, ascending=True)

    left, right = st.columns(2, gap="large")
    with left:
        render_mover_list(f"Stijgers ({periode})", stijgers, col,
                          "Geen stijgers in deze periode.", alles)
    with right:
        render_mover_list(f"Dalers ({periode})", dalers, col,
                          "Geen dalers in deze periode.", alles)


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
        st.caption("Hoogste score = verste onder de MA's. Balk links (rood) = onder de MA. 1u toont – als de beurs dicht is. Bij aandelen is 'dag' de verandering t.o.v. de vorige slotkoers. Tijdens voor- of nabeurs (alleen VS) staat die koers klein naast de prijs; de score blijft op de reguliere koers.")

    if not ranking.empty:
        render_movers(ranking)
        st.divider()

    # Twee kolommen naast elkaar: aandelen links, crypto rechts,
    # elk met de eigen Fear & Greed-gauge bovenaan.
    col_stocks, col_crypto = st.columns(2, gap="large")
    secties = [
        (col_stocks, "Aandelen & ETF's", False,
         fetch_stock_fear_greed, "Aandelen Fear & Greed", "CNN niet bereikbaar"),
        (col_crypto, "Crypto", True,
         fetch_crypto_fear_greed, "Crypto Fear & Greed", "Niet opgehaald"),
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
