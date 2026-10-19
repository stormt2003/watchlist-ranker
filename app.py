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
import time
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

FX_TICKER = "EURUSD=X"  # euro/dollar-wisselkoers (dollars per euro) voor het omrekenen

BAR_CAP_PCT = 50  # balk is "vol" bij ±50% afstand tot de MA

# ---------------------------------------------------------------------------
# MARKTTHERMOMETER: waardering op basis van de afstand tot de 200-weeks MA
# ---------------------------------------------------------------------------
# Per zone: (tot en met dit % boven de 200w-MA, naam). Alles boven de laatste
# grens valt in de laatste zone. "min"/"max" = bereik van de meter.
# Pas de grenzen hier aan als de markt verandert (bv. minder BTC-uitschieters).
THERMO_MA_WEKEN = 200
THERMO = {
    "aandelen": dict(
        ticker="VWRL.AS", min=-10, max=50,
        valuta={"EUR": "VWRL.AS"},  # koers en MA tonen in deze valuta('s), via deze ticker
        zones=[
            (0, "Spotgoedkoop"),     # op of onder de 200w-MA
            (10, "Zeer goedkoop"),
            (20, "Goedkoop"),
            (30, "Redelijk"),
            (40, "Duur"),
            (50, "Zeer duur"),       # en alles daarboven
        ],
    ),
    "crypto": dict(
        ticker="BTC-USD", min=-50, max=200,
        valuta={"EUR": "BTC-EUR", "USD": "BTC-USD"},
        zones=[
            (0, "Zeer goedkoop"),    # op of onder de 200w-MA
            (50, "Goedkoop"),
            (100, "Redelijk"),
            (150, "Duur"),
            (200, "Zeer duur"),      # en alles daarboven
        ],
    ),
}
ZONE_KLEUREN = {
    "Spotgoedkoop": "#1b7a43",
    "Zeer goedkoop": "#2e9e58",
    "Goedkoop": "#9acd5c",
    "Redelijk": "#e9c440",
    "Duur": "#ee8a35",
    "Zeer duur": "#d64545",
}
# Donkerdere tint voor de zonenaam onder het getal (geel/lichtgroen is slecht leesbaar)
ZONE_TEKSTKLEUREN = {"Goedkoop": "#5f9a2c", "Redelijk": "#b08a0c"}
# Afbreken van lange namen in de meter
ZONE_AFBREKEN = {"Spotgoedkoop": "Spot- goedkoop"}

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
    """Soort (crypto of aandeel/ETF), naam, valuta en beurs-tijdzone, 6 uur gecachet.
    De soort komt van Yahoo zelf (quoteType), zodat ook nieuwe tickers goed
    worden ingedeeld; alleen als Yahoo niet antwoordt kijken we naar de tickernaam.
    De naam komt uit Ticker.info (zwaar verzoek), dus alleen voor tickers
    die niet in NAMES staan."""
    t = ticker.upper()
    meta = {"naam": NAMES.get(t), "valuta": None, "tz": None, "soort": None}
    try:
        fi = yf.Ticker(ticker).fast_info
        meta["valuta"] = fi["currency"]
        meta["tz"] = fi["timezone"]
        meta["soort"] = fi["quoteType"]
    except Exception:  # noqa: BLE001
        pass
    if not meta["naam"]:
        try:
            info = yf.Ticker(ticker).info or {}
            meta["soort"] = meta["soort"] or info.get("quoteType")
            meta["naam"] = info.get("longName") or info.get("shortName")
        except Exception:  # noqa: BLE001
            pass
    soort = str(meta["soort"] or "").upper()
    meta["crypto"] = soort == "CRYPTOCURRENCY" if soort else looks_like_crypto(t)
    meta["naam"] = clean_name(meta["naam"] or "", meta["crypto"]) or ticker
    # terugval op basis van de tickernaam
    if not meta["valuta"]:
        meta["valuta"] = "EUR" if t.endswith(("-EUR", ".AS", ".PA", ".DE", ".BR")) else "USD"
    if not meta["tz"] and not meta["crypto"] and "." not in t:
        meta["tz"] = US_TZ  # ticker zonder beurs-achtervoegsel = VS
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


def crypto_pair(ticker: str, valuta: str) -> str:
    """BTC-USD -> BTC-EUR (of andersom): hetzelfde coin in een andere valuta."""
    return ticker.upper().rsplit("-", 1)[0] + f"-{valuta}"


def fx_rate(van: str, naar: str, intraday_map: dict):
    """Wisselkoers: 1 `van` = x `naar`. Eerst live (intraday), dan de laatste
    dagkoers; Yahoo heeft niet elk paar in beide richtingen, dus ook omgekeerd."""
    if van == naar:
        return 1.0
    for paar, omkeren in ((f"{van}{naar}=X", False), (f"{naar}{van}=X", True)):
        s = intraday_map.get(paar)
        if s is not None and not s.empty:
            r = float(s.iloc[-1])
            return 1 / r if omkeren else r
    for paar, omkeren in ((f"{van}{naar}=X", False), (f"{naar}{van}=X", True)):
        try:
            r = float(fetch_daily_closes(paar).iloc[-1])
            return 1 / r if omkeren else r
        except Exception:  # noqa: BLE001
            continue
    return None


def convert(prijs: float, valuta: str, naar: str, intraday_map: dict):
    """Prijs omrekenen naar `naar`; None als er geen wisselkoers is."""
    if valuta == "GBp":  # Londen noteert in pence
        prijs, valuta = prijs / 100, "GBP"
    r = fx_rate(valuta, naar, intraday_map)
    return prijs * r if r else None


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


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_weekly_max(ticker: str) -> pd.Series:
    """Weekslotkoersen over de hele beschikbare historie (voor de thermometer)."""
    hist = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
    if hist.empty:
        raise ValueError("geen historie")
    closes = hist["Close"].dropna()
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    return closes.resample("W").last().dropna()


def thermometer(ticker: str, live_price=None) -> dict:
    """Afstand van de koers tot de 200w-MA, plus de historische uitersten als context."""
    weekly = fetch_weekly_max(ticker)
    ma = weekly.rolling(THERMO_MA_WEKEN).mean()
    afstand = (weekly / ma - 1).dropna() * 100
    if afstand.empty:
        raise ValueError("te weinig historie voor een 200w-MA")
    verleden = afstand.iloc[:-1] if len(afstand) > 1 else afstand  # lopende week niet
    prijs = live_price if live_price is not None else float(weekly.iloc[-1])
    ma_nu = float(ma.iloc[-1])
    return {
        "afstand": (prijs / ma_nu - 1) * 100,
        "prijs": prijs,
        "ma": ma_nu,
        "laagste": (float(verleden.min()), verleden.idxmin()),
        "hoogste": (float(verleden.max()), verleden.idxmax()),
        "sinds": verleden.index[0],
    }


def thermo_zone(afstand: float, cfg: dict) -> tuple:
    """(naam, ondergrens, bovengrens) van de zone waar de afstand in valt."""
    onder = None
    for tot, naam in cfg["zones"]:
        if afstand <= tot:
            return naam, onder, tot
        onder = tot
    return cfg["zones"][-1][1], cfg["zones"][-2][0], None


# Crypto Fear & Greed: CoinMarketCap, met alternative.me als terugval.
# - Met een (gratis) CMC API-sleutel in .streamlit/secrets.toml (CMC_API_KEY = "...")
#   gebruikt de app de officiële API.
# - Zonder sleutel: het JSON-endpoint dat de CMC-website zelf gebruikt. Geen
#   officiële API, dus kan zonder aankondiging veranderen.
# - Lukt beide niet, dan alternative.me (die werkt maar één keer per dag bij).
CMC_FNG_LATEST = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/latest"
CMC_FNG_CHART = "https://api.coinmarketcap.com/data-api/v3/fear-greed/chart"
ALT_FNG_URL = "https://api.alternative.me/fng/?limit=1"


def _cmc_api_key():
    try:
        return st.secrets.get("CMC_API_KEY")
    except Exception:  # noqa: BLE001  (geen secrets.toml)
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_crypto_fear_greed() -> dict:
    fouten = []
    sleutel = _cmc_api_key()
    if sleutel:
        try:
            resp = requests.get(
                CMC_FNG_LATEST, timeout=10,
                headers={"X-CMC_PRO_API_KEY": sleutel, "Accept": "application/json"},
            )
            resp.raise_for_status()
            d = resp.json()["data"]
            return {"waarde": int(round(float(d["value"]))),
                    "classificatie": d.get("value_classification", ""),
                    "bron": "CoinMarketCap", "zones": "cmc"}
        except Exception as e:  # noqa: BLE001
            fouten.append(f"CMC-API: {e}")
    try:
        nu = int(time.time())
        resp = requests.get(
            CMC_FNG_CHART, params={"start": nu - 7 * 86400, "end": nu},
            headers={"User-Agent": CNN_HEADERS["User-Agent"], "Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        d = resp.json()["data"]
        laatste = (d.get("historicalValues") or {}).get("now") or d["dataList"][-1]
        return {"waarde": int(round(float(laatste["score"]))),
                "classificatie": laatste.get("name", ""),
                "bron": "CoinMarketCap", "zones": "cmc"}
    except Exception as e:  # noqa: BLE001
        fouten.append(f"CMC-website: {e}")
    resp = requests.get(ALT_FNG_URL, timeout=10)
    resp.raise_for_status()
    d = resp.json()["data"][0]
    return {"waarde": int(d["value"]), "classificatie": d["value_classification"],
            "bron": "alternative.me", "zones": "standaard", "fouten": fouten}


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


def looks_like_crypto(ticker: str) -> bool:
    """Gok op basis van de tickernaam (BTC-USD, ETH-EUR, ...)."""
    return ticker.upper().endswith(CRYPTO_SUFFIXES)


def is_crypto(ticker: str) -> bool:
    """Crypto volgens Yahoo (zie fetch_meta), met de tickernaam als terugval."""
    return fetch_meta(ticker)["crypto"]


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

    # Prijs in euro én dollar. Crypto: bij voorkeur het echte handelspaar op Yahoo
    # (bv. BTC-EUR naast BTC-USD). Verder omrekenen met de live wisselkoers.
    omgerekend_prijzen = {}
    for doel in ("EUR", "USD"):
        if currency == doel:
            omgerekend_prijzen[doel] = (price, False)
            continue
        paar = intraday_map.get(crypto_pair(ticker, doel)) if crypto else None
        if paar is not None and not paar.empty:
            omgerekend_prijzen[doel] = (float(paar.iloc[-1]), False)
        else:
            v = convert(price, currency, doel, intraday_map)
            omgerekend_prijzen[doel] = (v, v is not None)

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
        "prijs_eur": omgerekend_prijzen["EUR"][0],
        "eur_omgerekend": omgerekend_prijzen["EUR"][1],
        "prijs_usd": omgerekend_prijzen["USD"][0],
        "usd_omgerekend": omgerekend_prijzen["USD"][1],
        **{f"%_vs_{k}": p for k, p in pcts.items()},
        "score": round(score, 1),
    }


def build_ranking(tickers: list, weights: dict):
    # Alles in één verzoek: de tickers zelf, voor crypto het paar in de andere
    # valuta (BTC-EUR bij BTC-USD), en de wisselkoersen om de rest om te rekenen.
    batch = list(tickers)
    for t in tickers:
        if is_crypto(t):
            batch += [p for p in (crypto_pair(t, "EUR"), crypto_pair(t, "USD")) if p != t.upper()]
    batch.append(FX_TICKER)
    for cur in {fetch_meta(t)["valuta"] for t in tickers} - {"EUR", "USD"}:
        basis = "GBP" if cur == "GBp" else cur
        batch += [f"{basis}USD=X", f"{basis}EUR=X"]
    batch = list(dict.fromkeys(batch))  # dubbele eruit, volgorde behouden
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


# Donkerdere tint voor de zonenaam onder het getal (geel/lichtgroen is slecht leesbaar)
# CoinMarketCap deelt de index in vijf gelijke stukken van 20 in
CMC_ZONES = [
    (0, 20, "Extreme Fear", "#d64545"),
    (20, 40, "Fear", "#ee8a35"),
    (40, 60, "Neutral", "#e9c440"),
    (60, 80, "Greed", "#9acd5c"),
    (80, 100, "Extreme Greed", "#2e9e58"),
]
FNG_ZONES = {"standaard": ZONES, "cmc": CMC_ZONES}


def _ticks(zones) -> list:
    return [(g, str(g)) for g in sorted({lo for lo, *_ in zones} | {hi for _, hi, *_ in zones})]


FNG_TEKSTKLEUREN = {"Neutral": "#b08a0c", "Greed": "#5f9a2c"}


def gauge_svg(value, title: str, zones=ZONES, label=None, ticks=None,
              sublabel=None, sublabel_color="currentColor") -> str:
    """Halve-cirkelmeter. `value` (0-100) = stand van de naald, `zones` in
    dezelfde 0-100-schaal, `ticks` = [(positie, tekst)] langs de buitenkant."""
    cx, cy, r, w = 110, 105, 85, 20
    gap = 0.8
    parts = []
    for lo, hi, name, color in zones:
        a0, a1 = lo + (gap if lo > 0 else 0), hi - (gap if hi < 100 else 0)
        x0, y0 = _polar(cx, cy, r, a0)
        x1, y1 = _polar(cx, cy, r, a1)
        parts.append(
            f'<path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 0 1 {x1:.1f},{y1:.1f}" '
            f'stroke="{color}" stroke-width="{w}" fill="none"/>'
        )
        # naam binnen de boog, bij twee woorden op twee regels
        tx, ty = _polar(cx, cy, r - w - 8, (lo + hi) / 2)
        woorden = ZONE_AFBREKEN.get(name, name).split(" ", 1)
        regels = "".join(
            f'<tspan x="{tx:.1f}" dy="{0 if i == 0 else 6.5}">{wd}</tspan>'
            for i, wd in enumerate(woorden)
        )
        parts.append(
            f'<text x="{tx:.1f}" y="{ty - (3 if len(woorden) > 1 else 0):.1f}" font-size="6" '
            f'text-anchor="middle" fill="currentColor" opacity="0.7">{regels}</text>'
        )
    for p, txt in ticks or []:
        tx, ty = _polar(cx, cy, r + w / 2 + 7, p)
        parts.append(
            f'<text x="{tx:.1f}" y="{ty + 2:.1f}" font-size="6" text-anchor="middle" '
            f'fill="currentColor" opacity="0.55">{txt}</text>'
        )
    if value is not None:
        nx, ny = _polar(cx, cy, r - 4, max(0, min(100, value)))
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" '
            f'stroke="currentColor" stroke-width="3" stroke-linecap="round"/>'
        )
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="currentColor"/>')
        label = label if label is not None else f"{value}"
    else:
        label = "n.v.t."
    parts.append(
        f'<text x="{cx}" y="{cy + 22}" font-size="18" font-weight="600" '
        f'text-anchor="middle" fill="currentColor">{label}</text>'
    )
    if sublabel:
        parts.append(
            f'<text x="{cx}" y="{cy + 36}" font-size="10" font-weight="700" '
            f'text-anchor="middle" fill="{sublabel_color}">{sublabel}</text>'
        )
    return (
        f'<div style="text-align:center">'
        f'<div style="font-size:0.9rem;opacity:0.8">{title}</div>'
        f'<svg viewBox="0 -6 220 150" style="width:100%;max-width:240px">'
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


def title_html(row: pd.Series, size: str = "0.95rem", stacked: bool = False) -> str:
    """Naam als hoofdtekst, de ticker klein en gedempt erachter (of eronder)."""
    naam = html.escape(str(row.get("naam") or row["ticker"]))
    richting = "flex-direction:column;gap:0;line-height:1.2" if stacked else "align-items:baseline;gap:6px"
    return (
        f'<div style="display:flex;{richting};min-width:0">'
        f'<span style="font-weight:650;font-size:{size};white-space:nowrap;overflow:hidden;'
        f'text-overflow:ellipsis">{naam}</span>'
        f'<span style="font-size:0.7rem;opacity:0.5;white-space:nowrap">{row["ticker"]}</span></div>'
    )


def other_prices(row: pd.Series) -> list:
    """De prijs in euro en/of dollar naast de eigen valuta.
    '≈' = omgerekend met de wisselkoers, zonder = echt handelspaar."""
    out = []
    for cur in ("EUR", "USD"):
        if row["valuta"] == cur:
            continue
        v = row.get(f"prijs_{cur.lower()}")
        if v is None or pd.isna(v):
            continue
        approx = "≈ " if row.get(f"{cur.lower()}_omgerekend") else ""
        out.append(f"{approx}{format_price(float(v), cur)}")
    return out


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
    andere = "".join(
        f'<span style="font-size:0.8rem;opacity:0.6;font-variant-numeric:tabular-nums">{p}</span>'
        for p in other_prices(row)
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
        f'{format_price(row["prijs"], row["valuta"])}</span>{andere}{ext_chip_html(row)}</div></div>'
        f'{score_html(row["score"])}'
        f"</div>"
        # wijzigingen + tijdstip op één regel
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:baseline;font-size:0.72rem;'
        f'margin:4px 0 3px 22px">{chips}'
        f'<span style="margin-left:auto;opacity:0.45;font-variant-numeric:tabular-nums">'
        f'{row["bijgewerkt"]}</span></div>'
        f'<div style="margin-left:22px">{bars}</div></div>'
    )


# Per soort een eigen label voor de korte periode: crypto handelt 24/7,
# aandelen vergelijken we met de vorige slotkoers.
MOVER_PERIODES = {
    False: {"Dag": "%_24u", "7d": "%_7d", "30d": "%_30d"},
    True: {"24u": "%_24u", "7d": "%_7d", "30d": "%_30d"},
}
MOVERS_TOP = 3


def mover_row_html(rank: int, row: pd.Series, pct: float) -> str:
    """Compacte regel: rang, naam (ticker klein), prijs en percentage rechts."""
    color = "#2e9e58" if pct >= 0 else "#d64545"
    pct_txt = f"{pct:+.2f}%".replace(".", ",")
    return (
        f'<div style="display:flex;align-items:center;gap:8px;padding:5px 9px;'
        f'border:1px solid rgba(128,128,128,0.18);border-radius:7px;margin-bottom:4px">'
        f'<span style="opacity:0.45;font-size:0.75rem;width:14px;flex-shrink:0">{rank}</span>'
        f'<div style="flex:1;min-width:0">{title_html(row, "0.88rem", stacked=True)}</div>'
        # eigen valuta met de euro/dollar-prijs eronder
        f'<div style="text-align:right;line-height:1.2;white-space:nowrap;font-variant-numeric:tabular-nums">'
        f'<div style="opacity:0.8;font-size:0.8rem">{format_price(row["prijs"], row["valuta"])}</div>'
        + "".join(f'<div style="opacity:0.5;font-size:0.7rem">{p}</div>' for p in other_prices(row))
        + '</div>'
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
        st.caption(f"Nog {len(rows) - MOVERS_TOP} meer, zet 'Toon alle' aan om ze te zien.")


def render_movers(ranking: pd.DataFrame, crypto: bool):
    perioden = MOVER_PERIODES[crypto]
    st.subheader("Hardste stijgers & dalers")
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        periode = st.radio(
            "Periode", list(perioden), horizontal=True,
            key=f"mover_periode_{'crypto' if crypto else 'aandelen'}",
            label_visibility="collapsed",
        )
    with c2:
        # Een schakelaar i.p.v. een uitklapper: de lijst loopt zo gewoon door,
        # en de stand blijft bewaard bij het automatisch verversen.
        alles = st.toggle("Toon alle", key="mover_alles")
    col = perioden[periode]
    if col == "%_24u":
        st.caption("Laatste 24 uur." if crypto else "T.o.v. de slotkoers van de vorige handelsdag.")
    data = ranking.dropna(subset=[col])
    stijgers = data[data[col] > 0].sort_values(col, ascending=False)
    dalers = data[data[col] < 0].sort_values(col, ascending=True)

    left, right = st.columns(2, gap="medium")
    with left:
        render_mover_list(f"Stijgers ({periode})", stijgers, col,
                          "Geen stijgers in deze periode.", alles)
    with right:
        render_mover_list(f"Dalers ({periode})", dalers, col,
                          "Geen dalers in deze periode.", alles)


# ---------------------------------------------------------------------------
# PAGINA
# ---------------------------------------------------------------------------

SOORTEN = {
    "aandelen": dict(crypto=False, titel="Aandelen & ETF's", fetch=lambda: fetch_stock_fear_greed(),
                     gauge="Aandelen Fear & Greed", fout="CNN niet bereikbaar"),
    "crypto": dict(crypto=True, titel="Crypto", fetch=lambda: fetch_crypto_fear_greed(),
                   gauge="Crypto Fear & Greed", fout="Niet opgehaald"),
}

with st.sidebar:
    st.header("Instellingen")
    tickers_text = st.text_area(
        "Tickers (één per regel)", "\n".join(DEFAULT_TICKERS), height=200
    )
    st.caption(
        "Tickers worden automatisch ingedeeld als aandeel/ETF of crypto, "
        "op basis van wat Yahoo Finance over de ticker zegt. Crypto in "
        "yfinance-formaat, bv. BTC-USD."
    )
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

tickers = list(dict.fromkeys(t.strip().upper() for t in tickers_text.splitlines() if t.strip()))

with st.spinner("Tickers indelen…"):
    indeling = {t: ("crypto" if is_crypto(t) else "aandelen") for t in tickers}
per_soort = {k: [t for t in tickers if indeling[t] == k] for k in SOORTEN}

st.title("Watchlist Ranker")

# De grote schakelaar: alles hieronder (gauge, stijgers/dalers, scores,
# tabel) gaat alleen over de gekozen soort.
st.markdown(
    """<style>
    .st-key-soort button { min-height: 3rem; font-size: 1.05rem; font-weight: 600; }
    .st-key-soort button p { font-size: 1.05rem; }
    </style>""",
    unsafe_allow_html=True,
)
soort_kwargs = dict(
    label="Soort", options=list(SOORTEN), default="aandelen", required=True,
    key="soort", label_visibility="collapsed",
    format_func=lambda k: f"{SOORTEN[k]['titel']}  ({len(per_soort[k])})",
)
try:
    soort = st.segmented_control(**soort_kwargs, width="stretch")
except TypeError:  # oudere Streamlit zonder width-parameter
    soort = st.segmented_control(**soort_kwargs)
soort = soort or "aandelen"


def render_gauge(fetch_fn, title: str, fout_prefix: str):
    """Fear & Greed-gauge, met de zone-indeling van de bron; bij een fout 'n.v.t.'."""
    try:
        fng = fetch_fn()
        waarde = fng["waarde"]
        zones = FNG_ZONES.get(fng.get("zones", "standaard"), ZONES)
        if fng.get("bron") and fng["bron"] != "CNN":
            title = f"{title} ({fng['bron']})"
        # zone volgens de bron zelf ("Extreme fear", "greed", ...), anders op basis van de waarde
        kleuren = {n.lower(): (n, c) for _, _, n, c in zones}
        naam, kleur = kleuren.get(
            str(fng.get("classificatie", "")).strip().lower(),
            next(((n, c) for lo, hi, n, c in zones if waarde < hi), zones[-1][2:]),
        )
        st.markdown(
            gauge_svg(waarde, title, zones=zones, ticks=_ticks(zones), sublabel=naam,
                      sublabel_color=FNG_TEKSTKLEUREN.get(naam, kleur)),
            unsafe_allow_html=True,
        )
    except Exception as e:  # noqa: BLE001
        st.markdown(gauge_svg(None, title, ticks=_ticks(ZONES)), unsafe_allow_html=True)
        st.caption(f"{fout_prefix}: {e}")


MAANDEN = ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]


def _maand(ts) -> str:
    return f"{MAANDEN[ts.month - 1]} {ts.year}"


def _pct_nl(v: float, dec: int = 0) -> str:
    return f"{v:+.{dec}f}%".replace(".", ",")


def render_thermometer(soort: str, ranking: pd.DataFrame):
    """Waarderingsmeter: afstand tot de 200w-MA, ingedeeld in vaste zones (zie THERMO)."""
    cfg = THERMO[soort]
    ticker = cfg["ticker"]
    titel = f"{NAMES.get(ticker, ticker)} t.o.v. 200w-MA"
    lo_m, hi_m = cfg["min"], cfg["max"]

    def pos(pct):  # % boven MA -> positie 0-100 op de meter
        return (max(lo_m, min(hi_m, pct)) - lo_m) / (hi_m - lo_m) * 100

    grenzen = [lo_m] + [tot for tot, _ in cfg["zones"]]
    grenzen[-1] = hi_m
    zones = [
        (pos(grenzen[i]), pos(grenzen[i + 1]), naam, ZONE_KLEUREN.get(naam, "#999"))
        for i, (_, naam) in enumerate(cfg["zones"])
    ]
    ticks = [(pos(g), f"{g:+d}%".replace("+0%", "0%")) for g in grenzen]

    live = ranking.loc[ranking["ticker"] == ticker, "prijs"]
    try:
        t = thermometer(ticker, float(live.iloc[0]) if not live.empty else None)
    except Exception as e:  # noqa: BLE001
        st.markdown(gauge_svg(None, titel, zones, ticks=ticks), unsafe_allow_html=True)
        st.caption(f"Niet berekend: {e}")
        return

    naam, onder, boven = thermo_zone(t["afstand"], cfg)
    st.markdown(
        gauge_svg(pos(t["afstand"]), titel, zones, label=_pct_nl(t["afstand"]), ticks=ticks,
                  sublabel=naam,
                  sublabel_color=ZONE_TEKSTKLEUREN.get(naam, ZONE_KLEUREN.get(naam, "currentColor"))),
        unsafe_allow_html=True,
    )
    # Koers en 200w-MA per valuta. Elke valuta heeft een eigen handelspaar
    # (BTC-EUR / BTC-USD), zodat de MA echt in die valuta berekend is.
    koersen, mas = [], []
    for cur, tk in cfg["valuta"].items():
        if tk == ticker:
            tc = t
        else:
            live_c = ranking.loc[ranking["ticker"] == ticker, f"prijs_{cur.lower()}"]
            live_c = float(live_c.iloc[0]) if not live_c.empty and pd.notna(live_c.iloc[0]) else None
            try:
                tc = thermometer(tk, live_c)
            except Exception:  # noqa: BLE001
                continue
        koersen.append(format_price(tc["prijs"], cur))
        mas.append(format_price(tc["ma"], cur))
    if koersen:
        # Als HTML i.p.v. st.caption: Streamlit leest twee $-tekens in een
        # bijschrift als een formule (ander lettertype/kleur). &#36; = $.
        regels = [f"Koers {' / '.join(koersen)}", f"200w-MA {' / '.join(mas)}"]
        st.markdown(
            '<div style="text-align:center;font-size:0.85rem;opacity:0.6;line-height:1.5;'
            'font-variant-numeric:tabular-nums">'
            + "<br>".join(html.escape(r).replace("$", "&#36;") for r in regels)
            + "</div>",
            unsafe_allow_html=True,
        )


def render_live(tickers: list, weights: dict, soort: str):
    """Alles wat live ververst, voor één soort: gauge, stijgers/dalers en ranglijst."""
    cfg = SOORTEN[soort]
    crypto = cfg["crypto"]
    now = pd.Timestamp.now(tz="Europe/Amsterdam").strftime("%H:%M:%S")
    st.caption(f"Laatst ververst om {now}")

    if not tickers:
        st.info(f"Er staan geen {cfg['titel'].lower()} in je lijst. Voeg tickers toe in de zijbalk.")
        return

    ranking, errors = build_ranking(tickers, weights)

    if errors:
        st.warning("Niet verwerkt:\n\n" + "\n\n".join(errors))

    if ranking.empty:
        st.error(
            "Geen koersdata opgehaald. Controleer de tickers, of Yahoo Finance "
            "geblokkeerd wordt door je netwerk (bijv. een bedrijfsproxy)."
        )
        render_gauge(cfg["fetch"], cfg["gauge"], cfg["fout"])
        return

    # Bovenaan: de twee graadmeters naast elkaar
    g1, g2 = st.columns(2, gap="large")
    with g1:
        render_gauge(cfg["fetch"], cfg["gauge"], cfg["fout"])
    with g2:
        render_thermometer(soort, ranking)

    st.divider()
    render_movers(ranking, crypto)

    st.divider()
    st.subheader("Ranglijst")
    uitleg = ("Hoogste score = verste onder de MA's. Balk links (rood) = onder de MA. 1u toont – als er niet gehandeld wordt. "
              "≈ voor een prijs = omgerekend met de actuele euro/dollar-koers.")
    if not crypto:
        uitleg += (" 'Dag' is de verandering t.o.v. de vorige slotkoers. Tijdens voor- of nabeurs (alleen VS) "
                   "staat die koers klein naast de prijs; de score blijft op de reguliere koers.")
    st.caption(uitleg)
    # Kaartjes in een raster dat zich aanpast aan de schermbreedte
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));'
        'column-gap:10px">'
        + "".join(card_html(i + 1, row) for i, row in ranking.iterrows())
        + "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Ruwe tabel"):
        st.dataframe(ranking, width="stretch", hide_index=True)


# Fragment: alleen dit deel draait opnieuw, om de `interval` seconden.
# Alleen de tickers van de gekozen soort worden opgehaald.
st.fragment(run_every=interval if auto else None)(render_live)(per_soort[soort], weights, soort)
