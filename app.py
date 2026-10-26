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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:  # oudere yfinance
    class YFRateLimitError(Exception):
        pass

try:  # om threads aan de Streamlit-sessie te koppelen (voorkomt waarschuwingen)
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:  # pragma: no cover
    add_script_run_ctx = get_script_run_ctx = None

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = [
    # Aandelen & ETF's
    "VWRL.AS", "AMZN", "ACHR", "BESI.AS", "BSX", "MU", "NFLX", "MSTR", "TSLA", "PATH",
    "NBIS",
    # Crypto (Yahoo gebruikt bij naamconflicten een nummer, bv. HYPE32196-USD)
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "HYPE32196-USD",
    "NEAR-USD", "AAVE-USD", "ONDO-USD", "INJ-USD", "VIRTUAL-USD",
]

# Vaste watchlist bij het openen. Tickers die al in je portefeuille staan
# (bv. AMZN, TSLA, BESI) hoeven hier niet: die komen er automatisch bij.
DEFAULT_WATCHLIST = [
    # Gevraagd
    "ASML.AS", "NVDA", "BABA", "META",
    # Rest van de 'Magnificent 7' (AMZN en TSLA zitten al in je portefeuille)
    "AAPL", "MSFT", "GOOGL",
    # Grootste bedrijven ter wereld naar beurswaarde (top 20)
    "AVGO", "TSM", "BRK-B", "LLY", "WMT", "JPM", "2222.SR", "V", "ORCL",
    "0700.HK", "XOM", "MA", "JNJ",
    # Uit je TradingView-lijst
    "ASM.AS", "HIMS", "LMND", "CRM", "AMD", "NOW", "PLTR", "LTC-USD",
    # AI, chips & datacenters
    "000660.KS", "005930.KS", "8035.T", "6857.T", "AMAT", "LRCX", "ANET", "VRT", "SU.PA", "MOD",
    # Kernenergie & stroom voor AI
    "CEG", "VST", "GEV", "CCJ", "ENR.DE", "RR.L", "OKLO", "SMR",
    # Quantum, ruimtevaart, drones & robotica
    "IONQ", "RGTI", "QBTS", "RKLB", "ASTS", "RHM.DE", "AVAV", "SYM", "ISRG", "JOBY",
]

# Sector- en thema-ETF's: altijd bovenaan de stijgers & dalers in de watchlist
# (aandelen). Amerikaanse fondsen: ideaal om te volgen, voor NL-beleggers vaak
# niet te koop (daar bestaan Europese varianten van).
SECTOR_ETFS = [
    # (ticker, naam, uitleg)
    ("SPY", "S&P 500", "De 500 grootste Amerikaanse bedrijven. Dé graadmeter voor de Amerikaanse beurs; vergelijk de sectoren hiermee."),
    ("QQQ", "Nasdaq 100", "De 100 grootste niet-financiële bedrijven op de Nasdaq, vooral tech. Laat zien hoe het met de groeiaandelen gaat."),
    ("XLK", "Technologie", "Software, hardware en chips, zoals Apple, Microsoft en NVIDIA."),
    ("XLC", "Communicatie", "Internet, media en telecom, zoals Meta, Alphabet en Netflix."),
    ("XLY", "Luxe consumptie", "Wat mensen kopen als het goed gaat: Amazon, Tesla, McDonald's, Nike. Stijgt vaak als consumenten vertrouwen hebben."),
    ("XLP", "Basisconsumptie", "Dagelijkse benodigdheden die altijd verkocht worden: Procter & Gamble, Coca-Cola, Costco. Defensief: doet het relatief goed als de markt bang is."),
    ("XLE", "Energie", "Olie- en gasbedrijven, zoals ExxonMobil en Chevron. Beweegt sterk mee met de olieprijs."),
    ("XLF", "Financials", "Banken, verzekeraars en betaalbedrijven, zoals Berkshire Hathaway, JPMorgan en Visa."),
    ("XLV", "Gezondheidszorg", "Farmacie, biotech, medische apparatuur en zorgverzekeraars, zoals Eli Lilly en Johnson & Johnson."),
    ("XLI", "Industrie", "Luchtvaart, defensie, machines en transport, zoals GE Aerospace, Caterpillar en Uber."),
    ("XLB", "Grondstoffen", "Chemie, metalen, mijnbouw en verpakkingen, zoals Linde."),
    ("XLU", "Nutsbedrijven", "Stroom-, gas- en waterbedrijven. Normaal defensief, maar door de stroomhonger van AI-datacenters ook een AI-speler."),
    ("XLRE", "Vastgoed", "Vastgoedfondsen: kantoren, winkels, woningen, maar ook datacenters en zendmasten. Gevoelig voor de rente."),
    ("SMH", "Halfgeleiders", "De grootste chipbedrijven ter wereld, zoals NVIDIA, TSMC, Broadcom en ASML. Dé AI-chip-graadmeter."),
    ("URA", "Uranium & kernenergie", "Uraniummijnen en kernenergiebedrijven, zoals Cameco. Profiteert van de vraag naar kernstroom voor AI."),
    ("QTUM", "Quantum & AI", "Bedrijven in quantumcomputing en machine learning; breed gespreid, ook over grote techbedrijven."),
    ("UFO", "Ruimtevaart", "Satellieten, raketten en ruimtevaartdiensten, zoals Rocket Lab en AST SpaceMobile."),
    ("BOTZ", "Robotica & AI", "Robot- en automatiseringsbedrijven, zoals NVIDIA, ABB en Intuitive Surgical."),
    ("ITA", "Defensie & luchtvaart", "Amerikaanse defensie- en luchtvaartbedrijven, zoals GE Aerospace, RTX en Boeing."),
]
SECTOR_TICKERS = [t for t, _, _ in SECTOR_ETFS]
ETF_UITLEG = {t: u for t, _, u in SECTOR_ETFS}

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
    "NBIS": "Nebius Group",
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
    # Watchlist
    "ASML.AS": "ASML",
    "NVDA": "NVIDIA",
    "BABA": "Alibaba",
    "META": "Meta Platforms",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "AVGO": "Broadcom",
    "TSM": "TSMC",
    "BRK-B": "Berkshire Hathaway",
    "LLY": "Eli Lilly",
    "WMT": "Walmart",
    "JPM": "JPMorgan Chase",
    "2222.SR": "Saudi Aramco",
    "V": "Visa",
    "ORCL": "Oracle",
    "0700.HK": "Tencent",
    "XOM": "ExxonMobil",
    "MA": "Mastercard",
    "JNJ": "Johnson & Johnson",
    "ASM.AS": "ASM International",
    "HIMS": "Hims & Hers Health",
    "LMND": "Lemonade",
    "CRM": "Salesforce",
    "AMD": "AMD",
    "NOW": "ServiceNow",
    "PLTR": "Palantir",
    "LTC-USD": "Litecoin",
    "000660.KS": "SK Hynix",
    "005930.KS": "Samsung Electronics",
    "8035.T": "Tokyo Electron",
    "6857.T": "Advantest",
    "AMAT": "Applied Materials",
    "LRCX": "Lam Research",
    "ANET": "Arista Networks",
    "VRT": "Vertiv",
    "SU.PA": "Schneider Electric",
    "MOD": "Modine",
    "CEG": "Constellation Energy",
    "VST": "Vistra",
    "GEV": "GE Vernova",
    "CCJ": "Cameco",
    "ENR.DE": "Siemens Energy",
    "RR.L": "Rolls-Royce",
    "OKLO": "Oklo",
    "SMR": "NuScale Power",
    "IONQ": "IonQ",
    "RGTI": "Rigetti Computing",
    "QBTS": "D-Wave Quantum",
    "RKLB": "Rocket Lab",
    "ASTS": "AST SpaceMobile",
    "RHM.DE": "Rheinmetall",
    "AVAV": "AeroVironment",
    "SYM": "Symbotic",
    "ISRG": "Intuitive Surgical",
    "JOBY": "Joby Aviation",
}

NAMES.update({t: naam for t, naam, _ in SECTOR_ETFS})

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

# Zet op True zodra Yahoo 'Too Many Requests' teruggeeft; de pagina toont dan
# een duidelijke melding in plaats van stilletjes lege data.
RATE_LIMIT = {"geraakt": False, "tijd": None}


def _markeer_rate_limit():
    RATE_LIMIT["geraakt"] = True
    RATE_LIMIT["tijd"] = pd.Timestamp.now(tz="Europe/Amsterdam")


def _intraday_een(ticker: str) -> pd.Series:
    """5-minuten-closes (2 dagen, incl. voor- en nabeurs VS) voor één ticker."""
    try:
        hist = yf.Ticker(ticker).history(
            period="2d", interval="5m", auto_adjust=True, prepost=True,
        )
    except YFRateLimitError:
        _markeer_rate_limit()
        raise
    s = hist["Close"].dropna() if not hist.empty else pd.Series(dtype=float)
    if not s.empty and s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s


def _download_intraday(tickers: tuple) -> dict:
    """Intraday-data voor een lijst tickers, via de gedeelde pool (vast aantal
    threads). Niet via yf.download: die start intern één thread per ticker en
    is niet veilig om vanuit meerdere sessies tegelijk aan te roepen.
    prepost=True: voor- en nabeurskoersen worden in compute_row weer van de
    reguliere handel gescheiden."""
    if not tickers:
        return {}
    res = parallel(_intraday_een, tickers)
    return {t: (s if s is not None else pd.Series(dtype=float)) for t, s in zip(tickers, res)}


@st.cache_data(ttl=12, max_entries=4, show_spinner=False)
def fetch_intraday_batch(tickers: tuple) -> dict:
    """Intraday-koersen voor alle tickers, 12 sec bewaard (elke ververs-ronde vers)."""
    RATE_LIMIT["geraakt"] = False
    return _download_intraday(tickers)


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


def fetch_meta(ticker: str) -> dict:
    """Metadata met cache; lukt het ophalen niet (bv. rate limit bij Yahoo), dan
    een tijdelijke gok op basis van de tickernaam die NIET 7 dagen bewaard wordt."""
    try:
        return _fetch_meta_cached(ticker)
    except Exception:  # noqa: BLE001
        return _meta_terugval(ticker)


def _meta_terugval(ticker: str) -> dict:
    t = ticker.upper()
    crypto = looks_like_crypto(t)
    return {
        "naam": NAMES.get(t) or ticker, "crypto": crypto, "soort": None,
        "valuta": "EUR" if t.endswith(("-EUR", ".AS", ".PA", ".DE", ".BR")) else "USD",
        "tz": US_TZ if not crypto and "." not in t else None,
    }


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _fetch_meta_cached(ticker: str) -> dict:
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
    except YFRateLimitError:
        _markeer_rate_limit()
        raise  # niet cachen: straks opnieuw proberen
    except Exception:  # noqa: BLE001
        pass
    if not meta["naam"]:
        try:
            info = yf.Ticker(ticker).info or {}
            meta["soort"] = meta["soort"] or info.get("quoteType")
            meta["naam"] = info.get("longName") or info.get("shortName")
        except YFRateLimitError:
            _markeer_rate_limit()
            raise
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


MAX_THREADS = 6  # totaal voor de hele app, over alle sessies en fragmenten heen


@st.cache_resource
def _gedeelde_pool() -> ThreadPoolExecutor:
    """Eén vaste pool met een paar threads voor de hele app. Voorheen werd er
    bij elke verversing een nieuwe pool gemaakt en startte yf.download één
    thread per ticker; op Streamlit Cloud gaf dat 'can't start new thread'."""
    return ThreadPoolExecutor(max_workers=MAX_THREADS, thread_name_prefix="yahoo")


def parallel(fn, items) -> list:
    """fn(x) voor elke x, verdeeld over de gedeelde pool. Geeft per item het
    resultaat terug, of None als het mislukte."""
    items = list(items)
    if not items:
        return []
    ctx = get_script_run_ctx() if get_script_run_ctx else None

    def taak(x):
        if ctx is not None:
            add_script_run_ctx(threading.current_thread(), ctx)
        try:
            return fn(x)
        except Exception:  # noqa: BLE001
            return None

    if len(items) == 1:
        return [taak(items[0])]
    return list(_gedeelde_pool().map(taak, items))


def vooraf_ophalen(fn, items):
    """Roept een gecachte functie voor veel tickers tegelijk aan, zodat de cache
    gevuld is. Fouten worden genegeerd: die komen later per ticker netjes terug."""
    items = list(items)
    if len(items) >= 2:
        parallel(fn, items)


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
    try:
        hist = yf.Ticker(ticker).history(period="5y", interval="1d", auto_adjust=True)
    except YFRateLimitError:
        _markeer_rate_limit()
        raise ValueError("Yahoo: te veel verzoeken, probeer het over een paar minuten opnieuw")
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


CMC_FNG_HISTORICAL = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical"
FNG_HISTORIE_START = pd.Timestamp("2018-01-01", tz="UTC")


def _cmc_tijd(t) -> pd.Timestamp:
    """CMC-tijdstempel (seconden of milliseconden, als tekst of getal) -> datum."""
    t = float(t)
    return pd.Timestamp(t, unit="ms" if t > 1e11 else "s", tz="UTC").normalize().tz_localize(None)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_crypto_fear_greed_historie() -> pd.DataFrame:
    """Dagwaarden van de CoinMarketCap Fear & Greed (dezelfde bron als de meter),
    met de BTC-prijs erbij. Kolommen: datum, fng, btc. 6 uur bewaard.
    Eerst de grafiekdata van de CMC-website (gratis, met BTC-prijs), per jaar
    opgehaald; lukt dat niet, dan de officiële API (kost credits)."""
    rijen = []
    nu = pd.Timestamp.now(tz="UTC")
    try:
        van = FNG_HISTORIE_START
        while van < nu:
            tot = min(van + pd.Timedelta(days=365), nu)
            resp = requests.get(
                CMC_FNG_CHART,
                params={"start": int(van.timestamp()), "end": int(tot.timestamp())},
                headers={"User-Agent": CNN_HEADERS["User-Agent"], "Accept": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            for d in (resp.json().get("data") or {}).get("dataList") or []:
                btc = d.get("btcPrice")
                rijen.append({"datum": _cmc_tijd(d["timestamp"]), "fng": float(d["score"]),
                              "btc": float(btc) if btc not in (None, "") else None})
            van = tot
    except Exception:  # noqa: BLE001
        rijen = []
    sleutel = _cmc_api_key()
    if not rijen and sleutel:
        for pagina in range(10):  # 500 dagen per verzoek
            resp = requests.get(
                CMC_FNG_HISTORICAL, params={"start": 1 + pagina * 500, "limit": 500}, timeout=15,
                headers={"X-CMC_PRO_API_KEY": sleutel, "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json().get("data") or []
            rijen += [{"datum": _cmc_tijd(d["timestamp"]), "fng": float(d["value"]), "btc": None}
                      for d in data]
            if len(data) < 500:
                break
    if not rijen:
        raise ValueError("geen Fear & Greed-geschiedenis van CoinMarketCap")
    df = (pd.DataFrame(rijen).dropna(subset=["fng"])
          .sort_values("datum").drop_duplicates("datum", keep="last"))
    # BTC-prijs ontbreekt (API-route of losse dagen): aanvullen met Yahoo-dagkoersen
    if df["btc"].isna().any():
        try:
            yahoo = fetch_daily_closes("BTC-USD")
            yahoo.index = pd.to_datetime(yahoo.index).normalize()
            df["btc"] = df["btc"].fillna(df["datum"].map(yahoo))
        except Exception:  # noqa: BLE001
            pass
    return df.dropna(subset=["btc"]).reset_index(drop=True)


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


@st.cache_data(ttl=12 * 3600, max_entries=500, show_spinner=False)
def _yahoo_vorige_slot(ticker: str, handelsdag: str) -> float:
    """Yahoo's eigen vorige slotkoers (regularMarketPreviousClose): hetzelfde getal
    waarmee de Yahoo-website en brokers het dagpercentage berekenen.
    `handelsdag` (vandaag op de beurs) zit in de cache-sleutel, zodat er per
    handelsdag precies één keer opgehaald wordt. Fouten worden niet gecachet."""
    try:
        info = yf.Ticker(ticker).info or {}
    except YFRateLimitError:
        _markeer_rate_limit()
        raise
    ref = info.get("regularMarketPreviousClose") or info.get("previousClose")
    if not ref:
        raise ValueError("geen vorige slotkoers van Yahoo")
    # Controle: Yahoo's quote moet al over deze handelsdag gaan. Anders is
    # 'vorige slotkoers' nog die van een dag eerder (niet cachen, later opnieuw).
    t = info.get("regularMarketTime")
    tz = info.get("exchangeTimezoneName")
    if t and tz:
        quote_dag = pd.Timestamp(t, unit="s", tz="UTC").tz_convert(tz).date().isoformat()
        if quote_dag != handelsdag:
            raise ValueError("Yahoo-quote nog niet bijgewerkt voor vandaag")
    return float(ref)


def vorige_slotkoers(ticker: str, handelsdag) -> float | None:
    """Vorige slotkoers volgens Yahoo, of None als die (nog) niet op te halen is."""
    try:
        return _yahoo_vorige_slot(ticker, handelsdag.isoformat())
    except Exception:  # noqa: BLE001
        return None


def vorige_werkdag(dag):
    """Werkdag vóór `dag` (ma -> vr). Feestdagen kent dit niet."""
    dag -= timedelta(days=1)
    while dag.weekday() >= 5:
        dag -= timedelta(days=1)
    return dag


@st.cache_data(ttl=900, show_spinner=False)
def daily_stats(ticker: str) -> dict:
    """Alles wat alleen van de dagdata afhangt (5 jaar historie, MA's), 15 min
    bewaard. Scheelt bij elke verversing het opnieuw doorrekenen van de MA's."""
    daily = fetch_daily_closes(ticker)
    weekly = daily.resample("W").last().dropna()  # laatste bar = lopende week
    return {
        "daily": daily,
        "mas": {
            "20w": weekly.rolling(20).mean().iloc[-1],
            "50w": weekly.rolling(50).mean().iloc[-1],
            "200w": weekly.rolling(200).mean().iloc[-1],
            "200d": daily.rolling(200).mean().iloc[-1],
        },
    }


def compute_row(ticker: str, weights: dict, intraday_map: dict) -> dict:
    stats = daily_stats(ticker)
    daily = stats["daily"]
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

    # Marktstatus voor het lampje: open (reguliere handel), voor-/nabeurs, of dicht.
    if crypto:
        markt = "open"  # crypto handelt 24/7
    elif ext_sessie:
        markt = ext_sessie
    elif live:
        markt = "open"
    else:
        markt = "dicht"

    # 1 uur: alleen zinvol als er nu gehandeld wordt
    ch_1h = _pct(price, _at_or_before(intraday, last_ts - pd.Timedelta(hours=1))) if live else None

    # 24 uur: crypto echt 24u terug; aandelen t.o.v. slotkoers vorige handelsdag.
    # Aandelen: alleen als er vandaag (kalenderdag op de beurs) al gehandeld is.
    # Anders is het percentage van een vorige dag; dan tonen we alleen de prijs
    # (de slotkoers) en eventueel de voorbeurs.
    vandaag_beurs = now_utc.tz_convert(meta["tz"] or "Europe/Amsterdam").date()
    dag_actueel = crypto or last_date == vandaag_beurs
    ref = None
    if crypto and last_ts is not None:
        ref = _at_or_before(intraday, last_ts - pd.Timedelta(hours=24))
    elif dag_actueel:
        # Referentie = Yahoo's eigen vorige slotkoers (zoals op de website).
        ref = vorige_slotkoers(ticker, last_date)
        if ref is None:
            # Terugval: dagdata, maar alleen als die rij echt van de vorige
            # handelsdag is. Yahoo mist soms de dagbar van gisteren; dan zou
            # het percentage stilletjes over twee dagen gaan. Liever "–".
            prev = daily[daily.index.date < last_date]
            if not prev.empty and prev.index[-1].date() >= vorige_werkdag(last_date):
                ref = float(prev.iloc[-1])
    ch_24h = _pct(price, ref)
    ref_24h = ref

    # 7 dagen: slotkoers van (uiterlijk) een week geleden
    week_ago = daily[daily.index.date <= last_date - timedelta(days=7)]
    ref_7d = float(week_ago.iloc[-1]) if not week_ago.empty else None
    ch_7d = _pct(price, ref_7d)

    # 30 dagen: slotkoers van (uiterlijk) 30 dagen geleden
    month_ago = daily[daily.index.date <= last_date - timedelta(days=30)]
    ref_30d = float(month_ago.iloc[-1]) if not month_ago.empty else None
    ch_30d = _pct(price, ref_30d)

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

    # Aandelen alleen in dollar of euro tonen. Andere valuta (pond, Zwitserse
    # frank, Deense kroon, ...) -> hoofdprijs in dollar, euro als tweede.
    # Percentages en MA-afstanden blijven in de eigen valuta (die gaan over het
    # aandeel zelf, niet over de wisselkoers).
    weergave_valuta, prijs_omgerekend = currency, False
    weergave_prijs, weergave_ext = price, ext_prijs
    if not crypto and currency not in ("USD", "EUR"):
        usd = omgerekend_prijzen["USD"][0]
        if usd is not None:
            factor = usd / price if price else None
            weergave_valuta, weergave_prijs, prijs_omgerekend = "USD", usd, True
            if ext_prijs is not None and factor:
                weergave_ext = ext_prijs * factor

    # Referentiekoersen in de weergavevaluta, voor de stijging/daling in geld.
    f = weergave_prijs / price if price else None
    refs = {k: (v * f if v is not None and f else None)
            for k, v in (("24u", ref_24h), ("7d", ref_7d), ("30d", ref_30d))}

    mas = stats["mas"]
    pcts = {
        k: None if pd.isna(v) else round((price - float(v)) / float(v) * 100, 2)
        for k, v in mas.items()
    }
    # Lager onder de MA = hogere score. Ontbrekende MA's tellen niet mee.
    score = sum(-p * weights[k] for k, p in pcts.items() if p is not None)

    return {
        "ticker": ticker,
        "naam": meta["naam"],
        "prijs": weergave_prijs,
        "prijs_omgerekend": prijs_omgerekend,
        "ext_sessie": ext_sessie,
        "ext_prijs": weergave_ext,
        "%_ext": ext_pct,
        "valuta": weergave_valuta,
        "eigen_valuta": currency,
        "dag_actueel": dag_actueel,
        "markt": markt,
        "bijgewerkt": updated,
        "%_1u": ch_1h,
        "%_24u": ch_24h,
        "%_7d": ch_7d,
        "%_30d": ch_30d,
        **{f"ref_{k}": v for k, v in refs.items()},
        "prijs_eur": omgerekend_prijzen["EUR"][0],
        "eur_omgerekend": omgerekend_prijzen["EUR"][1],
        "prijs_usd": omgerekend_prijzen["USD"][0],
        "usd_omgerekend": omgerekend_prijzen["USD"][1],
        **{f"%_vs_{k}": p for k, p in pcts.items()},
        "score": round(score, 1),
    }


def build_ranking(tickers: list, weights: dict):
    vooraf_ophalen(fetch_meta, tickers)
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

    vooraf_ophalen(daily_stats, tickers)  # 5 jaar historie + MA's, gelijktijdig
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


def hoofdprijs(row: pd.Series, kolom: str = "prijs") -> str:
    """Prijs in de weergavevaluta; '≈' als hij uit een andere valuta is omgerekend."""
    approx = "≈ " if row.get("prijs_omgerekend") is True else ""
    return approx + format_price(row[kolom], row["valuta"])


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
        f'<span style="opacity:0.85">{hoofdprijs(row, "ext_prijs")}</span>'
        f"{pct_html}</span>"
    )


LAMPJES = {  # status: (kleur, pulseren, uitleg)
    "open": ("#2e9e58", True, "Live: er wordt nu gehandeld"),
    "pre": ("#e0a030", True, "Voorbeurs: handel buiten de reguliere uren"),
    "post": ("#e0a030", True, "Nabeurs: handel buiten de reguliere uren"),
    "dicht": ("#888888", False, "Beurs dicht: laatste koers is de slotkoers"),
}


VOORBEURS_KLEUR = LAMPJES["pre"][0]  # zelfde oranje als het lampje


ZON_SVG = (
    '<svg width="12" height="12" viewBox="0 0 24 24" style="vertical-align:-1px">'
    '<circle cx="12" cy="12" r="5" fill="currentColor"/>'
    '<g stroke="currentColor" stroke-width="2.4" stroke-linecap="round">'
    '<path d="M12 1.5v3M12 19.5v3M1.5 12h3M19.5 12h3M4.6 4.6l2.1 2.1M17.3 17.3l2.1 2.1'
    'M4.6 19.4l2.1-2.1M17.3 6.7l2.1-2.1"/></g></svg>'
)
MAAN_SVG = (
    '<svg width="11" height="11" viewBox="0 0 24 24" style="vertical-align:-1px">'
    '<path fill="currentColor" d="M21 14.5A9 9 0 0 1 9.5 3a9 9 0 1 0 11.5 11.5z"/></svg>'
)
NABEURS_KLEUR = "#5b9bd5"
SESSIE_STIJL = {  # sessie: (label, kleur, icoon)
    "pre": ("PRE", VOORBEURS_KLEUR, ZON_SVG),
    "post": ("POST", NABEURS_KLEUR, MAAN_SVG),
}
DAG_KOLOMMEN = ("%_24u", "%_dag_rang")


def _getal(x) -> bool:
    return x is not None and pd.notna(x)


def _bedrag(x: float, decimalen: int) -> str:
    """Verandering in geld, met teken, Nederlandse notatie (geen valutateken)."""
    txt = f"{x:+,.{decimalen}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return txt.replace("-", "−")


def _procent(x: float) -> str:
    return f"{x:+.2f}%".replace(".", ",").replace("-", "−")


def verandering_html(prijs, ref, pct, decimalen: int) -> str:
    """'+15,68  +1,83%' in groen of rood. Zonder referentie alleen het percentage."""
    kleur = "#2e9e58" if pct >= 0 else "#d64545"
    delen = []
    if _getal(ref) and _getal(prijs):
        delen.append(_bedrag(prijs - ref, decimalen))
    delen.append(_procent(pct))
    return f'<span style="color:{kleur};font-weight:600">{"&nbsp;&nbsp;".join(delen)}</span>'


def koers_blok_html(row: pd.Series, col: str = "%_24u") -> str:
    """Rechterkant van een regel, in TradingView-stijl:
    prijs groot; eronder de verandering in geld en procent; tijdens voor-/nabeurs
    een extra regel met zon (PRE) of maan (POST), de prijs daar en de beweging
    t.o.v. de slotkoers. In de voorbeurs staat de dag van gisteren er níet bij."""
    prijs = row["prijs"]
    dec = 2 if prijs >= 1 else 4
    regel = ('<div style="font-size:0.74rem;white-space:nowrap;margin-top:2px;'
             'font-variant-numeric:tabular-nums">')
    regels = []
    if col in DAG_KOLOMMEN:
        dag = row.get("%_24u")
        if _getal(dag):
            regels.append(regel + verandering_html(prijs, row.get("ref_24u"), dag, dec) + "</div>")
        sessie, e = row.get("ext_sessie"), row.get("%_ext")
        if isinstance(sessie, str) and sessie in SESSIE_STIJL and _getal(e):
            label, kleur, icoon = SESSIE_STIJL[sessie]
            uitleg = EXT_LABELS[sessie][1]
            ext_prijs = row.get("ext_prijs")
            regels.append(
                f'{regel}<span title="{uitleg}" style="color:{kleur};font-weight:800;'
                f'font-size:0.92em">{icoon} {label}</span>&nbsp;'
                f'<span style="opacity:0.85">{hoofdprijs(row, "ext_prijs")}</span>&nbsp;&nbsp;'
                f'{verandering_html(ext_prijs, prijs, e, dec)}</div>'
            )
        elif not _getal(dag) and row.get("dag_actueel") is False:
            regels.append(f'{regel}<span style="opacity:0.5">slotkoers · beurs dicht</span></div>')
    else:
        pct = row.get(col)
        if _getal(pct):
            regels.append(regel + verandering_html(prijs, row.get("ref" + col[1:]), pct, dec) + "</div>")
    return (
        '<div style="text-align:right;flex-shrink:0;line-height:1.15">'
        f'<div style="font-weight:750;font-size:0.98rem;font-variant-numeric:tabular-nums;'
        f'white-space:nowrap">{hoofdprijs(row)}</div>{"".join(regels)}</div>'
    )


def lampje_html(row: pd.Series) -> str:
    """Klein rondje vóór de naam: groen = live, oranje = voor-/nabeurs, grijs = dicht."""
    status = row.get("markt")
    if not isinstance(status, str) or status not in LAMPJES:
        return ""
    kleur, puls, uitleg = LAMPJES[status]
    return (
        f'<span title="{uitleg}" class="{"lampje-puls" if puls else ""}" '
        f'style="display:inline-block;width:7px;height:7px;border-radius:50%;'
        f'background:{kleur};color:{kleur};flex-shrink:0;align-self:center;'
        f'{"" if puls else "opacity:0.6;"}"></span>'
    )


def title_html(row: pd.Series, size: str = "0.95rem", stacked: bool = False, tag: str = "",
               extra: str = "") -> str:
    """Naam als hoofdtekst, de ticker klein en gedempt erachter (of eronder)."""
    naam = html.escape(str(row.get("naam") or row["ticker"]))
    richting = "flex-direction:column;gap:0;line-height:1.2" if stacked else "align-items:baseline;gap:6px"
    tag_html = (
        f'<span style="font-size:0.62rem;padding:0 6px;border-radius:999px;white-space:nowrap;'
        f'background:rgba(46,158,88,0.14);color:#2e9e58;font-weight:600">{tag}</span>'
        if tag else ""
    )
    lampje = lampje_html(row)
    naam_html = (
        f'<span style="display:inline-flex;align-items:center;gap:6px;min-width:0">{lampje}'
        f'<span style="font-weight:650;font-size:{size};white-space:nowrap;overflow:hidden;'
        f'text-overflow:ellipsis">{naam}</span></span>'
    )
    return (
        f'<div style="display:flex;{richting};min-width:0">'
        f'{naam_html}'
        f'<span style="font-size:0.7rem;opacity:0.5;white-space:nowrap">{row["ticker"]}{extra}</span>'
        f"{tag_html}</div>"
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


def card_html(rank: int, row: pd.Series, in_bezit: bool = False) -> str:
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
        f'<div style="flex:1;min-width:0">{title_html(row, tag="in bezit" if in_bezit else "")}'
        # prijs met euro-prijs en eventueel voor-/nabeurs ernaast
        f'<div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-top:1px">'
        f'<span style="font-weight:700;font-size:0.92rem;font-variant-numeric:tabular-nums">'
        f'{hoofdprijs(row)}</span>{andere}{ext_chip_html(row)}</div></div>'
        f'{score_html(row["score"])}'
        f"</div>"
        # wijzigingen + tijdstip op één regel
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:baseline;font-size:0.72rem;'
        f'margin:4px 0 3px 22px">{chips}'
        f'<span style="margin-left:auto;opacity:0.45;font-variant-numeric:tabular-nums">'
        f'{row["bijgewerkt"]}</span></div>'
        f'<div style="margin-left:22px">{bars}</div></div>'
    )


def holding_card_html(row: pd.Series) -> str:
    """Kaartje voor een positie in de portefeuille: koers in TradingView-stijl
    rechts, koersverloop (1u/7d/30d) onderaan, geen score."""
    chips = "".join(
        change_chip(lbl, row[col]) for lbl, col in [("1u", "%_1u"), ("7d", "%_7d"), ("30d", "%_30d")]
    )
    andere = " · ".join(other_prices(row))
    andere_html = (
        f'<div style="font-size:0.78rem;opacity:0.6;font-variant-numeric:tabular-nums;'
        f'margin-top:1px">{andere}</div>' if andere else ""
    )
    return (
        f'<div style="border:1px solid rgba(128,128,128,0.22);border-radius:8px;'
        f'padding:7px 10px 6px;margin-bottom:6px">'
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="flex:1;min-width:0">{title_html(row)}{andere_html}</div>'
        f'{koers_blok_html(row, "%_24u")}'
        f"</div>"
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:baseline;font-size:0.72rem;'
        f'margin-top:5px">{chips}'
        f'<span style="margin-left:auto;opacity:0.45;font-variant-numeric:tabular-nums">'
        f'{row["bijgewerkt"]}</span></div></div>'
    )


# Per soort een eigen label voor de korte periode: crypto handelt 24/7,
# aandelen vergelijken we met de vorige slotkoers.
MOVER_PERIODES = {
    False: {"Dag": "%_24u", "7d": "%_7d", "30d": "%_30d"},
    True: {"24u": "%_24u", "7d": "%_7d", "30d": "%_30d"},
}
MOVERS_TOP = 3


def mover_row_html(rank: int, row: pd.Series, col: str) -> str:
    """Compacte regel: rang, naam (ticker en euro/dollar-prijs klein eronder),
    rechts de koers in TradingView-stijl voor de gekozen periode."""
    uitleg = ETF_UITLEG.get(row["ticker"])
    info = (
        '<span title="Klik voor uitleg" style="display:inline-flex;align-items:center;justify-content:center;'
        'width:14px;height:14px;border-radius:50%;border:1px solid currentColor;font-size:0.6rem;'
        'font-weight:700;opacity:0.5;margin-left:5px;flex-shrink:0;cursor:pointer">i</span>'
        if uitleg else ""
    )
    andere = "".join(f" · {p}" for p in other_prices(row))
    regel = (
        f'<div style="display:flex;align-items:center;gap:8px;padding:5px 9px">'
        f'<span style="opacity:0.45;font-size:0.75rem;width:14px;flex-shrink:0">{rank}</span>'
        f'<div style="flex:1;min-width:0;display:flex;align-items:center">'
        f'<div style="min-width:0">{title_html(row, "0.88rem", stacked=True, extra=andere)}</div>{info}</div>'
        f'{koers_blok_html(row, col)}</div>'
    )
    kader = 'border:1px solid rgba(128,128,128,0.18);border-radius:7px;margin-bottom:4px'
    if not uitleg:
        return f'<div style="{kader}">{regel}</div>'
    # <details>: klik op de regel klapt de uitleg open, zonder JavaScript
    return (
        f'<details class="etf-info" style="{kader}"><summary style="list-style:none;cursor:pointer">{regel}</summary>'
        f'<div style="padding:0 10px 7px 31px;font-size:0.78rem;opacity:0.75;line-height:1.4">'
        f"{html.escape(uitleg)}</div></details>"
    )


def render_mover_list(titel: str, part: pd.DataFrame, col: str, leeg: str, alles: bool):
    """Top 3, of met 'alles' de hele lijst als één doorlopende rij kaartjes."""
    st.markdown(f"**{titel}**")
    if part.empty:
        st.caption(leeg)
        return
    rows = [mover_row_html(i + 1, r, col)
            for i, (_, r) in enumerate(part.iterrows())]
    zichtbaar = rows if alles else rows[:MOVERS_TOP]
    stijl = "<style>details.etf-info summary::-webkit-details-marker{display:none}</style>"
    st.markdown(stijl + "".join(zichtbaar), unsafe_allow_html=True)
    if not alles and len(rows) > MOVERS_TOP:
        st.caption(f"Nog {len(rows) - MOVERS_TOP} meer, zet 'Toon alle' aan om ze te zien.")


def _dag_rangwaarde(row: pd.Series):
    """Waarde waarop de dag-lijst sorteert:
    - voorbeurs (nog geen handel vandaag): alleen de voorbeurs;
    - nabeurs: reguliere dag en nabeurs samen (t.o.v. de vorige slotkoers);
    - anders: de gewone dagverandering (crypto: 24 uur)."""
    dag, e, sessie = row.get("%_24u"), row.get("%_ext"), row.get("ext_sessie")
    heeft_dag = dag is not None and pd.notna(dag)
    heeft_ext = isinstance(sessie, str) and e is not None and pd.notna(e)
    if heeft_ext and sessie == "post" and heeft_dag:
        return ((1 + dag / 100) * (1 + e / 100) - 1) * 100
    if heeft_ext and sessie == "pre" and not heeft_dag:
        return e
    return dag if heeft_dag else None


def _mover_blok(data: pd.DataFrame, col: str, periode: str, alles: bool):
    """Stijgers links, dalers rechts, gesorteerd op de gekozen periode."""
    if col == "%_24u":
        data = data.assign(**{"%_dag_rang": data.apply(_dag_rangwaarde, axis=1)})
        col = "%_dag_rang"
    data = data.dropna(subset=[col])
    stijgers = data[data[col] > 0].sort_values(col, ascending=False)
    dalers = data[data[col] < 0].sort_values(col, ascending=True)
    left, right = st.columns(2, gap="medium")
    with left:
        render_mover_list(f"Stijgers ({periode})", stijgers, col,
                          "Geen stijgers in deze periode.", alles)
    with right:
        render_mover_list(f"Dalers ({periode})", dalers, col,
                          "Geen dalers in deze periode.", alles)


def render_movers(ranking: pd.DataFrame, crypto: bool, etfs=None):
    """Hardste stijgers & dalers. Met `etfs` eerst een blok sector-/thema-ETF's,
    daarna de rest; beide volgen dezelfde periodeknop en 'Toon alle'."""
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
        alles = st.toggle("Toon alle", value=True, key="mover_alles")
    col = perioden[periode]
    if col == "%_24u":
        st.caption("Laatste 24 uur." if crypto else
                   "Reguliere handel t.o.v. de vorige slotkoers. Voorbeurs (VS): gesorteerd op de "
                   "voorbeurs. Nabeurs: dag en nabeurs staan apart, gesorteerd op beide samen.")

    if etfs is not None and not etfs.empty:
        st.markdown("##### Sectoren & thema's")
        st.caption("Amerikaanse sector- en thema-ETF's. Klik op een regel met ⓘ voor uitleg.")
        _mover_blok(etfs, col, periode, alles)
        st.markdown("##### Aandelen")
    _mover_blok(ranking, col, periode, alles)


# Ophalen op de achtergrond ("stale-while-revalidate"):
# - De pagina leest altijd de laatst opgehaalde data uit een gedeeld geheugen;
#   dat is direct klaar, dus de pagina wordt niet grijs en loopt niet vast.
# - Is die data ouder dan VERS_SEC, dan start er op de achtergrond één
#   ophaal-taak. Bij de volgende verversing staat de nieuwe data klaar.
# - Alleen de allereerste keer (nog niets in het geheugen) wordt er gewacht.
VERS_SEC = 10
VERGEET_NA_SEC = 15 * 60  # data die zo lang niet gelezen is wordt opgeruimd


@st.cache_resource
def _data_geheugen() -> dict:
    return {"lock": threading.Lock(), "data": {}, "bezig": set()}


def _ophalen_en_bewaren(key: tuple):
    """Haalt de ranking op en zet hem in het gedeelde geheugen."""
    geh = _data_geheugen()
    tickers, gewichten = key
    try:
        ranking, errors = build_ranking(list(tickers), dict(gewichten))
    except Exception as e:  # noqa: BLE001  (oude data blijft staan)
        with geh["lock"]:
            oud = geh["data"].get(key)
            if oud:
                oud["errors"] = [f"Verversen mislukt: {e}"]
        return
    nu = time.time()
    with geh["lock"]:
        oud = geh["data"].get(key, {})
        geh["data"][key] = {"ranking": ranking, "errors": errors, "tijd": nu,
                            "gelezen": oud.get("gelezen", nu)}
        for k in [k for k, v in geh["data"].items() if nu - v["gelezen"] > VERGEET_NA_SEC]:
            del geh["data"][k]


def _achtergrond(key: tuple):
    geh = _data_geheugen()
    ctx = get_script_run_ctx() if get_script_run_ctx else None

    def taak():
        if ctx is not None:
            add_script_run_ctx(threading.current_thread(), ctx)
        try:
            _ophalen_en_bewaren(key)
        finally:
            with geh["lock"]:
                geh["bezig"].discard(key)

    with geh["lock"]:
        if key in geh["bezig"]:
            return  # er loopt al een ophaal-taak voor deze lijst
        geh["bezig"].add(key)
    threading.Thread(target=taak, daemon=True, name="verversen").start()


def _ranking_uit_geheugen(tickers: tuple, gewichten: tuple):
    key = (tickers, gewichten)
    geh = _data_geheugen()
    with geh["lock"]:
        entry = geh["data"].get(key)
    if entry is None:  # eerste keer: wachten tot de data er is
        with st.spinner("Koersen ophalen…"):
            _ophalen_en_bewaren(key)
        with geh["lock"]:
            entry = geh["data"].get(key)
        if entry is None:
            return pd.DataFrame(), ["Ophalen mislukt"], time.time()
    entry["gelezen"] = time.time()
    if time.time() - entry["tijd"] > VERS_SEC:
        _achtergrond(key)
    return entry["ranking"], entry["errors"], entry["tijd"]


def wis_data_geheugen():
    geh = _data_geheugen()
    with geh["lock"]:
        geh["data"].clear()


def laad_ranking(tickers: list, weights: dict, met_etfs: bool):
    """Koersen + berekening voor een lijst tickers, uit het gedeelde geheugen
    (zie hierboven). Met `met_etfs` komen de sector-ETF's mee en worden ze
    apart teruggegeven. Geeft ook het tijdstip van de data terug."""
    alles = tuple(dict.fromkeys(list(tickers) + (SECTOR_TICKERS if met_etfs else [])))
    ranking, errors, tijd = _ranking_uit_geheugen(alles, tuple(sorted(weights.items())))
    etfs = None
    if met_etfs and not ranking.empty:
        is_etf = ranking["ticker"].isin(SECTOR_TICKERS) & ~ranking["ticker"].isin(tickers)
        etfs, ranking = ranking[is_etf], ranking[~is_etf].reset_index(drop=True)
    return ranking, etfs, errors, tijd


def _klok(tijd: float) -> str:
    return pd.Timestamp(tijd, unit="s", tz="UTC").tz_convert("Europe/Amsterdam").strftime("%H:%M:%S")


# --- 4-jaarcyclus ------------------------------------------------------------
# Halving -> top was 526, 548 en 534 dagen (2016, 2020, 2024; de eerste cyclus
# van 2012 met 367 dagen laten we weg). Gemiddeld 536 dagen. Het theoretische
# einde van de bullrun = halving + 536 dagen. Top -> top (1.424 en 1.426 dagen)
# komt voor 2029 op vrijwel hetzelfde uit.
HALVING_NAAR_TOP = 536
HALVINGS = {
    2024: pd.Timestamp("2024-04-20"),
    2028: pd.Timestamp("2028-04-17"),  # SCHATTING op basis van het bloktempo; bijwerken als hij er is
}
KERN_MARGE = (pd.Timedelta(days=30), pd.Timedelta(days=30))       # kernzone rond het einde
AANDACHT_MARGE = (pd.Timedelta(days=180), pd.Timedelta(days=75))  # eerste top kan ~half jaar eerder (2013, 2021)


def cyclus_zones() -> list:
    """De laatste en de aankomende cyclus: theoretisch einde, kern- en aandachtszone."""
    zones = []
    for jaar, halving in sorted(HALVINGS.items()):
        einde = halving + pd.Timedelta(days=HALVING_NAAR_TOP)
        zones.append({
            "halving": halving, "einde": einde,
            "kern_van": einde - KERN_MARGE[0], "kern_tot": einde + KERN_MARGE[1],
            "aandacht_van": einde - AANDACHT_MARGE[0], "aandacht_tot": einde + AANDACHT_MARGE[1],
        })
    return zones


NL_MAANDEN = ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]


def _nl_datum(d: pd.Timestamp, dag: bool = True) -> str:
    """'5 okt 2029' of 'okt 2029' (los van de taalinstelling van de server)."""
    return (f"{d.day} " if dag else "") + f"{NL_MAANDEN[d.month - 1]} {d.year}"


def _aftelling(van: pd.Timestamp, tot: pd.Timestamp) -> str:
    """'2 jaar, 11 maanden en 12 dagen' (of minder delen als die 0 zijn)."""
    from dateutil.relativedelta import relativedelta
    r = relativedelta(tot.date(), van.date())
    delen = [(r.years, "jaar", "jaar"), (r.months, "maand", "maanden"), (r.days, "dag", "dagen")]
    tekst = [f"{n} {enk if n == 1 else mv}" for n, enk, mv in delen if n]
    if not tekst:
        return "vandaag"
    return tekst[0] if len(tekst) == 1 else ", ".join(tekst[:-1]) + " en " + tekst[-1]


def render_cyclus_kaart():
    """Aftelling naar het theoretische einde van de bullrun, met een balkje
    voor waar we tussen het vorige en het volgende theoretische einde staan."""
    vandaag = pd.Timestamp.now(tz="Europe/Amsterdam").tz_localize(None).normalize()
    zones = cyclus_zones()
    komend = next((z for z in zones if z["aandacht_tot"] >= vandaag), zones[-1])
    vorig = next((z for z in reversed(zones) if z["einde"] < komend["einde"]), None)

    if vandaag < komend["aandacht_van"]:
        fase, kleur = "opbouwfase: plan je exit-strategie, nog geen haast", "#2e9e58"
    elif vandaag < komend["kern_van"]:
        fase, kleur = "aandachtszone: let op een eerste top en bearish divergenties", "#ee8a35"
    elif vandaag <= komend["kern_tot"]:
        fase, kleur = "kernzone: historisch lag hier de top, voer je exit-plan uit", "#d64545"
    else:
        fase, kleur = "na de kernzone: historisch begint hier de bear market", "#d64545"

    if vandaag <= komend["einde"]:
        kop = (f"Nog <b>{_aftelling(vandaag, komend['einde'])}</b> tot het theoretische "
               f"einde van de bullrun (± {_nl_datum(komend['einde'])})")
    else:
        kop = (f"Theoretisch einde van de bullrun was {_aftelling(komend['einde'], vandaag)} "
               f"geleden (± {_nl_datum(komend['einde'])})")

    balk = ""
    if vorig is not None:
        totaal = (komend["einde"] - vorig["einde"]).days
        pct = min(max((vandaag - vorig["einde"]).days / totaal, 0), 1) * 100
        a = (komend["aandacht_van"] - vorig["einde"]).days / totaal * 100
        k = (komend["kern_van"] - vorig["einde"]).days / totaal * 100
        balk = (
            '<div style="position:relative;height:8px;border-radius:4px;margin:8px 0 3px;'
            'background:rgba(128,128,128,0.18);overflow:hidden">'
            f'<div style="position:absolute;left:{a:.1f}%;right:0;top:0;bottom:0;background:rgba(214,69,69,0.18)"></div>'
            f'<div style="position:absolute;left:{k:.1f}%;right:0;top:0;bottom:0;background:rgba(214,69,69,0.35)"></div>'
            f'<div style="position:absolute;left:0;width:{pct:.1f}%;top:0;bottom:0;background:#2a78d6;opacity:0.75"></div>'
            '</div>'
            '<div style="display:flex;justify-content:space-between;font-size:0.68rem;opacity:0.55">'
            f'<span>einde vorige run (± {_nl_datum(vorig["einde"], dag=False)})</span>'
            f'<span>{pct:.0f}% van de cyclus</span>'
            f'<span>± {_nl_datum(komend["einde"], dag=False)}</span></div>'
        )
    st.markdown(
        '<div style="border:1px solid rgba(128,128,128,0.22);border-radius:8px;padding:9px 12px;margin:4px 0 8px">'
        f'<div style="font-size:0.95rem">⏳ {kop}</div>'
        f'<div style="font-size:0.78rem;margin-top:2px"><span style="color:{kleur};font-weight:700">●</span> '
        f'<span style="opacity:0.75">{fase}</span></div>{balk}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Op basis van de 4-jaarcyclus: halving + {HALVING_NAAR_TOP} dagen (gemiddelde van de "
        "laatste drie cycli). Een richtlijn, geen voorspelling. Halving 2028 is een schatting."
    )


FNG_PERIODES = {"1 jaar": 365, "2 jaar": 730, "4 jaar": 1460, "Alles": None}
RSI_PERIODE = 14


def rsi(closes: pd.Series, n: int = RSI_PERIODE) -> pd.Series:
    """RSI volgens Wilder (zoals TradingView): gemiddelde stijging vs. daling
    over n perioden, met Wilder-smoothing (alpha = 1/n)."""
    delta = closes.diff()
    stijg = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    daal = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + stijg / daal)


def btc_dagdata(df: pd.DataFrame) -> pd.DataFrame:
    """Zet BTC-prijs én RSI in df op dezelfde basis: de dagslotkoersen van Yahoo
    (dag sluit om 00:00 UTC, net als op TradingView). De prijs van CMC is een
    momentopname op een ander tijdstip van de dag; als we die tonen maar de RSI
    op Yahoo-slotkoersen berekenen, lopen prijs en RSI niet precies gelijk.
    De RSI wordt over de volledige 5 jaar berekend, zodat hij aan het begin van
    de grafiek al ingelopen is. Lukt Yahoo niet, dan alles op de CMC-prijs."""
    try:
        yahoo = fetch_daily_closes("BTC-USD").copy()
        yahoo.index = pd.to_datetime(yahoo.index).normalize()
        yahoo = yahoo[~yahoo.index.duplicated(keep="last")]
        prijs = df["datum"].map(yahoo)
        if prijs.notna().mean() > 0.9:
            return df.assign(btc=prijs.fillna(df["btc"]), rsi=df["datum"].map(rsi(yahoo)),
                             bron="Yahoo-slotkoersen")
    except Exception:  # noqa: BLE001
        pass
    return df.assign(rsi=rsi(df.set_index("datum")["btc"]).to_numpy(), bron="CoinMarketCap")


# Weeks-MA's in de BTC-prijsgrafiek: (aantal weken, kleur van de lijn)
BTC_WEEK_MAS = {"ma50w": (50, "#f2a900"), "ma200w": (200, "#2e9e58")}


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def btc_week_mas() -> pd.DataFrame:
    """50w- en 200w-MA van BTC op weekslotkoersen (hele historie), 12 uur bewaard.
    Wordt alleen opgehaald als een van de vinkjes aan staat."""
    weekly = fetch_weekly_max("BTC-USD")
    return pd.DataFrame({k: weekly.rolling(w).mean() for k, (w, _) in BTC_WEEK_MAS.items()})


def met_week_mas(df: pd.DataFrame, kolommen: list) -> pd.DataFrame:
    """Zet de gekozen week-MA's als kolom bij de dagdata. Elke dag krijgt de MA van
    de week waar hij in valt (trapjes, zoals een weeks-MA op een daggrafiek)."""
    mas = btc_week_mas()
    dagen = mas.reindex(df["datum"], method="bfill")  # weeklabel = zondag van die week
    return df.assign(**{k: dagen[k].to_numpy() for k in kolommen})


GRAFIEK_CSS_TELEFOON = """<style>
/* Zo breed mogelijk: de as-getallen rechts steken ~35px buiten het tekengebied uit
   en vallen zo net binnen de rand van het uitklapblok. Zo ver mogelijk naar links. */
.st-key-fng_grafiek { width: calc(100% - 2px) !important; max-width: calc(100% - 2px) !important;
                      margin-left: -18px !important; position: relative; }
/* Schakelaar 'Aflezen' rechtsboven in de prijsgrafiek, boven de hoogste prijs */
.st-key-fng_lock { position: absolute !important; top: 2px; right: 44px; z-index: 5;
                   width: auto !important; }
.st-key-fng_lock label p { font-size: 0.8rem; }
</style>"""
# Uit (standaard): aanraken is voor de pagina, overal vrij scrollen
GRAFIEK_CSS_SCROLLEN = """<style>
.st-key-fng_grafiek .vega-embed, .st-key-fng_grafiek canvas, .st-key-fng_grafiek svg { touch-action: auto; }
</style>"""
# Aan: aanraken is voor de grafiek (tikken/slepen), de pagina scrolt daar niet
GRAFIEK_CSS_AFLEZEN = """<style>
.st-key-fng_grafiek .vega-embed, .st-key-fng_grafiek canvas, .st-key-fng_grafiek svg { touch-action: none; }
</style>"""

# Vega-expressies voor Nederlandse datums (los van de taal van de browser)
_NL_MND = "['jan','feb','mrt','apr','mei','jun','jul','aug','sep','okt','nov','dec']"
# As-labels: '8 mrt' voor tussenliggende dagen, 'mrt' op de 1e, 'jan 2026' bij een nieuw jaar
X_LABEL_EXPR = (
    f"(date(datum.value) == 1 ? '' : date(datum.value) + ' ') + {_NL_MND}[month(datum.value)]"
    " + (month(datum.value) == 0 && date(datum.value) == 1 ? ' ' + year(datum.value) : '')"
)
DATUM_EXPR = f"date(datum.datum) + ' ' + {_NL_MND}[month(datum.datum)] + ' ' + year(datum.datum)"


def _y_domein(reeks: pd.Series, marge: float = 3, lo: float = 0, hi: float = 100) -> list:
    """y-as die meeschaalt met wat er te zien is (binnen 0–100)."""
    r = reeks.dropna()
    if r.empty:
        return [lo, hi]
    a, b = max(lo, float(r.min()) - marge), min(hi, float(r.max()) + marge)
    return [a, b] if b > a else [lo, hi]


def render_fng_grafiek():
    """BTC-prijs, CMC Fear & Greed en RSI op dagbasis, onder elkaar met een
    gedeelde datum-as. Twee weergaven (schakelaar bovenaan, standaard telefoon):
    - telefoon: periode kiezen met een schuifbalk met twee uiteinden, y-assen
      schalen mee met wat je ziet; tikken/slepen kiest een dag, het punt blijft
      staan, de waarden staan bij elk paneel;
    - pc: periodeknoppen, zoomen/schuiven met de muis, tooltip bij de muis.
    Draait als eigen fragment: alleen de grafiek ververst bij een klik."""
    import altair as alt  # wordt met Streamlit meegeleverd

    try:
        df = fetch_crypto_fear_greed_historie()
    except Exception as e:  # noqa: BLE001
        st.warning(f"Fear & Greed-geschiedenis niet opgehaald: {e}")
        return
    if df.empty:
        st.info("Geen data.")
        return
    # Prijs en RSI op dezelfde slotkoersen; RSI over de hele historie berekenen,
    # anders loopt hij bij elk venster opnieuw in.
    df = btc_dagdata(df)
    prijsbron = df["bron"].iloc[0]
    data_van, data_tot = df["datum"].min(), df["datum"].max()
    zones = cyclus_zones()

    k1, k2 = st.columns(2)
    with k1:
        pc = st.toggle("🖥️ Pc-weergave", key="fng_pc",
                       help="Uit = telefoon (aanraken en schuifbalk). "
                            "Aan = pc (zoomen en schuiven met de muis).")
    with k2:
        toon_cyclus = st.checkbox("Theoretisch einde bullrun", key="fng_cyclus",
                                  help="Kernzone (donker) en aandachtszone (licht) van de "
                                       "vorige en de aankomende cyclus.")
    m1, m2 = st.columns(2)
    with m1:
        toon_ma50 = st.checkbox(":orange[50w MA]", key="fng_ma50", value=False,
                                help="50-weeks voortschrijdend gemiddelde (weekslotkoersen).")
    with m2:
        toon_ma200 = st.checkbox(":green[200w MA]", key="fng_ma200", value=False,
                                 help="200-weeks voortschrijdend gemiddelde (weekslotkoersen).")
    ma_kolommen = [k for k, aan in (("ma50w", toon_ma50), ("ma200w", toon_ma200)) if aan]
    if ma_kolommen:
        try:
            df = met_week_mas(df, ma_kolommen)
        except Exception as e:  # noqa: BLE001
            st.caption(f"Week-MA's niet opgehaald: {e}")
            ma_kolommen = []

    as_tot = data_tot
    if toon_cyclus:
        as_tot = max(data_tot, max(z["aandacht_tot"] for z in zones) + pd.Timedelta(days=150))

    # --- welk stuk van de tijd is te zien -----------------------------------
    if pc:
        keuze = st.radio("Periode", list(FNG_PERIODES), index=0, horizontal=True,
                         key="fng_periode", label_visibility="collapsed")
        dagen = FNG_PERIODES[keuze]
        zicht = df[df["datum"] >= data_tot - pd.Timedelta(days=dagen)] if dagen else df
        x_van, x_tot = zicht["datum"].min(), as_tot
    else:
        laagste = max(data_van, data_tot - pd.DateOffset(years=4)).date()
        standaard = (max(data_van, data_tot - pd.DateOffset(years=1)).date(), data_tot.date())
        van, tot = st.slider(
            "Periode", min_value=laagste, max_value=as_tot.date(), value=standaard,
            step=timedelta(days=1), format="DD-MM-YYYY", key=f"fng_venster_{toon_cyclus}",
            help="Sleep de twee uiteinden om precies te kiezen welke periode je ziet.",
        )
        x_van, x_tot = pd.Timestamp(van), pd.Timestamp(tot)
        if x_tot <= x_van:
            x_tot = x_van + pd.Timedelta(days=1)
        zicht = df[(df["datum"] >= x_van) & (df["datum"] <= x_tot)]
    if zicht.empty:
        zicht = df.tail(1)
    # 'laatste' = meest recente dag in beeld: die waarden staan bij de panelen
    # zolang er (op de telefoon) nog geen dag gekozen is, zoals bij TradingView.
    zicht = zicht.assign(laatste=zicht["datum"] == zicht["datum"].max())

    aflezen = (not pc) and bool(st.session_state.get("fng_aflezen", False))

    # --- selecties -------------------------------------------------------------
    if pc:
        # Muis: punt volgt de muis, weg bij verlaten; zoomen/schuiven met de muis.
        hover = alt.selection_point(name="hover", nearest=True, fields=["datum"], empty=False,
                                    on="pointerover", clear="pointerout")
        zoom = alt.selection_interval(name="zoom", bind="scales", encodings=["x"])
        params = [hover]
        gekozen = {"param": "hover", "empty": False}
    else:
        # Telefoon heeft twee standen (schakelaar 'Aflezen' rechtsboven in de grafiek):
        # - uit (standaard): de grafiek reageert niet op aanraken, je scrolt overal vrij;
        # - aan: tikken en slepen kiest een dag, de pagina scrolt dan niet in de grafiek.
        #   Bij slepen rekenen we de dag uit via de x-positie (een gewone 'nearest'-
        #   selectie blijft bij touch hangen op het beginpunt). Na loslaten blijft het
        #   punt staan; opnieuw aanraken begint opnieuw.
        zoom = None
        if aflezen:
            hover = alt.selection_point(name="hover", nearest=True, fields=["datum"], empty=False,
                                        on="mousemove, touchstart", clear="mouseout")
            sleep = alt.selection_interval(name="sleep", encodings=["x"], clear="touchstart",
                                           on="[touchstart, touchend] > touchmove",
                                           translate=False, zoom=False,
                                           mark=alt.BrushConfig(fillOpacity=0, strokeOpacity=0))
            params = [hover, sleep]
            vinger = "data('sleep_store')[0].values[0][1]"  # datum onder de vinger
            # de dag waarin de vinger valt: datum <= vinger < datum + 1 dag
            gekozen = {"or": [
                {"and": ["length(data('sleep_store')) > 0",
                         f"+{vinger} - +datum.datum >= 0 && +{vinger} - +datum.datum < 86400000"]},
                {"and": ["length(data('sleep_store')) == 0", {"param": "hover", "empty": False}]},
            ]}
        else:
            params, gekozen = [], "false"

    def bij_keuze(aan, uit):
        return alt.value(uit, condition={"test": gekozen, "value": aan})

    x = alt.X("datum:T", title=None,
              axis=alt.Axis(grid=False, labelExpr=X_LABEL_EXPR, labelOverlap=True),
              scale=alt.Scale(domain=[x_van.isoformat(), x_tot.isoformat()]))
    basis = alt.Chart(zicht).encode(x=x)
    tooltip = [alt.Tooltip("datum:T", title="Datum", format="%d-%m-%Y"),
               alt.Tooltip("btc:Q", title="BTC ($)", format=",.0f"),
               alt.Tooltip("fng:Q", title="Fear & Greed", format=".0f"),
               alt.Tooltip("rsi:Q", title=f"RSI ({RSI_PERIODE})", format=".0f")]
    tooltip += [alt.Tooltip(f"{k}:Q", title=f"{BTC_WEEK_MAS[k][0]}w MA ($)", format=",.0f")
                for k in ma_kolommen]

    def hoverlagen(y_veld, *extra):
        # Alle selecties op deze ene (onzichtbare) laag per paneel: Altair verdubbelt
        # ze anders over de lagen heen ("Duplicate signal name").
        punten = basis.mark_point(opacity=0, size=60).encode(
            y=y_veld, **({"tooltip": tooltip} if pc else {})).add_params(*params, *extra)
        lijn = basis.mark_rule(color="#888", strokeDash=[3, 3]).encode(opacity=bij_keuze(0.8, 0))
        stip = basis.mark_circle(size=45).encode(y=y_veld, opacity=bij_keuze(1, 0))
        return [punten, lijn, stip]

    geen_keuze = ("length(data('hover_store')) == 0 && length(data('sleep_store')) == 0"
                  if (not pc and aflezen) else "true")

    def waarde_label(expr: str, kleur: str):
        """Telefoon: linksboven in het paneel de waarde van de gekozen dag, of van
        de laatste dag zolang er niets gekozen is. Dient meteen als paneelnaam."""
        if pc:
            return []
        stijl = dict(align="left", baseline="top", fontSize=12, fontWeight="bold",
                     color=kleur, dx=4, dy=4)
        plek = dict(x=alt.value(0), y=alt.value(0), text="t:N")
        return [
            basis.transform_calculate(t=expr).transform_filter(gekozen)
                 .mark_text(**stijl).encode(**plek),
            basis.transform_calculate(t=expr).transform_filter(f"{geen_keuze} && datum.laatste")
                 .mark_text(**stijl).encode(**plek),
        ]

    zone_lagen = []
    if toon_cyclus:
        zdf = pd.DataFrame(zones)
        zone_lagen = [
            alt.Chart(zdf).mark_rect(color="#d64545", opacity=0.07, clip=True).encode(
                x="aandacht_van:T", x2="aandacht_tot:T"),
            alt.Chart(zdf).mark_rect(color="#d64545", opacity=0.16, clip=True).encode(
                x="kern_van:T", x2="kern_tot:T"),
            alt.Chart(zdf).mark_rule(color="#d64545", strokeDash=[5, 3], opacity=0.8, clip=True).encode(
                x="einde:T"),
        ]

    # Telefoon: y-as-labels strak tegen de grafiek: geen streepjes en aslijn, 3px afstand
    # (was ~17px). Zo blijft er meer breedte over voor de grafiek zelf.
    y_as = {} if pc else dict(ticks=False, domain=False, labelPadding=3, labelFontSize=11)

    # --- prijs -------------------------------------------------------------------
    in_beeld = zicht[["btc", *ma_kolommen]]
    lo, hi = float(in_beeld.min().min()), float(in_beeld.max().max())
    btc_schaal = alt.Scale(type="log", domain=[lo * 0.95, hi * 1.05], nice=False)
    y_btc = alt.Y("btc:Q", title="BTC ($)" if pc else None, scale=btc_schaal,
                  axis=alt.Axis(orient="right", format="$~s", tickCount=5, **y_as))
    label = []
    if toon_cyclus:
        label = [alt.Chart(pd.DataFrame(zones)).mark_text(
            color="#d64545", fontSize=11, fontWeight="bold", baseline="top", align="right",
            dx=-5, clip=True,
        ).encode(x="einde:T", y=alt.value(4 if pc else 22),  # telefoon: onder de waarden
                 text=alt.value("theoretisch einde bullrun"))]
    prijs = alt.layer(
        *zone_lagen, *label,
        *[basis.mark_line(color=BTC_WEEK_MAS[k][1], strokeWidth=1.4, clip=True)
               .encode(y=alt.Y(f"{k}:Q", scale=btc_schaal)) for k in ma_kolommen],
        basis.mark_line(color="#2a78d6", strokeWidth=1.6, clip=True).encode(y=y_btc),
        *hoverlagen(y_btc, *([zoom] if zoom else [])),
        *waarde_label(f"{DATUM_EXPR} + '   BTC $' + format(datum.btc, ',.0f')", "#2a78d6"),
    ).properties(height=240)

    # --- Fear & Greed --------------------------------------------------------------
    # Banden en hulplijnen krijgen hetzelfde y-domein, anders rekt hun data
    # (0–100) de as weer op en schaalt hij niet mee.
    fng_dom = [0, 100] if pc else _y_domein(zicht["fng"])
    fng_banden = alt.Chart(pd.DataFrame(
        [{"van": a, "tot": b, "kleur": k} for a, b, _, k in CMC_ZONES]
    )).mark_rect(opacity=0.10, clip=True).encode(
        y=alt.Y("van:Q", scale=alt.Scale(domain=fng_dom, nice=False)), y2="tot:Q",
        color=alt.Color("kleur:N", scale=None, legend=None))
    y_fng = alt.Y("fng:Q", title="Fear & Greed" if pc else None,
                  scale=alt.Scale(domain=fng_dom, nice=False),
                  axis=alt.Axis(orient="right", tickCount=5, **y_as,
                                **({} if pc else {"labelExpr": "datum.value + '%'"})))
    index = alt.layer(
        fng_banden, *zone_lagen,
        basis.mark_line(color="#eb6834", strokeWidth=1.4, clip=True).encode(y=y_fng),
        *hoverlagen(y_fng),
        *waarde_label("'Fear & Greed ' + format(datum.fng, '.0f') + '%'", "#eb6834"),
    ).properties(height=130)

    # --- RSI: lichte banden boven 70 (overbought) en onder 30 (oversold) ----------
    rsi_dom = [0, 100] if pc else _y_domein(zicht["rsi"])
    rsi_schaal = alt.Scale(domain=rsi_dom, nice=False)
    rsi_banden = alt.Chart(pd.DataFrame([
        {"van": 70, "tot": 100, "kleur": "#2e9e58"}, {"van": 0, "tot": 30, "kleur": "#d64545"},
    ])).mark_rect(opacity=0.08, clip=True).encode(
        y=alt.Y("van:Q", scale=rsi_schaal), y2="tot:Q",
        color=alt.Color("kleur:N", scale=None, legend=None))
    rsi_lijnen = alt.Chart(pd.DataFrame({"y": [30, 50, 70]})).mark_rule(
        strokeDash=[4, 3], color="#888", opacity=0.6, clip=True).encode(
        y=alt.Y("y:Q", scale=rsi_schaal))
    y_rsi = alt.Y("rsi:Q", title=f"RSI ({RSI_PERIODE})" if pc else None,
                  scale=alt.Scale(domain=rsi_dom, nice=False),
                  axis=alt.Axis(orient="right", tickCount=5, **y_as))
    rsi_paneel = alt.layer(
        rsi_banden, *zone_lagen, rsi_lijnen,
        basis.mark_line(color="#6250d6", strokeWidth=1.4, clip=True).encode(y=y_rsi),
        *hoverlagen(y_rsi),
        *waarde_label(f"'RSI ({RSI_PERIODE}) ' + (isValid(datum.rsi) ? format(datum.rsi, '.0f') : '–')",
                      "#6250d6"),
    ).properties(height=110)

    grafiek = alt.vconcat(prijs, index, rsi_paneel, spacing=8).resolve_scale(x="shared")
    if pc:
        st.altair_chart(grafiek, use_container_width=True)
        uitleg = "Scrollen = zoomen, slepen = schuiven, dubbelklik = terug."
    else:
        # Geen zwevende tooltip op de telefoon: die hoort bij het punt waar je vinger
        # begon en klopt tijdens het slepen niet. De waarden staan bij de panelen.
        spec = grafiek.configure_mark(tooltip=False).to_dict()
        st.markdown(GRAFIEK_CSS_TELEFOON + (GRAFIEK_CSS_AFLEZEN if aflezen else GRAFIEK_CSS_SCROLLEN),
                    unsafe_allow_html=True)
        with st.container(key="fng_grafiek"):
            with st.container(key="fng_lock"):
                st.toggle("👆 Aflezen", key="fng_aflezen",
                          help="Aan: tik of sleep in de grafiek om een dag af te lezen. "
                               "Uit: vrij scrollen over de grafiek.")
            st.vega_lite_chart(spec, use_container_width=True)
        uitleg = ("Zet 'Aflezen' aan en tik of sleep in de grafiek om een dag te kiezen; het "
                  "punt blijft staan. Uit = vrij scrollen. De y-assen schalen mee met de periode.")
    st.caption(
        f"{uitleg} Fear & Greed: dagwaarden van CoinMarketCap ({data_van:%d-%m-%Y} t/m "
        f"{data_tot:%d-%m-%Y}). BTC-prijs en RSI: {prijsbron}. "
        f"RSI({RSI_PERIODE}) op dagslotkoersen, Wilder-methode (zoals TradingView)."
    )

# ---------------------------------------------------------------------------
# PAGINA
# ---------------------------------------------------------------------------

SOORTEN = {
    "aandelen": dict(crypto=False, titel="Aandelen & ETF's", fetch=lambda: fetch_stock_fear_greed(),
                     gauge="Aandelen Fear & Greed", fout="CNN niet bereikbaar"),
    "crypto": dict(crypto=True, titel="Crypto", fetch=lambda: fetch_crypto_fear_greed(),
                   gauge="Crypto Fear & Greed", fout="Niet opgehaald"),
}

def parse_tickers(text: str) -> list:
    """Eén ticker per regel (komma's mogen ook), hoofdletters, zonder dubbele."""
    return list(dict.fromkeys(
        t.strip().upper() for t in text.replace(",", "\n").splitlines() if t.strip()
    ))


with st.sidebar:
    st.header("Tickers")
    port_text = st.text_area(
        "Mijn portefeuille", "\n".join(DEFAULT_TICKERS), height=190, key="port_tickers",
        help="Wat je nu bezit. Eén ticker per regel; crypto in yfinance-formaat, bv. BTC-USD.",
    )
    wl_text = st.text_area(
        "Watchlist (extra)", "\n".join(DEFAULT_WATCHLIST), height=160, key="wl_tickers",
        placeholder="bv.\nNVDA\nASML.AS\nLINK-USD",
        help="Tickers die je volgt maar niet bezit.",
    )
    portefeuille = parse_tickers(port_text)
    extra_ingevuld = parse_tickers(wl_text)
    dubbel = [t for t in extra_ingevuld if t in portefeuille]
    extra = [t for t in extra_ingevuld if t not in portefeuille]
    st.caption(
        "Je portefeuille staat automatisch ook in de watchlist, dus die hoef je "
        "daar niet nog eens in te vullen. Aandelen en crypto worden automatisch herkend."
    )
    if dubbel:
        st.caption(f"Al in je portefeuille, dus overgeslagen: {', '.join(dubbel)}")

    with st.expander("Watchlist-score: gewichten"):
        weights = {
            k: st.slider(f"{k} MA", 0.0, 3.0, v, 0.25) for k, v in DEFAULT_WEIGHTS.items()
        }
    st.subheader("Verversen")
    auto = st.toggle("Automatisch verversen", value=True)
    interval = st.select_slider(
        "Koersen en stijgers/dalers: elke … seconden", options=[15, 30, 60], value=15,
        disabled=not auto,
    )
    rang_interval = st.select_slider(
        "Watchlist-ranglijst: elke … seconden", options=[30, 60, 120], value=30,
        disabled=not auto,
        help="De ranglijst is het zwaarste deel van de pagina, dus die ververst rustiger.",
    )
    if st.button("Nu verversen"):
        st.cache_data.clear()
        wis_data_geheugen()

watchlist = portefeuille + extra
with st.spinner("Tickers indelen…"):
    vooraf_ophalen(fetch_meta, watchlist)
    indeling = {t: ("crypto" if is_crypto(t) else "aandelen") for t in watchlist}


def per_soort(tickers: list) -> dict:
    return {k: [t for t in tickers if indeling[t] == k] for k in SOORTEN}


port_per_soort = per_soort(portefeuille)
wl_per_soort = per_soort(watchlist)

# Drie tabbladen. De portefeuille is per soort gesplitst; de watchlist
# krijgt een eigen, kleinere schakelaar voor aandelen/crypto.
WEERGAVEN = {
    "port_aandelen": ("Mijn aandelen & ETF's", lambda: len(port_per_soort["aandelen"])),
    "port_crypto": ("Mijn crypto", lambda: len(port_per_soort["crypto"])),
    "watchlist": ("Watchlist", lambda: len(watchlist)),
}

st.title("Watchlist Ranker")

st.markdown(
    """<style>
    .st-key-weergave button { min-height: 3rem; font-weight: 600; }
    .st-key-weergave button p { font-size: 1.05rem; }
    @keyframes lampje-puls {
        0%   { box-shadow: 0 0 0 0 currentColor; }
        70%  { box-shadow: 0 0 0 5px transparent; }
        100% { box-shadow: 0 0 0 0 transparent; }
    }
    .lampje-puls { animation: lampje-puls 2s ease-out infinite; }
    /* Tijdens het verversen niet grijs maken: de oude data blijft gewoon
       zichtbaar en bruikbaar tot de nieuwe er staat. */
    [data-stale="true"], .stale-element, .element-container.stale-element {
        opacity: 1 !important; transition: none !important; filter: none !important;
    }
    </style>""",
    unsafe_allow_html=True,
)


def schakelaar(key: str, opties: dict, default: str):
    """Segmented control over de volle breedte (met terugval voor oudere Streamlit)."""
    kwargs = dict(
        label=key, options=list(opties), default=default, required=True, key=key,
        label_visibility="collapsed", format_func=lambda k: opties[k],
    )
    try:
        keuze = st.segmented_control(**kwargs, width="stretch")
    except TypeError:
        keuze = st.segmented_control(**kwargs)
    return keuze or default


weergave = schakelaar(
    "weergave", {k: f"{titel}  ({n()})" for k, (titel, n) in WEERGAVEN.items()}, "port_aandelen"
)
if weergave == "watchlist":
    modus = "watchlist"
    soort = schakelaar(
        "wl_soort",
        {k: f"{SOORTEN[k]['titel']}  ({len(wl_per_soort[k])})" for k in SOORTEN},
        "aandelen",
    )
    lijst = wl_per_soort[soort]
else:
    modus = "portefeuille"
    soort = "aandelen" if weergave == "port_aandelen" else "crypto"
    lijst = port_per_soort[soort]


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


RASTER = ('<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));'
          'column-gap:10px">{}</div>')


def render_live_boven(tickers: list, weights: dict, soort: str, modus: str):
    """Snel deel, bovenaan: meldingen en de twee graadmeters."""
    cfg = SOORTEN[soort]
    crypto = cfg["crypto"]
    if not tickers:
        waar = "je portefeuille" if modus == "portefeuille" else "je watchlist"
        st.info(f"Er staan geen {cfg['titel'].lower()} in {waar}. Voeg tickers toe in de zijbalk.")
        return

    ranking, etfs, errors, tijd = laad_ranking(tickers, weights, modus == "watchlist" and not crypto)
    st.caption(f"Koersen van {_klok(tijd)}")

    if RATE_LIMIT["geraakt"]:
        st.error(
            f"Yahoo Finance weigert tijdelijk verzoeken ('Too Many Requests', sinds "
            f"{RATE_LIMIT['tijd']:%H:%M}). Dit gaat meestal binnen 15–60 minuten vanzelf over. "
            "Zet zolang automatisch verversen uit en druk niet steeds op 'Nu verversen'."
        )
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


def render_live_onder(tickers: list, weights: dict, soort: str, modus: str, in_bezit: tuple):
    """Snel deel, onder de grafieken: stijgers/dalers en (portefeuille) de posities.
    De koersen komen uit hetzelfde gedeelde geheugen als render_live_boven."""
    crypto = SOORTEN[soort]["crypto"]
    if not tickers:
        return
    ranking, etfs, _, _ = laad_ranking(tickers, weights, modus == "watchlist" and not crypto)
    if ranking.empty:
        return

    st.divider()
    render_movers(ranking, crypto, etfs)

    if modus == "portefeuille":
        st.divider()
        st.subheader("Mijn posities")
        st.caption("≈ voor een prijs = omgerekend met de actuele euro/dollar-koers.")
        posities = ranking.sort_values("naam", key=lambda s: s.str.lower())
        st.markdown(RASTER.format("".join(holding_card_html(r) for _, r in posities.iterrows())),
                    unsafe_allow_html=True)
        with st.expander("Ruwe tabel"):
            st.dataframe(ranking, width="stretch", hide_index=True)


def render_ranglijst(tickers: list, weights: dict, soort: str, in_bezit: tuple, elke):
    """Het rustige deel: de watchlist-ranglijst met scores (veel kaartjes)."""
    if not tickers:
        return
    crypto = SOORTEN[soort]["crypto"]
    ranking, _, _, tijd = laad_ranking(tickers, weights, not crypto)
    if ranking.empty:
        return
    st.divider()
    st.subheader("Ranglijst")
    uitleg = (f"Koersen van {_klok(tijd)}" + (f", ververst elke {elke} s. " if elke else ". ")
              + "Hoogste score = verste onder de MA's. Balk links (rood) = onder de MA. "
              "1u toont – als er niet gehandeld wordt. "
              "≈ voor een prijs = omgerekend met de actuele euro/dollar-koers.")
    if not crypto:
        uitleg += (" 'Dag' is de verandering t.o.v. de vorige slotkoers. Tijdens voor- of nabeurs (alleen VS) "
                   "staat die koers klein naast de prijs; de score blijft op de reguliere koers.")
    st.caption(uitleg)
    st.markdown(
        RASTER.format("".join(card_html(i + 1, r, r["ticker"] in in_bezit) for i, r in ranking.iterrows())),
        unsafe_allow_html=True,
    )
    with st.expander("Ruwe tabel"):
        st.dataframe(ranking, width="stretch", hide_index=True)


MOMENTUM_MAANDEN = 1


def momentum_rij(ticker: str) -> dict:
    """Rendement over de afgelopen MOMENTUM_MAANDEN maanden op dagslotkoersen.
    Gebruikt dezelfde (gecachete) dagdata als de ranglijst: geen extra verzoeken."""
    closes = fetch_daily_closes(ticker)
    laatste = closes.index[-1]
    ref = _at_or_before(closes, laatste - pd.DateOffset(months=MOMENTUM_MAANDEN))
    return {"Naam": fetch_meta(ticker)["naam"], "Ticker": ticker,
            "pct": _pct(float(closes.iloc[-1]), ref), "Slotkoers van": laatste}


# ---------------------------------------------------------------------------
# FUNDAMENTELE ANALYSE (alleen losse aandelen, geen ETF's)
# ---------------------------------------------------------------------------
# Aannames reverse DCF: pas hier aan
DCF_RENDEMENT = 0.09   # gewenst jaarrendement (disconteringsvoet)
DCF_JAREN = 10         # jaren met de ingeprijsde groei
DCF_EINDGROEI = 0.025  # groei daarna, voor altijd
# Waarde: elke maand een meetpunt over deze periode; 'normaal' = de mediaan
WAARDE_HISTORIE_JAREN = 3
# Jaarcijfers zijn pas een paar maanden na het boekjaar bekend: zolang wacht de code
RAPPORT_VERTRAGING_MND = 3
KWARTAAL_VERTRAGING_MND = 2  # kwartaalcijfers komen sneller
# Werkelijke groei: gemiddelde over (maximaal) zoveel recente kwartalen, elk t.o.v. een jaar eerder
KWARTALEN_GEMIDDELD = 4
# Minimaal aantal maanden met winst om 'normaal' en de balk te tonen
MIN_MAANDEN_VERGELIJKING = 12
# Verhoudingen: (naam, 'je betaalt X× ...')
VERHOUDINGEN = {"ev_ebit": ("EV/EBIT", "de bedrijfswinst"), "kw": ("K/W", "de winst"),
                "p_fcf": ("Koers/vrije kasstroom", "de vrije kasstroom"),
                "p_b": ("Koers/boekwaarde", "de boekwaarde")}
# Weging van de eindscore (wordt herschaald als een onderdeel ontbreekt)
SCORE_GEWICHTEN = {"waarde": 0.35, "kwaliteit": 0.35, "groei": 0.20, "veiligheid": 0.10}
# Mohanram G-score: het origineel vergelijkt met de mediaan van de bedrijfstak. Die
# vergelijkingsgroep hebben we niet, dus vaste drempels (pas aan naar smaak).
G_DREMPELS = {"roa": 0.06, "cfo_roa": 0.06, "roa_spreiding": 0.03,
              "groei_spreiding": 0.10, "rd": 0.02, "capex": 0.03}


def _maak_naief(index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index)
    return idx.tz_localize(None) if idx.tz is not None else idx


@st.cache_data(ttl=24 * 3600, max_entries=200, show_spinner=False)
def fetch_jaarcijfers(ticker: str) -> dict:
    """Jaarcijfers (resultaat, balans, kasstroom), wat basisinfo en slotkoersen
    zonder dividendcorrectie (zodat oude koersen echt de koers van toen zijn).
    24 uur bewaard: jaarcijfers veranderen maar een paar keer per jaar."""
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
        stukken = {"resultaat": t.income_stmt, "balans": t.balance_sheet, "kasstroom": t.cashflow}
        # vaste startdatum i.p.v. period="6y": oudere yfinance-versies kennen alleen 5y/10y
        start = (pd.Timestamp.now() - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
        koers = t.history(start=start, interval="1d", auto_adjust=False)
    except YFRateLimitError:
        _markeer_rate_limit()
        raise  # niet cachen
    for k, df in stukken.items():
        if df is None or df.empty:
            raise ValueError("geen jaarcijfers bij Yahoo")
        df = df.copy()
        df.columns = _maak_naief(df.columns)
        stukken[k] = df[sorted(df.columns)]  # oud -> nieuw
    if koers.empty:
        raise ValueError("geen koersdata")
    # Kwartaalcijfers voor de winst over de laatste 12 maanden (TTM). Niet verplicht.
    kwartaal = {}
    for k, naam in (("resultaat", "quarterly_income_stmt"), ("balans", "quarterly_balance_sheet"),
                    ("kasstroom", "quarterly_cashflow")):
        try:
            df = getattr(t, naam)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                df.columns = _maak_naief(df.columns)
                kwartaal[k] = df[sorted(df.columns)]
        except YFRateLimitError:
            _markeer_rate_limit()
            raise
        except Exception:  # noqa: BLE001
            pass
    analisten = {}
    for naam in ("revenue_estimate", "earnings_estimate", "earnings_history"):
        try:
            df = getattr(t, naam)
            if isinstance(df, pd.DataFrame) and not df.empty:
                analisten[naam] = df
        except YFRateLimitError:
            _markeer_rate_limit()
            raise
        except Exception:  # noqa: BLE001
            pass
    slot = koers["Close"].dropna()
    slot.index = _maak_naief(slot.index)
    return {
        **stukken, "slot": slot, "analisten": analisten, "kwartaal": kwartaal,
        "info": {k: info.get(k) for k in ("sector", "industry", "financialCurrency",
                                           "currency", "marketCap", "sharesOutstanding", "forwardPE",
                                           "trailingPE", "freeCashflow")},
    }


def _regel(df: pd.DataFrame, *namen) -> pd.Series:
    """De regel onder de eerste naam die bestaat; lege plekken worden aangevuld uit de
    volgende namen (Yahoo gebruikt niet altijd dezelfde naam, ook niet per kwartaal)."""
    uit = None
    for n in namen:
        if df is not None and n in df.index:
            r = pd.to_numeric(df.loc[n], errors="coerce")
            uit = r if uit is None else uit.combine_first(r)
    return uit if uit is not None else pd.Series(dtype=float)


def _vrije_kasstroom(kas: pd.DataFrame) -> pd.Series:
    """Free Cash Flow, of anders operationele kasstroom + investeringen (die zijn negatief)."""
    fcf = _regel(kas, "Free Cash Flow")
    berekend = _regel(kas, "Operating Cash Flow") + _regel(kas, "Capital Expenditure")
    return fcf.combine_first(berekend) if not berekend.empty else fcf


def _w(reeks: pd.Series, datum):
    """Waarde op een jaareinde, of None."""
    v = reeks.get(datum) if not reeks.empty else None
    return None if v is None or pd.isna(v) else float(v)


def _deel(a, b):
    return a / b if a is not None and b not in (None, 0) else None


def _fx_reeks(van: str, naar: str):
    """Dagelijkse wisselkoers (1 van = x naar) of None als die er niet is."""
    if van == naar:
        return pd.Series(dtype=float)  # leeg = koers 1
    for paar, omkeren in ((f"{van}{naar}=X", False), (f"{naar}{van}=X", True)):
        try:
            s = fetch_daily_closes(paar)
            return 1 / s if omkeren else s
        except Exception:  # noqa: BLE001
            continue
    return None


def _fx_op(reeks: pd.Series, datum) -> float:
    if reeks.empty:
        return 1.0
    v = _at_or_before(reeks, datum)
    return v if v is not None else float(reeks.iloc[0])  # ouder dan de reeks: eerste koers


def _cagr(reeks: pd.Series):
    """Gemiddelde jaarlijkse groei van eerste naar laatste jaar (alleen bij positieve waarden)."""
    r = reeks.dropna()
    if len(r) < 2 or r.iloc[0] <= 0 or r.iloc[-1] <= 0:
        return None
    jaren = (r.index[-1] - r.index[0]).days / 365.25
    return (r.iloc[-1] / r.iloc[0]) ** (1 / jaren) - 1 if jaren > 0.5 else None


def dcf_waarde(fcf: float, groei: float) -> float:
    r, gt = DCF_RENDEMENT, DCF_EINDGROEI
    waarde, kas = 0.0, fcf
    for jaar in range(1, DCF_JAREN + 1):
        kas *= 1 + groei
        waarde += kas / (1 + r) ** jaar
    return waarde + kas * (1 + gt) / (r - gt) / (1 + r) ** DCF_JAREN


def ingeprijsde_groei(fcf: float, marktwaarde: float):
    """Reverse DCF: welke jaarlijkse groei maakt de DCF-waarde gelijk aan de
    marktwaarde? Geeft (groei, teken): teken is '>' of '<' als het buiten -50..100% valt."""
    if fcf is None or marktwaarde is None or fcf <= 0 or marktwaarde <= 0:
        return None, ""
    lo, hi = -0.5, 1.0
    if dcf_waarde(fcf, hi) < marktwaarde:
        return hi, ">"
    if dcf_waarde(fcf, lo) > marktwaarde:
        return lo, "<"
    for _ in range(60):  # halveren: de waarde stijgt met de groei
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if dcf_waarde(fcf, mid) < marktwaarde else (lo, mid)
    return (lo + hi) / 2, ""


PIOTROSKI_TESTS = ["Winst", "Kasstroom", "Rendement stijgt", "Kas > winst", "Schuld lager",
                   "Liquiditeit beter", "Geen uitgifte", "Marge stijgt", "Efficiënter"]


def piotroski(res, bal, kas, nu, vorig) -> list:
    """Piotroski F-score: per test True/False, of None als de data ontbreekt."""
    ni, cfo = _regel(res, "Net Income", "Net Income Common Stockholders"), _regel(kas, "Operating Cash Flow")
    activa = _regel(bal, "Total Assets")
    lt_schuld = _regel(bal, "Long Term Debt", "Long Term Debt And Capital Lease Obligation")
    vlot_a, vlot_p = _regel(bal, "Current Assets"), _regel(bal, "Current Liabilities")
    aandelen = _regel(bal, "Ordinary Shares Number", "Share Issued")
    bruto, omzet = _regel(res, "Gross Profit"), _regel(res, "Total Revenue", "Operating Revenue")

    def twee(reeks_a, reeks_b=None):
        return [(_deel(_w(reeks_a, d), _w(reeks_b, d)) if reeks_b is not None else _w(reeks_a, d))
                for d in (nu, vorig)]

    def groter(paar, of_gelijk=False):
        if None in paar:
            return None
        return paar[0] >= paar[1] if of_gelijk else paar[0] > paar[1]

    roa = twee(ni, activa)
    st_ = twee(aandelen)
    verhouding = _deel(st_[0], st_[1])
    hef = twee(lt_schuld, activa)
    return [
        None if roa[0] is None else roa[0] > 0,
        None if _w(cfo, nu) is None else _w(cfo, nu) > 0,
        groter(roa),
        None if None in (_w(cfo, nu), _w(ni, nu)) else _w(cfo, nu) > _w(ni, nu),
        None if None in hef else hef[0] <= hef[1],
        groter(twee(vlot_a, vlot_p)),
        # > 1,5x is vrijwel altijd een aandelensplitsing, geen uitgifte: niet meetellen
        None if verhouding is None or verhouding > 1.5 else verhouding <= 1.0,
        groter(twee(bruto, omzet)),
        groter(twee(omzet, activa)),
    ]


def _is_financieel(info: dict) -> bool:
    branche = str(info.get("industry") or "")
    return info.get("sector") == "Financial Services" and any(
        w in branche for w in ("Bank", "Insurance", "Capital Markets"))


def _analisten(a: dict) -> dict:
    """Groeiverwachtingen van analisten (geen koersdoelen) en hoe vaak de winst
    hun verwachting versloeg."""
    uit = {}
    for bron, sleutel in (("revenue_estimate", "omzet"), ("earnings_estimate", "winst")):
        df = a.get(bron)
        if df is None or "growth" not in df.columns:
            continue
        groei = pd.to_numeric(df["growth"], errors="coerce")
        for periode, naam in (("+1q", "kw"), ("0y", "dit_jr"), ("+1y", "jr")):
            if periode in groei.index and pd.notna(groei[periode]):
                uit[f"{sleutel}_{naam}"] = float(groei[periode])
    hist = a.get("earnings_history")
    if hist is not None and {"epsActual", "epsEstimate"} <= set(hist.columns):
        h = hist[["epsActual", "epsEstimate"]].apply(pd.to_numeric, errors="coerce").dropna().tail(4)
        if not h.empty:
            uit["verslagen"], uit["kwartalen"] = int((h["epsActual"] > h["epsEstimate"]).sum()), len(h)
    return uit


def _trend(reeks: list):
    """Richting van een marge t.o.v. vorig jaar: 1 omhoog, -1 omlaag, 0 gelijk (±1 procentpunt)."""
    if len(reeks) < 2 or None in reeks[-2:]:
        return None
    verschil = reeks[-1] - reeks[-2]
    return 1 if verschil > 0.01 else -1 if verschil < -0.01 else 0


def _std(waarden: list):
    w = [v for v in waarden if v is not None]
    return float(pd.Series(w).std()) if len(w) >= 3 else None


def fundamentele_rij(ticker: str) -> dict:
    """Alle cijfers voor één aandeel. Bedragen worden omgerekend naar de valuta van
    de jaarcijfers (bv. Tencent: koers in HKD, cijfers in CNY)."""
    rij = {"ticker": ticker}
    d = fetch_jaarcijfers(ticker)
    res, bal, kas, slot, info = d["resultaat"], d["balans"], d["kasstroom"], d["slot"], d["info"]
    jaren = list(res.columns)
    nu_jaar = jaren[-1]
    rij["jaren"] = len(jaren)

    # --- valuta en marktwaarde ---------------------------------------------------
    koers_val = info.get("currency") or fetch_meta(ticker)["valuta"]
    deler = 100 if koers_val == "GBp" else 1  # Londen noteert in pence
    koers_val = "GBP" if koers_val == "GBp" else koers_val
    cijfer_val = info.get("financialCurrency") or koers_val
    fx = _fx_reeks(koers_val, cijfer_val)
    if fx is None:
        raise ValueError(f"geen wisselkoers {koers_val}->{cijfer_val}")
    aandelen = _regel(bal, "Ordinary Shares Number", "Share Issued")
    laatste_aandelen = (_w(aandelen, aandelen.last_valid_index())
                        if not aandelen.dropna().empty else None)
    # Aantal volgens de koersnotering (bij ADR's al omgerekend), anders uit de balans
    aandelen_nu = info.get("sharesOutstanding") or laatste_aandelen
    if not aandelen_nu:
        raise ValueError("aantal aandelen onbekend")
    koers_nu = float(slot.iloc[-1]) / deler
    mcap_nu = koers_nu * aandelen_nu * _fx_op(fx, slot.index[-1])  # in de valuta van de cijfers

    schuld = _regel(bal, "Total Debt")
    kasgeld = _regel(bal, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
    ebit = _regel(res, "EBIT", "Operating Income")
    fcf = _vrije_kasstroom(kas)
    cfo = _regel(kas, "Operating Cash Flow")
    eigen_vermogen = _regel(bal, "Stockholders Equity", "Common Stock Equity")
    netto_winst = _regel(res, "Net Income", "Net Income Common Stockholders")
    omzet = _regel(res, "Total Revenue", "Operating Revenue")
    activa = _regel(bal, "Total Assets")

    financieel = _is_financieel(info) or fcf.dropna().empty
    rij["financieel"] = financieel

    # --- waarde ---------------------------------------------------------------------
    # Periodes met cijfers: elk boekjaar, plus 'laatste 12 maanden' (TTM) uit de
    # kwartaalcijfers. Per periode: vanaf wanneer bekend, en de bedragen.
    periodes = [{"datum": j, "vanaf": j + pd.DateOffset(months=RAPPORT_VERTRAGING_MND), "bron": "jaar",
                 "ebit": _w(ebit, j), "ni": _w(netto_winst, j), "fcf": _w(fcf, j),
                 "eq": _w(eigen_vermogen, j), "schuld": _w(schuld, j), "kas": _w(kasgeld, j),
                 "aandelen": _w(aandelen, j)} for j in jaren]
    kw = d.get("kwartaal", {})
    if {"resultaat", "kasstroom"} <= set(kw):
        k_res, k_kas, k_bal = kw["resultaat"], kw["kasstroom"], kw.get("balans", pd.DataFrame())
        k_ebit, k_ni = _regel(k_res, "EBIT", "Operating Income"), _regel(k_res, "Net Income",
                                                                          "Net Income Common Stockholders")
        k_fcf = _vrije_kasstroom(k_kas)
        k_eq = _regel(k_bal, "Stockholders Equity", "Common Stock Equity") if not k_bal.empty else pd.Series(dtype=float)
        k_schuld = _regel(k_bal, "Total Debt") if not k_bal.empty else pd.Series(dtype=float)
        k_kasg = (_regel(k_bal, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
                  if not k_bal.empty else pd.Series(dtype=float))
        k_aand = _regel(k_bal, "Ordinary Shares Number", "Share Issued") if not k_bal.empty else pd.Series(dtype=float)
        kols = list(k_res.columns)

        def som4(reeks, venster):
            w = [_w(reeks, c) for c in venster]
            return None if None in w else sum(w)

        for i in range(3, len(kols)):
            venster = kols[i - 3:i + 1]
            if not 240 <= (venster[-1] - venster[0]).days <= 300:  # 4 opeenvolgende kwartalen
                continue
            q = venster[-1]
            periodes.append({"datum": q, "vanaf": q + pd.DateOffset(months=KWARTAAL_VERTRAGING_MND),
                             "bron": "ttm", "ebit": som4(k_ebit, venster), "ni": som4(k_ni, venster),
                             "fcf": som4(k_fcf, venster), "eq": _w(k_eq, q), "schuld": _w(k_schuld, q),
                             "kas": _w(k_kasg, q), "aandelen": _w(k_aand, q)})
    # bij dezelfde datum wint TTM (vollediger), daarna gewoon de nieuwste
    periodes.sort(key=lambda p: (p["datum"], p["bron"] == "ttm"))

    def samengesteld(kandidaten):
        """Per cijfer de nieuwste periode die dat cijfer heeft (laatste 12 maanden,
        anders het laatste boekjaar). Zo maakt één leeg kwartaal niet alles leeg."""
        if not kandidaten:
            return None
        uit = {"bronnen": {}}
        for veld in ("ebit", "ni", "fcf", "eq", "schuld", "kas", "aandelen"):
            for p in reversed(kandidaten):
                if p[veld] is not None:
                    uit[veld] = p[veld]
                    uit["bronnen"][veld] = p
                    break
            else:
                uit[veld] = None
        return uit

    nu_p = samengesteld(periodes)
    bron = nu_p["bronnen"].get("ni") or nu_p["bronnen"].get("ebit") or periodes[-1]
    rij["bron_nu"] = (f"laatste 12 maanden t/m {_maand(bron['datum'])}" if bron["bron"] == "ttm"
                      else f"boekjaar t/m {_maand(bron['datum'])}")
    ref_aandelen = nu_p["aandelen"] or laatste_aandelen

    def marktwaarde_op(datum, periode):
        """Marktwaarde op een datum, afgeleid van de huidige (zo kloppen ook ADR's):
        koersverhouding × wisselkoersverhouding × verhouding aantal aandelen."""
        k = _at_or_before(slot, datum)
        if k is None:
            return None
        m = mcap_nu * (k / deler) / koers_nu * _fx_op(fx, datum) / _fx_op(fx, slot.index[-1])
        verh = _deel(periode["aandelen"], ref_aandelen)
        if verh is not None and 0.67 <= verh <= 1.5:  # anders waarschijnlijk een splitsing
            m *= verh
        return m

    def verhoudingen(mcap, p):
        """Per verhouding de waarde, of 'verlies' (noemer ≤ 0) of None (geen data)."""
        if mcap is None or p is None:
            return {}

        def maal(teller, noemer):
            if noemer is None or teller is None:
                return None
            return teller / noemer if noemer > 0 and teller > 0 else "verlies"

        if financieel:
            return {"p_b": maal(mcap, p["eq"])}
        ev = mcap + (p["schuld"] or 0.0) - (p["kas"] or 0.0)
        return {"ev_ebit": maal(ev, p["ebit"]), "kw": maal(mcap, p["ni"]), "p_fcf": maal(mcap, p["fcf"])}

    nu = verhoudingen(mcap_nu, nu_p)
    rij["reserve"] = []
    if not financieel:
        tpe, yfcf = info.get("trailingPE"), info.get("freeCashflow")
        if nu.get("kw") is None and isinstance(tpe, (int, float)) and 0 < tpe < 2000:
            nu["kw"] = float(tpe)
            rij["reserve"].append("winst")
        if nu.get("p_fcf") is None and isinstance(yfcf, (int, float)):
            nu["p_fcf"] = mcap_nu / yfcf if yfcf > 0 else "verlies"
            rij["reserve"].append("kasstroom")
    laatste_dag = slot.index[-1]
    n_maanden = WAARDE_HISTORIE_JAREN * 12
    punten = []
    for i in range(1, n_maanden + 1):
        maand = laatste_dag - pd.DateOffset(months=i)
        # de cijfers die op dat moment het laatst bekend waren
        p = samengesteld([p for p in periodes if p["vanaf"] <= maand])
        punten.append(verhoudingen(marktwaarde_op(maand, p) if p else None, p))
    rij["maanden"] = n_maanden
    for k, v_nu in nu.items():
        waarden = [pt.get(k) for pt in punten]
        historie = [v for v in waarden if isinstance(v, float)]
        info_k = {"nu": v_nu if isinstance(v_nu, float) else None,
                  "nu_verlies": v_nu == "verlies", "totaal": n_maanden, "n": len(historie),
                  "verlies": sum(v == "verlies" for v in waarden),
                  "ontbreekt": sum(v is None for v in waarden)}
        if info_k["nu"] is not None and len(historie) >= MIN_MAANDEN_VERGELIJKING:
            info_k.update(normaal=float(pd.Series(historie).median()),
                          min=min(historie + [v_nu]), max=max(historie + [v_nu]),
                          goedkoper=sum(h > v_nu for h in historie) / len(historie))
        rij[k] = info_k
    if rij.get("ev_ebit", {}).get("nu"):
        rij["ey_nu"] = 1 / rij["ev_ebit"]["nu"]  # voor de Magic Formula
    waarschuwingen = []
    for reeks in (ebit, netto_winst):
        nu_w, vorig_w = (_w(reeks, nu_jaar), _w(reeks, jaren[-2])) if len(jaren) >= 2 else (None, None)
        if nu_w is not None and vorig_w is not None and nu_w > 0:
            if vorig_w <= 0:
                waarschuwingen.append("net_winstgevend")
                break
            if nu_w > 2 * vorig_w:
                waarschuwingen.append("verdubbeld")
                break
    rij["waarschuwingen"] = waarschuwingen

    # forward K/W (koers ÷ verwachte winst volgend jaar) en PEG
    fpe = info.get("forwardPE")
    rij["forward_kw"] = float(fpe) if isinstance(fpe, (int, float)) and 0 < fpe < 1000 else None

    # --- groei: ingeprijsd, verwacht, werkelijk ------------------------------------------
    # Werkelijke omzetgroei: per kwartaal t.o.v. hetzelfde kwartaal een jaar eerder,
    # gemiddeld over de recente kwartalen. Zo is het direct vergelijkbaar met de
    # verwachtingen (die ook 'jaar op jaar' zijn). Zonder kwartaalcijfers: per jaar.
    k_omzet = (_regel(kw["resultaat"], "Total Revenue", "Operating Revenue")
               if "resultaat" in kw else pd.Series(dtype=float)).dropna()
    groeien = []
    for q, waarde in k_omzet.items():
        jaar_eerder = [(abs((q - p_).days - 365), v) for p_, v in k_omzet.items() if 330 <= (q - p_).days <= 400]
        if jaar_eerder and min(jaar_eerder)[1] > 0:
            groeien.append(waarde / min(jaar_eerder)[1] - 1)
    if groeien:
        groeien = groeien[-KWARTALEN_GEMIDDELD:]
        rij["omzetgroei"] = sum(groeien) / len(groeien)
        rij["omzetgroei_bron"] = (f"gem. laatste {len(groeien)} kwartalen" if len(groeien) > 1
                                  else "laatste kwartaal")
    else:
        rij["omzetgroei"] = _cagr(omzet)
        r = omzet.dropna()
        rij["omzetgroei_bron"] = (f"per jaar {r.index[0].year}–{r.index[-1].year}"
                                  if len(r) >= 2 else "")
    if not financieel:
        # Reverse DCF vanaf de vrije kasstroom. Is die negatief of nul (bv. door zware
        # investeringen, zoals bij Amazon), dan als reserve vanaf de winst: die smeert
        # investeringen uit over de jaren in plaats van ze in één keer af te trekken.
        basis_fcf, basis_ni = nu_p["fcf"], nu_p["ni"]
        if basis_fcf is not None and basis_fcf > 0:
            rij["ingeprijsd"], rij["ingeprijsd_teken"] = ingeprijsde_groei(basis_fcf, mcap_nu)
            rij["ingeprijsd_basis"] = "kasstroom"
        elif basis_ni is not None and basis_ni > 0:
            rij["ingeprijsd"], rij["ingeprijsd_teken"] = ingeprijsde_groei(basis_ni, mcap_nu)
            rij["ingeprijsd_basis"] = "winst"
            rij["ingeprijsd_reden"] = ("vrije kasstroom negatief door investeringen"
                                       if basis_fcf is not None else "geen kasstroomcijfers")
        else:
            rij["ingeprijsd"], rij["ingeprijsd_teken"] = None, ""
    rij.update(_analisten(d.get("analisten", {})))
    if rij["forward_kw"] and rij.get("winst_jr") and rij["winst_jr"] > 0:
        rij["peg"] = rij["forward_kw"] / (rij["winst_jr"] * 100)

    # --- kwaliteit: marges, ROIC/ROE, Piotroski, Mohanram -----------------------------------
    for naam, teller in (("bruto", _regel(res, "Gross Profit")), ("operationeel", ebit),
                         ("netto", netto_winst)):
        reeks = [_deel(_w(teller, j), _w(omzet, j)) for j in jaren]
        rij[f"marge_{naam}"], rij[f"marge_{naam}_trend"] = reeks[-1], _trend(reeks)

    if financieel:
        rendementen = [_deel(_w(netto_winst, j), _w(eigen_vermogen, j)) for j in jaren]
    else:
        winst_vb, belasting = _regel(res, "Pretax Income"), _regel(res, "Tax Provision")
        rendementen = []
        for j in jaren:
            e = _w(ebit, j)
            # Geïnvesteerd kapitaal = schuld + eigen vermogen. Kas trekken we níet af: bij bedrijven
            # met veel kas blijft er anders bijna niets over en krijg je onzin als 2000%.
            kapitaal = (_w(schuld, j) or 0.0) + (_w(eigen_vermogen, j) or 0.0)
            ta = _w(activa, j)
            if e is None or kapitaal <= 0 or (ta and kapitaal < 0.1 * ta):  # te klein om iets te zeggen
                rendementen.append(None)
                continue
            tarief = _deel(_w(belasting, j), _w(winst_vb, j))
            tarief = 0.21 if tarief is None else min(max(tarief, 0.0), 0.35)
            rendementen.append(e * (1 - tarief) / kapitaal)
    reeks = [r for r in rendementen if r is not None]
    if reeks:
        rij["rendement_nu"], rij["rendement_min"], rij["rendement_max"] = reeks[-1], min(reeks), max(reeks)

    if not financieel and len(jaren) >= 2:
        rij["piotroski"] = piotroski(res, bal, kas, nu_jaar, jaren[-2])

        roa = [_deel(_w(netto_winst, j), _w(activa, j)) for j in jaren]
        groeien = [_deel(_w(omzet, b), _w(omzet, a)) for a, b in zip(jaren, jaren[1:])]
        groeien = [g - 1 if g is not None else None for g in groeien]
        rd = _deel(_w(_regel(res, "Research And Development"), nu_jaar), _w(activa, nu_jaar))
        capex = _w(_regel(kas, "Capital Expenditure"), nu_jaar)
        capex = _deel(abs(capex), _w(activa, nu_jaar)) if capex is not None else None
        cfo_roa = _deel(_w(cfo, nu_jaar), _w(activa, nu_jaar))
        roa_sd, groei_sd = _std(roa), _std(groeien)
        gd = G_DREMPELS
        rij["mohanram"] = [
            None if roa[-1] is None else roa[-1] >= gd["roa"],
            None if cfo_roa is None else cfo_roa >= gd["cfo_roa"],
            None if None in (_w(cfo, nu_jaar), _w(netto_winst, nu_jaar))
            else _w(cfo, nu_jaar) > _w(netto_winst, nu_jaar),
            None if roa_sd is None else roa_sd <= gd["roa_spreiding"],
            None if groei_sd is None else groei_sd <= gd["groei_spreiding"],
            None if rd is None else rd >= gd["rd"],
            None if capex is None else capex >= gd["capex"],
        ]

        # --- veiligheid: Altman Z ---------------------------------------------------------
        ta = _w(activa, nu_jaar)
        delen = [
            _deel((_w(_regel(bal, "Current Assets"), nu_jaar) or 0.0)
                  - (_w(_regel(bal, "Current Liabilities"), nu_jaar) or 0.0), ta),
            _deel(_w(_regel(bal, "Retained Earnings"), nu_jaar), ta),
            _deel(_w(ebit, nu_jaar), ta),
            _deel(mcap_nu, _w(_regel(bal, "Total Liabilities Net Minority Interest",
                                     "Total Liabilities"), nu_jaar)),
            _deel(_w(omzet, nu_jaar), ta),
        ]
        if None not in delen:
            rij["altman"] = sum(f * x for f, x in zip((1.2, 1.4, 3.3, 0.6, 1.0), delen))
    return rij


def _fund_veilig(ticker: str):
    try:
        return fundamentele_rij(ticker)
    except Exception as e:  # noqa: BLE001
        return {"ticker": ticker, "fout": str(e)}


# --- scores ------------------------------------------------------------------------

def _lin(x, laag, hoog):
    """x lineair naar 0–100 (laag -> 0, hoog -> 100)."""
    return None if x is None else max(0.0, min(100.0, (x - laag) / (hoog - laag) * 100))


def _telling(tests):
    gedaan = [t for t in (tests or []) if t is not None]
    return sum(gedaan), len(gedaan)


def verwachte_groei(f: dict):
    """Omzetgroei volgend jaar volgens analisten, anders de werkelijke groei."""
    for k, bron in (("omzet_jr", "analisten"), ("omzet_dit_jr", "analisten"), ("omzetgroei", "historie")):
        if f.get(k) is not None:
            return f[k], bron
    return None, None


def scores(f: dict) -> dict:
    """Deelscores 0–100, oordelen en de eindscore volgens SCORE_GEWICHTEN."""
    s = {}
    # waarde
    # = in hoeveel procent van de maanden het aandeel duurder was dan nu
    delen = [f[k]["goedkoper"] * 100 for k in VERHOUDINGEN
             if isinstance(f.get(k), dict) and "goedkoper" in f[k]]
    s["waarde"] = sum(delen) / len(delen) if delen else None
    s["waarde_label"] = (None if s["waarde"] is None else "goedkoop" if s["waarde"] >= 65
                         else "duur" if s["waarde"] <= 35 else "normaal")
    # groei
    verwacht, _ = verwachte_groei(f)
    ing = f.get("ingeprijsd")
    if ing is not None and verwacht is not None:
        verschil = verwacht - ing
        s["groei"] = _lin(verschil, -0.10, 0.10)
        s["groei_label"] = "beter" if verschil >= 0.02 else "minder" if verschil <= -0.02 else "gelijk"
    elif verwacht is not None:
        s["groei"] = _lin(verwacht, -0.05, 0.20)
        s["groei_label"] = "sterk" if s["groei"] >= 60 else "zwak" if s["groei"] < 40 else "gemiddeld"
    else:
        s["groei"], s["groei_label"] = None, None
    # kwaliteit
    delen = []
    p, n = _telling(f.get("piotroski"))
    if n >= 5:
        delen.append(p / n * 100)
    g, n = _telling(f.get("mohanram"))
    if n >= 4:
        delen.append(g / n * 100)
    if f.get("rendement_nu") is not None:
        delen.append(_lin(f["rendement_nu"], 0.0, 0.20))
    trends = [f.get(f"marge_{m}_trend") for m in ("bruto", "operationeel", "netto")]
    trends = [x for x in trends if x is not None]
    if trends:
        delen.append(50 + 50 * sum(trends) / len(trends))
    s["kwaliteit"] = sum(delen) / len(delen) if delen else None
    s["kwaliteit_label"] = (None if s["kwaliteit"] is None else "sterk" if s["kwaliteit"] >= 65
                            else "zwak" if s["kwaliteit"] < 40 else "gemiddeld")
    # veiligheid (hoog risico is een label, geen straf: het telt gewoon mee met 10%)
    z = f.get("altman")
    s["veiligheid"] = _lin(z, 1.0, 4.0)
    s["veiligheid_label"] = (None if z is None else "veilig" if z > 2.99
                             else "grijs gebied" if z >= 1.81 else "risico hoog")
    # eindscore, herschaald over de onderdelen die er zijn
    beschikbaar = {k: w for k, w in SCORE_GEWICHTEN.items() if s.get(k) is not None}
    s["eind"] = (sum(s[k] * w for k, w in beschikbaar.items()) / sum(beschikbaar.values())
                 if beschikbaar else None)
    return s


def magic_formula(rijen: list) -> dict:
    """Greenblatt: rang op earnings yield + rang op ROIC, binnen de eigen lijst."""
    kandidaten = [r for r in rijen if not r.get("financieel")
                  and r.get("ey_nu") is not None and r.get("rendement_nu") is not None]
    if len(kandidaten) < 2:
        return {}
    df = pd.DataFrame([{"t": r["ticker"], "ey": r["ey_nu"], "roic": r["rendement_nu"]} for r in kandidaten])
    df["som"] = df["ey"].rank(ascending=False) + df["roic"].rank(ascending=False)
    df["rang"] = df["som"].rank(method="min").astype(int)
    return {r.t: (r.rang, len(df)) for r in df.itertuples()}


# --- weergave: kaartjes met sterke punten, aandachtspunten en een cijfertabel ----------

KLEUR = {  # (tekst, achtergrond): leesbaar in licht en donker thema
    "groen": ("#2e9e58", "rgba(46,158,88,0.15)"),
    "geel": ("#b08a0c", "rgba(233,196,64,0.20)"),
    "rood": ("#d64545", "rgba(214,69,69,0.14)"),
    "grijs": ("inherit", "rgba(128,128,128,0.15)"),
}
LABEL_KLEUR = {"goedkoop": "groen", "normaal": "geel", "duur": "rood",
               "beter": "groen", "gelijk": "geel", "minder": "rood",
               "sterk": "groen", "gemiddeld": "geel", "zwak": "rood",
               "veilig": "groen", "grijs gebied": "geel", "risico hoog": "rood"}
GROEI_TEKST = {"beter": "beter dan ingeprijsd", "gelijk": "zoals ingeprijsd", "minder": "minder dan ingeprijsd"}
# Namen in gewone taal (de vakterm staat in 'Hoe lees ik dit?')
WAARVAN = {"kw": "winst", "ev_ebit": "bedrijfswinst", "p_fcf": "kasstroom", "p_b": "boekwaarde"}
NAAM = {"kw": "Je betaalt × de winst", "ev_ebit": "Je betaalt × bedrijfswinst",
        "p_fcf": "Je betaalt × kasstroom", "p_b": "Je betaalt × boekwaarde"}
PIOTROSKI_NAMEN = ["Winst positief", "Kasstroom positief", "Rendement gestegen", "Kasstroom hoger dan winst",
                   "Schuld gedaald", "Liquiditeit verbeterd", "Geen nieuwe aandelen", "Brutomarge gestegen",
                   "Efficiënter (omzet per bezitting)"]
MOHANRAM_NAMEN = [f"Rendement ≥ {G_DREMPELS['roa']:.0%}", f"Kasstroom ≥ {G_DREMPELS['cfo_roa']:.0%} van bezittingen",
                  "Kasstroom hoger dan winst", "Stabiele winst", "Stabiele omzetgroei",
                  f"Investeert in R&D (≥ {G_DREMPELS['rd']:.0%})",
                  f"Investeert in bezittingen (≥ {G_DREMPELS['capex']:.0%})"]
MAX_PUNTEN = 4

FUND_CSS = """<style>
.fk details > summary { list-style: none; cursor: pointer; }
.fk details > summary::-webkit-details-marker { display: none; }
.fk details.fk-test > summary { list-style: none; }
.fk details[open] > summary .fk-pijl { transform: rotate(180deg); }
.fk details[open] > summary .fk-toon { display: none; }
.fk details:not([open]) > summary .fk-verberg { display: none; }
.fk-pt { display: flex; gap: 7px; font-size: 0.85rem; line-height: 1.45; padding: 2px 0; }
.fk-sub { opacity: 0.6; font-size: 0.8rem; }
.fk-kop { font-size: 0.75rem; font-weight: 600; padding: 7px 0 2px; }
.fk-sec { font-size: 0.75rem; opacity: 0.65; font-weight: 600; padding: 7px 0 2px;
          border-top: 1px solid rgba(128,128,128,0.22); margin-top: 4px; }
.fk table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 0.82rem;
            font-variant-numeric: tabular-nums; margin: 0; }
.fk td { padding: 2px 0; border: none; vertical-align: top; background: none; }
.fk td.l { opacity: 0.7; width: 44%; }
.fk td.n { text-align: right; font-weight: 600; width: 22%; }
.fk td.v { text-align: right; opacity: 0.55; font-size: 0.75rem; width: 34%; }
.fk tr { background: none !important; border: none; }
.fk-voet { font-size: 0.7rem; opacity: 0.5; padding-top: 2px; line-height: 1.35; }
</style>"""


def _p(v, dec=0, teken=False):
    """Fractie -> '12%' (of '+12%')."""
    if v is None or pd.isna(v):
        return "–"
    return f"{v * 100:{'+' if teken else ''}.{dec}f}%".replace(".", ",")


def _maal(v):
    """9.46 -> '9,5×', 42.3 -> '42×' (onder 30× één decimaal)."""
    return "–" if v is None else (f"{v:.1f}×" if v < 30 else f"{v:.0f}×").replace(".", ",")


def _chip(tekst, kleur):
    fg, bg = KLEUR[kleur]
    return (f'<span style="font-size:0.72rem;padding:1px 7px;border-radius:6px;white-space:nowrap;'
            f'color:{fg};background:{bg}">{html.escape(tekst)}</span>')


def _kleur_score(v):
    return "grijs" if v is None else "groen" if v >= 65 else "geel" if v >= 40 else "rood"


def _hoofd_verhouding(f: dict):
    """De verhouding om in zinnen te gebruiken: winst, anders bedrijfswinst, kasstroom of boekwaarde."""
    for k in ("kw", "ev_ebit", "p_fcf", "p_b"):
        v = f.get(k)
        if isinstance(v, dict) and "goedkoper" in v:
            return k, v
    return None, None


def _verlies_maanden(f: dict) -> int:
    return max((f[k]["verlies"] for k in ("kw", "ev_ebit") if isinstance(f.get(k), dict)), default=0)


def punten(f: dict, s: dict) -> tuple:
    """Sterke punten en aandachtspunten in gewone taal, belangrijkste eerst.
    Elk punt is (tekst, toelichting)."""
    sterk, let_op = [], []
    k, v = _hoofd_verhouding(f)
    zin_waarde = (f"Je betaalt {_maal(v['nu'])} de {WAARVAN[k]}", f"normaal {_maal(v['normaal'])}") if v else None
    verwacht, bron = verwachte_groei(f)
    ing = f.get("ingeprijsd")
    pio, moh = f.get("piotroski") or [], f.get("mohanram") or []
    trends = [f.get(f"marge_{m}_trend") for m in ("bruto", "netto")]
    z = f.get("altman")

    # waarde
    if s.get("waarde_label") == "goedkoop" and zin_waarde:
        sterk.append(zin_waarde)
    elif s.get("waarde_label") == "duur" and zin_waarde:
        let_op.append((zin_waarde[0], zin_waarde[1] + ": duurder dan normaal"))
    elif any(isinstance(f.get(x), dict) and f[x]["nu"] is not None and "goedkoper" not in f[x]
             for x in NAAM):
        let_op.append(("Geen vergelijking met vroeger mogelijk", "te veel maanden met verlies"))
    if isinstance(f.get("kw"), dict) and f["kw"]["nu_verlies"]:
        let_op.append(("Het bedrijf maakt nu verlies", "de laatste 12 maanden"))
    # groei
    lab = s.get("groei_label")
    if ing is not None and verwacht is not None:
        if lab == "beter":
            sterk.append((f"Verwacht {_p(verwacht, 0, True)} groei, markt rekent op {_p(ing, 0, True)}",
                          "per jaar" + (", op winst" if f.get("ingeprijsd_basis") == "winst" else "")))
        elif lab == "minder":
            let_op.append((f"Markt rekent op {_p(ing, 0, True)}, verwacht maar {_p(verwacht, 0, True)}",
                           "per jaar" + (", op winst" if f.get("ingeprijsd_basis") == "winst" else "")))
    elif lab == "sterk":
        sterk.append(("Sterke verwachte groei", f"{_p(verwacht, 0, True)} per jaar"))
    elif lab == "zwak":
        let_op.append(("Weinig groei verwacht", f"{_p(verwacht, 0, True)} per jaar"))
    # veiligheid
    if z is not None:
        if z > 2.99:
            sterk.append(("Financieel zeer gezond", "laag faillissementsrisico"))
        elif z < 1.81:
            let_op.append(("Verhoogd faillissementsrisico", f"score {z:.1f}".replace(".", ",")))
        else:
            let_op.append(("Faillissementsrisico niet helemaal laag", f"score {z:.1f}".replace(".", ",")))
    # winst die schommelt of net gesprongen is
    n_verlies = _verlies_maanden(f)
    if n_verlies >= 6 or (len(moh) > 3 and moh[3] is False):
        let_op.append(("Winst schommelt sterk",
                       f"{n_verlies} van {f.get('maanden', 36)} maanden verlies" if n_verlies else "wisselende jaren"))
    if "verdubbeld" in f.get("waarschuwingen", []):
        let_op.append(("Winst net verdubbeld: piek of blijvend?", ""))
    if "net_winstgevend" in f.get("waarschuwingen", []):
        let_op.append(("Pas net winstgevend", "vorig jaar nog verlies"))
    # kwaliteit
    if trends and all(t == 1 for t in trends):
        sterk.append(("Marges stijgen", f"netto {_p(f.get('marge_netto'))}"))
    elif trends and all(t == -1 for t in trends):
        let_op.append(("Marges dalen", f"netto {_p(f.get('marge_netto'))}"))
    r = f.get("rendement_nu")
    if r is not None and r >= 0.15:
        sterk.append(("Hoog rendement op kapitaal", _p(r)))
    p, n = _telling(pio)
    if n >= 5 and p >= 7:
        sterk.append(("Bedrijf wordt beter", f"{p} van {n} tests"))
    elif n >= 5 and p <= 3:
        let_op.append(("Bedrijf verslechtert", f"maar {p} van {n} tests"))
    if f.get("kwartalen") and f.get("verslagen") == f["kwartalen"]:
        sterk.append(("Versloeg steeds de verwachtingen", f"{f['verslagen']} van {f['kwartalen']} kwartalen"))
    if f.get("peg") is not None and f["peg"] <= 1:
        sterk.append(("Redelijke prijs voor de groei", f"prijs voor groei {f['peg']:.1f}".replace(".", ",")))
    # losse tests die er echt toe doen, in gewone taal
    if len(pio) > 6 and pio[6] is False:
        let_op.append(("Nieuwe aandelen uitgegeven", "jouw belang verwatert"))
    if len(pio) > 4 and pio[4] is False:
        let_op.append(("Schuld gestegen", "t.o.v. vorig jaar"))
    if len(pio) > 1 and pio[1] is False:
        let_op.append(("Negatieve kasstroom", "laatste boekjaar"))
    if moh and moh[0] is False:
        let_op.append(("Rendement onder 6%", "op alle bezittingen"))
    return sterk[:MAX_PUNTEN], let_op[:MAX_PUNTEN]


def _punten_html(titel, lijst, kleur, icoon):
    if not lijst:
        return ""
    fg = KLEUR[kleur][0]
    rijen = "".join(
        f'<div class="fk-pt"><span style="color:{fg}">{icoon}</span><span>{html.escape(t)}'
        + (f' <span class="fk-sub">· {html.escape(sub)}</span>' if sub else "") + "</span></div>"
        for t, sub in lijst)
    return f'<div class="fk-kop" style="color:{fg}">{titel}</div>{rijen}'


def _tabel(rijen) -> str:
    return "<table>" + "".join(
        f'<tr><td class="l">{l}</td><td class="n">{n}</td><td class="v">{v}</td></tr>'
        for l, n, v in rijen) + "</table>"


def _tests_html(titel, namen, tests) -> str:
    """Regel 'Verbetering 8/9 · mist 1 ▾' die openklapt tot de lijst met ✓ en ✗."""
    if not tests:
        return ""
    goed, n = _telling(tests)
    gemist = sum(t is False for t in tests)
    rechts = "alles gehaald ▾" if not gemist else f"{gemist} gemist ▾"
    lijst = "".join(
        f'<div style="display:flex;gap:6px;font-size:0.78rem;line-height:1.5">'
        f'<span style="width:12px;color:{KLEUR["groen"][0] if t else KLEUR["rood"][0] if t is False else "inherit"}">'
        f'{"✓" if t else "✗" if t is False else "–"}</span>'
        f'<span style="{"" if t is not None else "opacity:0.5"}">{html.escape(naam)}'
        f'{"" if t is not None else " (geen data)"}</span></div>'
        for naam, t in zip(namen, tests))
    return (f'<details class="fk-test"><summary style="display:flex;font-size:0.82rem;padding:2px 0;'
            f'font-variant-numeric:tabular-nums"><span style="width:44%;opacity:0.7">{titel}</span>'
            f'<span style="width:22%;text-align:right;font-weight:600">{goed}/{n}</span>'
            f'<span style="width:34%;text-align:right;opacity:0.55;font-size:0.75rem">{rechts}</span></summary>'
            f'<div style="padding:2px 0 6px 4px">{lijst}</div></details>')


def cijfers_html(f: dict, mf) -> str:
    """De compacte cijfertabel achter 'Toon alle cijfers'."""
    fin = f.get("financieel")
    delen = []
    # waarde
    rijen = []
    for k in NAAM:
        v = f.get(k)
        if not isinstance(v, dict):
            continue
        if v["nu"] is None:
            rijen.append((NAAM[k], "–", "verlies" if v["nu_verlies"] else "geen cijfers"))
        else:
            rijen.append((NAAM[k], _maal(v["nu"]),
                          f"normaal {_maal(v['normaal'])}" if "goedkoper" in v else "geen vergelijking"))
    if f.get("forward_kw"):
        rijen.append(("Je betaalt × verwachte winst", _maal(f["forward_kw"]),
                      f"prijs voor groei {f['peg']:.1f}".replace(".", ",") if f.get("peg") is not None else ""))
    if mf:
        rijen.append(("Goedkoop + rendabel", f"#{mf[0]}", f"van {mf[1]}"))
    if rijen:
        voet = (f"Normaal = mediaan van de maanden met winst in de afgelopen {WAARDE_HISTORIE_JAREN} jaar."
                + (f" Cijfers: {f['bron_nu']}." if f.get("bron_nu") else "")
                + (f" × {' en × '.join(f['reserve'])} nu uit Yahoo's eigen samenvatting (jaarcijfers ontbraken)."
                   if f.get("reserve") else ""))
        delen.append(f'<div class="fk-sec">Waarde</div>{_tabel(rijen)}<div class="fk-voet">{voet}</div>')
    # groei
    verwacht, bron = verwachte_groei(f)
    rijen = []
    if f.get("ingeprijsd") is not None:
        rijen.append(("Ingeprijsd door markt", f.get("ingeprijsd_teken", "") + _p(f["ingeprijsd"], 0, True),
                      f"op winst: {f.get('ingeprijsd_reden', '')}" if f.get("ingeprijsd_basis") == "winst"
                      else "op vrije kasstroom"))
    if verwacht is not None and bron == "analisten":
        rijen.append(("Verwacht (omzet)", _p(verwacht, 0, True), "analisten"))
    if f.get("omzetgroei") is not None:
        rijen.append(("Werkelijk (omzet)", _p(f["omzetgroei"], 0, True),
                      f.get("omzetgroei_bron") or "afgelopen periode"))
    if f.get("kwartalen"):
        rijen.append(("Verwachting verslagen", f"{f['verslagen']}/{f['kwartalen']}", "kwartalen"))
    if rijen:
        delen.append(f'<div class="fk-sec">Groei per jaar</div>{_tabel(rijen)}')
    # kwaliteit
    rijen = []
    marges = [f.get(f"marge_{m}") for m in ("bruto", "netto")]
    if any(m is not None for m in marges):
        trends = [f.get(f"marge_{m}_trend") for m in ("bruto", "netto")]
        richting = {(1, 1): "beide stijgend", (-1, -1): "beide dalend", (0, 0): "gelijk"}.get(
            tuple(trends), " / ".join({1: "stijgt", -1: "daalt", 0: "gelijk"}.get(t, "–") for t in trends))
        rijen.append(("Marge bruto / netto",
                      " / ".join("–" if m is None else f"{m * 100:.0f}%" for m in marges), richting))
    if f.get("rendement_nu") is not None:
        rijen.append(("Rendement op vermogen" if fin else "Rendement op kapitaal", _p(f["rendement_nu"]),
                      f"bereik {_p(f['rendement_min'])} tot {_p(f['rendement_max'])}"))
    tests = "".join(_tests_html(titel, namen, f.get(sleutel)) for titel, namen, sleutel in (
        ("Verbetering", PIOTROSKI_NAMEN, "piotroski"), ("Groeikwaliteit", MOHANRAM_NAMEN, "mohanram")))
    if rijen or tests:
        delen.append(f'<div class="fk-sec">Kwaliteit</div>{_tabel(rijen) if rijen else ""}{tests}')
    # veiligheid
    z = f.get("altman")
    if z is not None:
        oordeel = "veilig boven 3" if z > 2.99 else "grijs gebied 1,8–3" if z >= 1.81 else "verhoogd onder 1,8"
        delen.append('<div class="fk-sec">Veiligheid</div>'
                     + _tabel([("Faillissementsrisico", f"{z:.1f}".replace(".", ","), oordeel)]))
    elif fin:
        delen.append('<div class="fk-sec">Veiligheid</div><div class="fk-voet">'
                     "Niet te meten bij banken en verzekeraars.</div>")
    return "".join(delen) or '<div class="fk-voet">Geen cijfers beschikbaar.</div>'


def _koers_tekst(ranking: pd.DataFrame, ticker: str) -> str:
    """Actuele koers uit de ranglijst: euro-aandelen in euro, al het andere in dollars
    (yen, Hongkongse dollars, ponden enz. omgerekend, met ≈ ervoor)."""
    if ranking is None or ranking.empty or "ticker" not in ranking:
        return ""
    r = ranking.loc[ranking["ticker"] == ticker]
    if r.empty:
        return ""
    valuta = fetch_meta(ticker)["valuta"]
    cur = "EUR" if valuta == "EUR" else "USD"
    v = r.iloc[0].get(f"prijs_{cur.lower()}")
    if v is None or pd.isna(v):
        return ""
    return ("" if valuta in ("EUR", "USD") else "≈ ") + format_price(float(v), cur)


def fund_kaart_html(rang: int, naam: str, f: dict, s: dict, mom, mf, koers: str = "") -> str:
    verwacht, _ = verwachte_groei(f)
    ing, teken = f.get("ingeprijsd"), f.get("ingeprijsd_teken", "")

    # ---- kop (altijd zichtbaar) ----
    eind = s.get("eind")
    fg, bg = KLEUR[_kleur_score(eind)]
    cirkel = (f'<div style="width:42px;height:42px;border-radius:50%;flex-shrink:0;display:flex;'
              f'align-items:center;justify-content:center;font-weight:700;color:{fg};background:{bg}">'
              f'{"–" if eind is None else f"{eind:.0f}"}</div>')
    mom_tekst = "" if mom is None else f" · 1 mnd {_pct_nl(mom, 1)}{' ↗' if mom > 0 else ' ↘'}"
    groeiregel = ""
    op_winst = " (op winst)" if f.get("ingeprijsd_basis") == "winst" else ""
    if ing is not None and verwacht is not None and s.get("groei_label") in LABEL_KLEUR:
        groeiregel = (f'<div style="font-size:0.8rem;margin-top:5px;opacity:0.85">Markt rekent op '
                      f'<b>{teken}{_p(ing, 0, True)}</b>{op_winst} · verwacht <b style="color:'
                      f'{KLEUR[LABEL_KLEUR[s["groei_label"]]][0]}">{_p(verwacht, 0, True)}</b> per jaar</div>')
    chips = []
    for deel, titel in (("waarde", "Waarde"), ("groei", "Groei"), ("kwaliteit", "Kwaliteit")):
        lab = s.get(f"{deel}_label")
        if lab:
            chips.append(_chip(f"{titel}: {GROEI_TEKST.get(lab, lab) if deel == 'groei' else lab}",
                               LABEL_KLEUR[lab]))
    if s.get("veiligheid_label"):
        chips.append(_chip(s["veiligheid_label"].capitalize(), LABEL_KLEUR[s["veiligheid_label"]]))
    kop = (f'<div style="display:flex;align-items:center;gap:8px">'
           f'<span style="opacity:0.45;font-size:0.75rem;width:14px;flex-shrink:0">{rang}</span>'
           f'<div style="flex:1;min-width:0"><div style="display:flex;gap:6px;align-items:baseline">'
           f'<span style="font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
           f'{html.escape(naam)}</span>'
           + (f'<span style="font-size:0.85rem;white-space:nowrap;flex-shrink:0;opacity:0.8;'
              f'font-variant-numeric:tabular-nums">{html.escape(koers)}</span>' if koers else "")
           + f'</div><div style="font-size:0.78rem;opacity:0.6">'
           f'{html.escape(f["ticker"])}{mom_tekst}</div></div>{cirkel}'
           f'<span class="fk-pijl" style="opacity:0.45;font-size:0.8rem">▾</span></div>'
           f'{groeiregel}<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px">{"".join(chips)}</div>')

    # ---- uitgeklapt: sterke punten, aandachtspunten, en daaronder de cijfers ----
    if f.get("fout") and len(f) <= 2:
        inhoud = f'<div class="fk-voet" style="margin-top:8px">Niet berekend: {html.escape(f["fout"])}</div>'
    else:
        sterk, let_op = punten(f, s)
        inhoud = (_punten_html("Sterke punten", sterk, "groen", "✓")
                  + _punten_html("Aandachtspunten", let_op, "geel", "⚠")
                  or '<div class="fk-voet">Geen opvallende punten.</div>')
        inhoud += (f'<details style="margin-top:6px"><summary style="font-size:0.78rem;opacity:0.6;'
                   f'text-align:center;padding:4px 0"><span class="fk-toon">Toon alle cijfers ▾</span>'
                   f'<span class="fk-verberg">Verberg cijfers ▴</span></summary>{cijfers_html(f, mf)}</details>')
    return (f'<div class="fk" style="border:1px solid rgba(128,128,128,0.22);border-radius:8px;'
            f'padding:8px 10px;margin-bottom:6px"><details><summary>{kop}</summary>'
            f'<div style="margin-top:6px;border-top:1px solid rgba(128,128,128,0.22)">{inhoud}</div>'
            f'</details></div>')


HOE_LEES_IK_DIT = f"""
**Eindscore (0–100)**: waarde {{w}}, kwaliteit {{k}}, groei {{g}}, veiligheid {{v}}. Ontbreekt een onderdeel, dan tellen de andere naar verhouding zwaarder.

**Waarde: "je betaalt X× ..."**: hoeveel jaar winst je betaalt voor het aandeel, en vergeleken met elke maand van de afgelopen {WAARDE_HISTORIE_JAREN} jaar ('normaal' = de mediaan). Lager dan normaal = goedkoper.
- **Je betaalt × de winst** = koers/winst (K/W, P/E)
- **Je betaalt × bedrijfswinst** = ondernemingswaarde/bedrijfswinst (EV/EBIT): schulden tellen mee
- **Je betaalt × kasstroom** = koers/vrije kasstroom (P/FCF): het geld dat overblijft na investeringen
- **Je betaalt × boekwaarde** = koers/boekwaarde (P/B), alleen bij banken en verzekeraars
- **Je betaalt × verwachte winst** = forward K/W: op de winst die analisten volgend jaar verwachten
- **Prijs voor groei** = PEG: forward K/W gedeeld door de verwachte winstgroei. Rond 1 of lager is redelijk
- **Goedkoop + rendabel** = Magic Formula (Greenblatt): rang binnen jouw lijst

**Groei**: *ingeprijsd* = de jaarlijkse groei van de vrije kasstroom die de koers rechtvaardigt (reverse DCF: {DCF_RENDEMENT:.0%} rendement, {DCF_JAREN} jaar, daarna {DCF_EINDGROEI:.1%}). Is de vrije kasstroom negatief, bijvoorbeeld omdat een bedrijf zwaar investeert in fabrieken of datacenters, dan rekent de code vanaf de winst; dan staat er *(op winst)*. Winst verdeelt grote investeringen over de jaren dat ze gebruikt worden, terwijl de vrije kasstroom ze in één keer aftrekt. Is ook de winst negatief, dan is er niets om vanaf te rekenen en staat er niets. *Verwacht* = omzetgroei volgend jaar volgens analisten. *Werkelijk* = omzetgroei per kwartaal t.o.v. hetzelfde kwartaal een jaar eerder, gemiddeld over de laatste kwartalen (zonder kwartaalcijfers: gemiddeld per jaar). Verwacht hoger dan ingeprijsd = ruimte in de koers.

**Kwaliteit**
- **Marges**: van elke 100 omzet wat er overblijft na inkoop (bruto) en als winst (netto)
- **Rendement op kapitaal** = ROIC (bedrijfswinst na belasting ÷ schuld + eigen vermogen); bij banken rendement op eigen vermogen (ROE). Boven 15% is sterk
- **Verbetering** = Piotroski F-score: 9 tests of het bedrijf beter wordt dan vorig jaar. Tik erop voor de tests
- **Groeikwaliteit** = Mohanram G-score: rendabel, stabiel en investeert. Benaderd met vaste drempels. Tik erop voor de tests

**Veiligheid: faillissementsrisico** = Altman Z-score. Boven 3 veilig, 1,8–3 grijs gebied, daaronder verhoogd. Hoog risico kan ook hoge opbrengst betekenen.

Cyclische bedrijven (bv. chips, grondstoffen) lijken op het hoogtepunt van hun cyclus vaak goedkoop, omdat de winst dan tijdelijk hoog is. Let daar op bij het aandachtspunt "winst schommelt sterk".
"""


def render_fundamenteel(tickers: list, weights: dict, modus: str):
    """Ranglijst van losse aandelen op de eindscore, met per aandeel uitklapbare details."""
    aandelen = [t for t in tickers if str(fetch_meta(t).get("soort") or "").upper() != "ETF"]
    if not aandelen:
        st.info("Geen losse aandelen in deze lijst (ETF's worden overgeslagen).")
        return
    if not st.toggle("Berekenen", key="fund_aan",
                     help="Haalt jaarcijfers en groeiverwachtingen op bij Yahoo (de eerste keer even "
                          "wachten, daarna 24 uur bewaard). Staat uit zodat de pagina snel blijft."):
        st.caption("Zet 'Berekenen' aan om de analyse te laden.")
        return

    with st.spinner("Jaarcijfers ophalen en doorrekenen…"):
        fund = [f or {"ticker": t, "fout": "onbekende fout"}
                for t, f in zip(aandelen, parallel(_fund_veilig, aandelen))]
        mom = [(m or {}).get("pct") for m in parallel(momentum_rij, aandelen)]

    mf = magic_formula([f for f in fund if "fout" not in f])
    try:  # actuele koersen uit hetzelfde gedeelde geheugen als de ranglijst (geen extra verzoeken)
        ranking, _, _, _ = laad_ranking(tickers, weights, modus == "watchlist")
    except Exception:  # noqa: BLE001
        ranking = None
    lijst = [(f, scores(f), m) for f, m in zip(fund, mom)]
    lijst.sort(key=lambda x: -1 if x[1]["eind"] is None else x[1]["eind"], reverse=True)

    st.caption("Tik op een aandeel voor de sterke punten en aandachtspunten. ETF's tellen niet mee.")
    st.markdown(FUND_CSS, unsafe_allow_html=True)
    st.markdown(RASTER.format("".join(
        fund_kaart_html(i + 1, fetch_meta(f["ticker"])["naam"], f, s, m, mf.get(f["ticker"]),
                        _koers_tekst(ranking, f["ticker"]))
        for i, (f, s, m) in enumerate(lijst))), unsafe_allow_html=True)
    g = SCORE_GEWICHTEN
    if st.toggle("Hoe lees ik dit?", key="fund_uitleg"):
        st.markdown(HOE_LEES_IK_DIT.format(w=f"{g['waarde']:.0%}", k=f"{g['kwaliteit']:.0%}",
                                           g=f"{g['groei']:.0%}", v=f"{g['veiligheid']:.0%}"))


# Twee fragmenten die elk in hun eigen tempo verversen. Alleen de tickers van
# het gekozen tabblad worden opgehaald.
st.fragment(run_every=interval if auto else None)(render_live_boven)(
    lijst, weights, soort, modus
)
if soort == "aandelen" and lijst:
    # Buiten de fragmenten: ververst niet elke paar seconden (slotkoersen volstaan)
    with st.expander("📊 Aandelenanalyse: waarde, groei, kwaliteit en veiligheid"):
        render_fundamenteel(lijst, weights, modus)
if soort == "crypto":
    # Buiten de fragmenten: cycluskaart en grafiek verversen niet elke paar
    # seconden mee, zodat inzoomen blijft staan. De data wordt 6 uur bewaard.
    render_cyclus_kaart()
    with st.expander("📈 BTC-prijs, Fear & Greed en RSI (dagbasis)"):
        st.fragment(render_fng_grafiek)()  # klikken in de grafiek ververst alleen de grafiek
st.fragment(run_every=interval if auto else None)(render_live_onder)(
    lijst, weights, soort, modus, tuple(portefeuille)
)

if modus == "watchlist":
    st.fragment(run_every=rang_interval if auto else None)(render_ranglijst)(
        lijst, weights, soort, tuple(portefeuille), rang_interval if auto else None
    )
