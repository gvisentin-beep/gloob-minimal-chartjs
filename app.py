from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Mappa "asset" -> file CSV
ASSET_FILES: Dict[str, str] = {
    "ls80": "ls80.csv",
    "gold": "gold.csv",
    "btc": "btc.csv",
    # se hai altri asset, aggiungili qui
}

# Cache in memoria (evita riletture continue)
_CACHE: Dict[Tuple[str, float], pd.Series] = {}


def _clean_numeric(s: pd.Series) -> pd.Series:
    """
    Converte stringhe numeriche europee (es: '1.234,56') in float.
    """
    s = s.astype(str).str.strip()
    # rimuove separatore migliaia e converte virgola decimale in punto
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _read_asset(asset: str) -> pd.Series:
    """
    Legge un CSV e ritorna una Series con DateTimeIndex e valori float.
    Accetta separatori ',' o ';' e nomi colonna variabili.
    """
    if asset not in ASSET_FILES:
        raise ValueError(f"Asset sconosciuto: {asset}")

    path = DATA_DIR / ASSET_FILES[asset]
    if not path.exists():
        raise ValueError(f"File CSV mancante per {asset}: {path}")

    mtime = path.stat().st_mtime
    cache_key = (asset, mtime)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    # 1) prova autodetect separatore
    df = None
    last_err: Optional[Exception] = None
    for kwargs in (
        dict(sep=None, engine="python"),   # autodetect
        dict(sep=";"),
        dict(sep=","),
    ):
        try:
            df = pd.read_csv(path, **kwargs)
            if df is not None and df.shape[1] >= 2:
                break
        except Exception as e:
            last_err = e
            df = None

    if df is None or df.shape[1] < 2:
        msg = f"CSV {asset} deve avere almeno 2 colonne (data, valore)"
        if last_err:
            msg += f" | Dettaglio lettura: {type(last_err).__name__}: {last_err}"
        raise ValueError(msg)

    # Normalizza nomi colonna
    cols = [str(c).strip().lower() for c in df.columns]
    df.columns = cols

    # Se troviamo colonne “date” e “value” (o simili), usiamole; altrimenti prime due colonne
    date_col = None
    value_col = None

    for c in cols:
        if c in ("date", "data", "dt", "giorno"):
            date_col = c
            break

    for c in cols:
        if c in ("value", "valore", "close", "prezzo", "price", "nav"):
            value_col = c
            break

    if date_col is None:
        date_col = cols[0]
    if value_col is None:
        # se la prima colonna è data, prendiamo la seconda; altrimenti la prima numerica utile
        value_col = cols[1] if len(cols) >= 2 else cols[0]

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "value"]

    # Date: dayfirst=True (Italia), ma senza rompere se formato ISO
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True, utc=False)
    out["value"] = _clean_numeric(out["value"])

    out = out.dropna(subset=["date", "value"]).sort_values("date")
    if out.empty:
        raise ValueError(f"CSV {asset} letto ma senza righe valide (date/valore)")

    s = out.set_index("date")["value"]
    # elimina duplicati sulla data (tiene ultimo)
    s = s[~s.index.duplicated(keep="last")].sort_index()

    _CACHE.clear()  # semplice: puliamo cache vecchia
    _CACHE[cache_key] = s
    return s


def _freq_rule(freq: str) -> str:
    """
    Converte freq da query string a regola Pandas.
    Importante: usare 'ME' (month end) al posto di 'M' (deprecato in pandas nuove).
    """
    f = (freq or "daily").strip().lower()
    mapping = {
        "d": "D",
        "day": "D",
        "daily": "D",
        "w": "W",
        "week": "W",
        "weekly": "W",
        "m": "ME",
        "month": "ME",
        "monthly": "ME",
        "q": "QE",
        "quarter": "QE",
        "quarterly": "QE",
        "y": "YE",
        "year": "YE",
        "yearly": "YE",
        "annual": "YE",
    }
    return mapping.get(f, "D")


def _resample(s: pd.Series, freq: str) -> pd.Series:
    rule = _freq_rule(freq)
    if rule == "D":
        return s.dropna()
    # last() su fine periodo
    return s.resample(rule).last().dropna()


def _normalize_100(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    return (s / float(s.iloc[0])) * 100.0


@app.route("/")
def home():
    # Se hai index.html in templates, lo mostriamo
    # Se non esiste, non è un problema per le API
    try:
        return render_template("index.html")
    except Exception:
        return "OK - API attive. Prova /api/combined?freq=monthly", 200


@app.route("/api/data")
def api_data():
    asset = (request.args.get("asset") or "").strip().lower()
    freq = (request.args.get("freq") or "daily").strip().lower()

    try:
        s = _read_asset(asset)
        s = _resample(s, freq)

        payload = {
            "asset": asset,
            "base_date": s.index[0].date().isoformat() if not s.empty else None,
            "freq": freq,
            "labels": [d.strftime("%Y-%m") if _freq_rule(freq) == "ME" else d.strftime("%Y-%m-%d") for d in s.index],
            "values": [round(float(v), 4) for v in s.values],
            "points": int(len(s)),
        }
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/combined")
def api_combined():
    freq = (request.args.get("freq") or "monthly").strip().lower()

    try:
        # Benchmark: ls80
        bench = _normalize_100(_resample(_read_asset("ls80"), freq))

        # Portfolio: 80% ls80, 15% gold, 5% btc (se disponibili)
        # Se gold/btc mancano, ripiega su ls80
        ls = _resample(_read_asset("ls80"), freq)

        try:
            gold = _resample(_read_asset("gold"), freq)
        except Exception:
            gold = ls.copy()

        try:
            btc = _resample(_read_asset("btc"), freq)
        except Exception:
            btc = ls.copy()

        # Allineamento date comuni
        df = pd.concat(
            {
                "ls80": ls,
                "gold": gold,
                "btc": btc,
            },
            axis=1
        ).dropna()

        if df.empty:
            raise ValueError("Dati insufficienti per combinare le serie (date non sovrapposte).")

        portfolio = (0.80 * df["ls80"] + 0.15 * df["gold"] + 0.05 * df["btc"])
        portfolio = _normalize_100(portfolio)

        # Allinea portfolio e benchmark sull’intersezione
        both = pd.concat({"benchmark": bench, "portfolio": portfolio}, axis=1).dropna()
        if both.empty:
            raise ValueError("Benchmark e portafoglio non hanno date comuni dopo il resample.")

        labels = [d.strftime("%Y-%m") if _freq_rule(freq) == "ME" else d.strftime("%Y-%m-%d") for d in both.index]

        payload = {
            "base_date": both.index[0].date().isoformat(),
            "freq": freq,
            "labels": labels,
            "points": int(len(both)),
            "series": {
                "benchmark": [round(float(v), 4) for v in both["benchmark"].values],
                "portfolio": [round(float(v), 4) for v in both["portfolio"].values],
            },
        }
        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Render usa gunicorn, ma in locale puoi lanciare così:
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
