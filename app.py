from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ASSET_FILES: Dict[str, str] = {
    "ls80": "ls80.csv",
    "gold": "gold.csv",
    "btc": "btc.csv",
}

# Pesi portafoglio (modifica qui se vuoi)
W_LS80 = 0.80
W_GOLD = 0.15
W_BTC = 0.05


# --- Helpers ---------------------------------------------------------------

def _detect_and_read_csv(path: Path) -> pd.DataFrame:
    """
    Legge CSV auto-detectando separatore (, o ;) e gestendo BOM/righe vuote.
    """
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    # sep=None + engine='python' => tenta autodetect separatore
    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        skip_blank_lines=True,
    )
    if df is None or df.empty:
        raise ValueError(f"CSV vuoto o illeggibile: {path.name}")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _pick_date_value_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Cerca colonne ragionevoli per data e valore.
    Se non le trova, usa le prime 2 colonne.
    """
    cols = list(df.columns)

    date_candidates = {"date", "data", "datetime", "timestamp"}
    value_candidates = {"value", "valore", "close", "prezzo", "price", "adj close", "adj_close", "last"}

    date_col = None
    value_col = None

    for c in cols:
        if c in date_candidates:
            date_col = c
            break

    for c in cols:
        if c in value_candidates:
            value_col = c
            break

    if date_col and value_col:
        return date_col, value_col

    # fallback: prime due colonne
    if len(cols) < 2:
        raise ValueError("Il CSV deve avere almeno 2 colonne (data, valore).")
    return cols[0], cols[1]


def _to_series(df: pd.DataFrame, asset: str) -> pd.Series:
    df = _normalize_columns(df)

    date_col, value_col = _pick_date_value_columns(df)

    out = pd.DataFrame()
    out["date"] = df[date_col]
    out["value"] = df[value_col]

    # Date: gestisce dd/mm/yyyy (Italia) e vari formati
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)

    # Value: gestisce virgole decimali
    out["value"] = (
        out["value"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["date", "value"]).sort_values("date")
    if out.empty:
        raise ValueError(f"CSV {asset} non contiene righe valide (data/valore).")

    s = out.set_index("date")["value"].astype(float)
    # elimina duplicati data (tiene ultimo)
    s = s[~s.index.duplicated(keep="last")]
    return s


def _freq_to_pandas(freq: str) -> str:
    """
    Mappa frequenze “umane” a pandas offset.
    Nota: per il mensile, pandas recente preferisce 'ME' al posto di 'M'.
    """
    f = (freq or "monthly").strip().lower()

    mapping = {
        "daily": "D",
        "day": "D",
        "d": "D",
        "weekly": "W-FRI",
        "week": "W-FRI",
        "w": "W-FRI",
        "monthly": "ME",   # <-- importante
        "month": "ME",
        "m": "ME",         # <-- importante
        "quarterly": "QE",
        "quarter": "QE",
        "q": "QE",
        "yearly": "YE",
        "year": "YE",
        "y": "YE",
    }

    # se l'utente passa già un offset pandas, gestiamo il caso 'M'
    if f == "m":
        return "ME"
    if f == "M":
        return "ME"

    return mapping.get(f, "ME")


def _resample(s: pd.Series, freq: str) -> pd.Series:
    rule = _freq_to_pandas(freq)
    # last() = ultimo valore del periodo
    return s.resample(rule).last().dropna()


def _normalize_100(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    base = float(s.iloc[0])
    if base == 0:
        return s
    return (s / base) * 100.0


def _read_asset(asset: str) -> pd.Series:
    asset = asset.strip().lower()
    if asset not in ASSET_FILES:
        raise ValueError(f"Asset non supportato: {asset}. Disponibili: {list(ASSET_FILES.keys())}")

    path = DATA_DIR / ASSET_FILES[asset]
    df = _detect_and_read_csv(path)
    return _to_series(df, asset)


def _series_to_payload(asset: str, s: pd.Series, freq: str) -> dict:
    s = _resample(s, freq)
    s = _normalize_100(s)

    labels = [d.strftime("%Y-%m") for d in s.index.to_pydatetime()]
    values = [round(float(v), 4) for v in s.values]

    base_date = s.index[0].strftime("%Y-%m-%d") if not s.empty else None
    return {
        "asset": asset,
        "base_date": base_date,
        "freq": (freq or "monthly"),
        "points": len(values),
        "labels": labels,
        "values": values,
    }


# --- Routes ----------------------------------------------------------------

@app.get("/")
def root():
    # Se hai una pagina in /static (es. index.html), servila
    static_index = BASE_DIR / "static" / "index.html"
    if static_index.exists():
        return send_from_directory(BASE_DIR / "static", "index.html")
    return "OK - Flask up. Prova /api/data?asset=ls80&freq=monthly oppure /api/combined?freq=monthly", 200


@app.get("/api/data")
def api_data():
    asset = request.args.get("asset", "ls80")
    freq = request.args.get("freq", "monthly")
    try:
        s = _read_asset(asset)
        return jsonify(_series_to_payload(asset, s, freq))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/combined")
def api_combined():
    freq = request.args.get("freq", "monthly")
    try:
        s_ls80 = _read_asset("ls80")
        s_gold = _read_asset("gold")
        s_btc = _read_asset("btc")

        ls80 = _normalize_100(_resample(s_ls80, freq))
        gold = _normalize_100(_resample(s_gold, freq))
        btc = _normalize_100(_resample(s_btc, freq))

        # allineo sulle date comuni
        df = pd.concat(
            {
                "ls80": ls80,
                "gold": gold,
                "btc": btc,
            },
            axis=1,
        ).dropna()

        if df.empty:
            raise ValueError("Dopo l’allineamento delle date comuni, non ci sono dati sufficienti.")

        # Portfolio = somma pesata degli indici normalizzati a 100
        portfolio = (df["ls80"] * W_LS80) + (df["gold"] * W_GOLD) + (df["btc"] * W_BTC)
        benchmark = df["ls80"]

        labels = [d.strftime("%Y-%m") for d in df.index.to_pydatetime()]

        payload = {
            "base_date": df.index[0].strftime("%Y-%m-%d"),
            "freq": (freq or "monthly"),
            "labels": labels,
            "points": int(df.shape[0]),
            "series": {
                "benchmark": [round(float(v), 4) for v in benchmark.values],
                "portfolio": [round(float(v), 4) for v in portfolio.values],
            },
        }
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # locale
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
