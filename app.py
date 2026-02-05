from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Optional

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

# Pesi portafoglio
W_LS80 = 0.80
W_GOLD = 0.15
W_BTC = 0.05


# -------------------------------------------------------------------------
# CSV reading (robusto)
# -------------------------------------------------------------------------

def _read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Legge CSV in modo robusto:
    - prova autodetect separatore
    - se fallisce o trova <2 colonne, prova ; e ,
    - gestisce BOM (utf-8-sig)
    """
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    # tentativi in ordine
    attempts = [
        {"sep": None, "engine": "python"},  # autodetect
        {"sep": ";", "engine": "python"},
        {"sep": ",", "engine": "python"},
        {"sep": r"\s+", "engine": "python"},  # whitespace
    ]

    last_err: Optional[Exception] = None
    for opts in attempts:
        try:
            df = pd.read_csv(
                path,
                encoding="utf-8-sig",
                skip_blank_lines=True,
                **opts,
            )
            if df is None or df.empty:
                continue

            # Se ha 1 colonna sola, potrebbe essere separatore non riconosciuto
            if df.shape[1] < 2:
                continue

            return df
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise ValueError(f"Impossibile leggere il CSV {path.name}. Ultimo errore: {last_err}")

    raise ValueError(f"CSV vuoto o illeggibile: {path.name}")


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
    if len(cols) < 2:
        raise ValueError("Il CSV deve avere almeno 2 colonne (data, valore).")

    date_candidates = {"date", "data", "datetime", "timestamp", "time"}
    value_candidates = {
        "value", "valore", "close", "prezzo", "price",
        "adj close", "adj_close", "last", "nav"
    }

    date_col = next((c for c in cols if c in date_candidates), None)
    value_col = next((c for c in cols if c in value_candidates), None)

    if date_col and value_col:
        return date_col, value_col

    # fallback: prime due colonne
    return cols[0], cols[1]


def _coerce_series_from_df(df: pd.DataFrame, asset: str) -> pd.Series:
    """
    Converte un DataFrame in Series indicizzata per data.
    Gestisce:
    - header mancanti
    - date in vari formati (dd/mm/yyyy compreso)
    - numeri con virgola decimale
    """
    if df.shape[1] < 2:
        raise ValueError(f"CSV {asset} deve avere almeno 2 colonne (data, valore).")

    # Se sembra senza header (colonne tipo Unnamed o numeriche), rinomina a col0 col1...
    # (non sempre necessario, ma aiuta)
    df = df.copy()
    if any(str(c).lower().startswith("unnamed") for c in df.columns):
        df.columns = [f"col{i}" for i in range(df.shape[1])]

    df = _normalize_columns(df)

    date_col, value_col = _pick_date_value_columns(df)

    out = pd.DataFrame()
    out["date"] = df[date_col]
    out["value"] = df[value_col]

    # Date: dayfirst=True per Italia
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)

    # Value: gestisce virgole decimali, spazi, simboli
    out["value"] = (
        out["value"]
        .astype(str)
        .str.replace("\u00a0", "", regex=False)  # non-breaking space
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)      # toglie separatore migliaia "1.234,56"
        .str.replace(",", ".", regex=False)     # virgola -> punto
    )
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["date", "value"]).sort_values("date")
    if out.empty:
        raise ValueError(
            f"CSV {asset} non contiene righe valide (data/valore). "
            f"Colonne trovate: {list(df.columns)}"
        )

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
        "monthly": "ME",
        "month": "ME",
        "m": "ME",
        "quarterly": "QE",
        "quarter": "QE",
        "q": "QE",
        "yearly": "YE",
        "year": "YE",
        "y": "YE",
    }

    # Se qualcuno passa 'M' maiuscolo
    if freq == "M":
        return "ME"

    return mapping.get(f, "ME")


def _resample_last(s: pd.Series, freq: str) -> pd.Series:
    rule = _freq_to_pandas(freq)
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
    df = _read_csv_robust(path)
    return _coerce_series_from_df(df, asset)


def _series_to_payload(asset: str, s: pd.Series, freq: str) -> dict:
    s = _normalize_100(_resample_last(s, freq))
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


# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------

@app.get("/")
def root():
    static_index = BASE_DIR / "static" / "index.html"
    if static_index.exists():
        return send_from_directory(BASE_DIR / "static", "index.html")
    return "OK - Flask up. Prova /api/data?asset=ls80&freq=monthly oppure /api/combined?freq=monthly", 200


@app.get("/api/health")
def api_health():
    return jsonify({"status": "ok"}), 200


@app.get("/api/data")
def api_data():
    asset = request.args.get("asset", "ls80")
    freq = request.args.get("freq", "monthly")
    try:
        s = _read_asset(asset)
        return jsonify(_series_to_payload(asset, s, freq))
    except Exception as e:
        return jsonify({"error": str(e), "asset": asset, "freq": freq}), 500


@app.get("/api/combined")
def api_combined():
    freq = request.args.get("freq", "monthly")
    try:
        s_ls80 = _read_asset("ls80")
        s_gold = _read_asset("gold")
        s_btc = _read_asset("btc")

        ls80 = _normalize_100(_resample_last(s_ls80, freq))
        gold = _normalize_100(_resample_last(s_gold, freq))
        btc = _normalize_100(_resample_last(s_btc, freq))

        # Allineo sulle date comuni
        df = pd.concat({"ls80": ls80, "gold": gold, "btc": btc}, axis=1).dropna()
        if df.empty:
            raise ValueError("Dopo l’allineamento delle date comuni, non ci sono dati sufficienti.")

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
        return jsonify({"error": str(e), "freq": freq}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
