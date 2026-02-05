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

# Pesi portafoglio (modifica qui se vuoi)
W_LS80 = 0.80
W_GOLD = 0.15
W_BTC = 0.05


# --- Helpers ---------------------------------------------------------------

def _read_text_first_line(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            return (f.readline() or "").strip()
    except Exception:
        return ""


def _detect_and_read_csv(path: Path) -> pd.DataFrame:
    """
    Legge CSV in modo robusto:
    - gestisce BOM
    - gestisce riga iniziale "sep=;"
    - prova separatori comuni se la lettura produce 1 sola colonna
    """
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    first = _read_text_first_line(path).lower()
    skiprows = 1 if first.startswith("sep=") else 0

    # 1) prova autodetect
    try:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
            skip_blank_lines=True,
            skiprows=skiprows,
        )
    except Exception:
        df = pd.DataFrame()

    # Se df è vuoto o ha 1 sola colonna, prova separatori espliciti
    seps_to_try = [";", ",", "\t", "|"]
    if df is None or df.empty or df.shape[1] < 2:
        for sep in seps_to_try:
            try:
                df_try = pd.read_csv(
                    path,
                    sep=sep,
                    engine="python",
                    encoding="utf-8-sig",
                    skip_blank_lines=True,
                    skiprows=skiprows,
                )
                if df_try is not None and (not df_try.empty) and df_try.shape[1] >= 2:
                    df = df_try
                    break
            except Exception:
                continue

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

    if len(cols) < 2:
        raise ValueError("Il CSV deve avere almeno 2 colonne (data, valore).")

    date_candidates = {"date", "data", "datetime", "timestamp"}
    value_candidates = {
        "value", "valore", "close", "prezzo", "price", "adj close", "adj_close", "last"
    }

    date_col: Optional[str] = None
    value_col: Optional[str] = None

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
    return cols[0], cols[1]


def _to_series(df: pd.DataFrame, asset: str) -> pd.Series:
    df = _normalize_columns(df)

    date_col, value_col = _pick_date_value_columns(df)

    out = pd.DataFrame()
    out["date"] = df[date_col]
    out["value"] = df[value_col]

    # Date: robusto per formati italiani
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)

    # Value: gestisce virgole decimali, spazi, ecc.
    out["value"] = (
        out["value"]
        .astype(str)
        .str.replace("\u00a0", "", regex=False)  # NBSP
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["date", "value"]).sort_values("date")
    if out.empty:
        raise ValueError(f"CSV {asset} non contiene righe valide (data/valore).")

    s = out.set_index("date")["value"].astype(float)
    s = s[~s.index.duplicated(keep="last")]
    return s


def _freq_to_pandas(freq: str) -> str:
    """
    Mappa frequenze “umane” a pandas offset.
    Importante: mensile = 'ME' (non 'M').
    """
    f = (freq or "monthly").strip().lower()

    mapping = {
        "daily": "D", "day": "D", "d": "D",
        "weekly": "W-FRI", "week": "W-FRI", "w": "W-FRI",
        "monthly": "ME", "month": "ME", "m": "ME",
        "quarterly": "QE", "quarter": "QE", "q": "QE",
        "yearly": "YE", "year": "YE", "y": "YE",
    }
    return mapping.get(f, "ME")


def _resample(s: pd.Series, freq: str) -> pd.Series:
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
    df = _detect_and_read_csv(path)
    return _to_series(df, asset)


def _series_to_payload(asset: str, s: pd.Series, freq: str) -> dict:
    s = _normalize_100(_resample(s, freq))

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
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
