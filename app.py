from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ASSET_FILES: Dict[str, str] = {
    "ls80": "ls80.csv",
    "gold": "gold.csv",
    "btc": "btc.csv",
}

W_LS80 = 0.80
W_GOLD = 0.15
W_BTC = 0.05


def _detect_and_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", skip_blank_lines=True)
    if df is None or df.empty:
        raise ValueError(f"CSV vuoto o illeggibile: {path.name}")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _pick_date_value_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    date_candidates = {"date", "data", "datetime", "timestamp"}
    value_candidates = {"value", "valore", "close", "prezzo", "price", "adj close", "adj_close", "last"}

    date_col = next((c for c in cols if c in date_candidates), None)
    value_col = next((c for c in cols if c in value_candidates), None)

    if date_col and value_col:
        return date_col, value_col

    if len(cols) < 2:
        raise ValueError("Il CSV deve avere almeno 2 colonne (data, valore).")
    return cols[0], cols[1]


def _to_series(df: pd.DataFrame, asset: str) -> pd.Series:
    df = _normalize_columns(df)
    date_col, value_col = _pick_date_value_columns(df)

    out = pd.DataFrame()
    out["date"] = df[date_col]
    out["value"] = df[value_col]

    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)

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
    s = s[~s.index.duplicated(keep="last")]
    return s


def _freq_to_pandas(freq: str) -> str:
    f = (freq or "monthly").strip().lower()
    return {
        "daily": "D", "d": "D",
        "weekly": "W-FRI", "w": "W-FRI",
        "monthly": "ME", "m": "ME",
        "quarterly": "QE", "q": "QE",
        "yearly": "YE", "y": "YE",
    }.get(f, "ME")


def _resample(s: pd.Series, freq: str) -> pd.Series:
    return s.resample(_freq_to_pandas(freq)).last().dropna()


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
    return {
        "asset": asset,
        "base_date": s.index[0].strftime("%Y-%m-%d") if not s.empty else None,
        "freq": freq,
        "points": len(values),
        "labels": labels,
        "values": values,
    }


@app.get("/")
def index():
    # Homepage: templates/index.html
    return render_template("index.html")


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
        ls80 = _normalize_100(_resample(_read_asset("ls80"), freq))
        gold = _normalize_100(_resample(_read_asset("gold"), freq))
        btc = _normalize_100(_resample(_read_asset("btc"), freq))

        df = pd.concat({"ls80": ls80, "gold": gold, "btc": btc}, axis=1).dropna()
        if df.empty:
            raise ValueError("Dopo allineamento date comuni, non ci sono dati sufficienti.")

        portfolio = (df["ls80"] * W_LS80) + (df["gold"] * W_GOLD) + (df["btc"] * W_BTC)
        labels = [d.strftime("%Y-%m") for d in df.index.to_pydatetime()]

        return jsonify({
            "base_date": df.index[0].strftime("%Y-%m-%d"),
            "freq": freq,
            "labels": labels,
            "points": int(df.shape[0]),
            "series": {
                "benchmark": [round(float(v), 4) for v in df["ls80"].values],
                "portfolio": [round(float(v), 4) for v in portfolio.values],
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
