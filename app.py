import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ASSETS = {
    "ls80": "ls80.csv",
    "gold": "gold.csv",
    "btc": "btc.csv",
}

# pesi portafoglio
W_LS80 = 0.80
W_GOLD = 0.15
W_BTC = 0.05


# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------

def read_asset(asset: str) -> pd.Series:
    asset = asset.lower()
    if asset not in ASSETS:
        raise ValueError(f"Asset non supportato: {asset}")

    path = DATA_DIR / ASSETS[asset]
    if not path.exists():
        raise FileNotFoundError(f"File mancante: {path}")

    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(f"CSV {asset} deve avere almeno 2 colonne")

    df = df.iloc[:, :2]
    df.columns = ["date", "value"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["value"] = (
        df["value"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna().sort_values("date")

    if df.empty:
        raise ValueError(f"CSV {asset} senza dati validi")

    s = df.set_index("date")["value"]
    s = s[~s.index.duplicated(keep="last")]

    return s


def resample_series(s: pd.Series, freq: str) -> pd.Series:
    freq = (freq or "monthly").lower()

    if freq in ("monthly", "m"):
        rule = "ME"
    elif freq in ("yearly", "y"):
        rule = "YE"
    else:
        rule = "ME"

    return s.resample(rule).last().dropna()


def normalize_100(s: pd.Series) -> pd.Series:
    base = s.iloc[0]
    return (s / base) * 100


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@app.route("/")
def home():
    return "OK – servizio attivo", 200


@app.route("/api/data")
def api_data():
    try:
        asset = request.args.get("asset", "ls80")
        freq = request.args.get("freq", "monthly")

        s = read_asset(asset)
        s = resample_series(s, freq)
        s = normalize_100(s)

        return jsonify({
            "asset": asset,
            "base_date": s.index[0].strftime("%Y-%m-%d"),
            "freq": freq,
            "labels": [d.strftime("%Y-%m") for d in s.index],
            "points": len(s),
            "values": [round(float(v), 4) for v in s.values],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/combined")
def api_combined():
    try:
        freq = request.args.get("freq", "monthly")

        ls80 = normalize_100(resample_series(read_asset("ls80"), freq))
        gold = normalize_100(resample_series(read_asset("gold"), freq))
        btc = normalize_100(resample_series(read_asset("btc"), freq))

        df = pd.concat(
            {"ls80": ls80, "gold": gold, "btc": btc},
            axis=1
        ).dropna()

        if df.empty:
            raise ValueError("Date non allineabili tra gli asset")

        portfolio = (
            df["ls80"] * W_LS80 +
            df["gold"] * W_GOLD +
            df["btc"] * W_BTC
        )

        return jsonify({
            "base_date": df.index[0].strftime("%Y-%m-%d"),
            "freq": freq,
            "labels": [d.strftime("%Y-%m") for d in df.index],
            "points": int(len(df)),
            "series": {
                "benchmark": [round(float(v), 4) for v in df["ls80"].values],
                "portfolio": [round(float(v), 4) for v in portfolio.values],
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
