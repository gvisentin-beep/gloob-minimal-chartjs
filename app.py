from flask import Flask, jsonify, render_template, request
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# --- Percorsi ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "ls80": DATA_DIR / "ls80.csv",
    "gold": DATA_DIR / "gold.csv",
    "btc": DATA_DIR / "btc.csv",
}

# Pesi portafoglio B
WEIGHTS = {
    "ls80": 0.80,
    "gold": 0.10,
    "btc": 0.10,
}

# --- Utils ---
def _read_asset(asset: str) -> pd.Series:
    path = FILES.get(asset)
    if not path or not path.exists():
        raise ValueError(f"Asset non trovato: {asset}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return pd.Series(df["value"].values, index=df["date"])


def _resample(s: pd.Series, freq: str) -> pd.Series:
    if freq == "daily":
        return s
    if freq == "monthly":
        return s.resample("MS").last().dropna()  # FIX pandas
    raise ValueError("freq non valida")


def _normalize_100(s: pd.Series) -> pd.Series:
    base = float(s.iloc[0])
    return (s / base) * 100.0


# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    asset = request.args.get("asset", "ls80")
    freq = request.args.get("freq", "monthly").lower()

    s = _read_asset(asset)
    s = _resample(s, freq)
    s = _normalize_100(s)

    return jsonify({
        "asset": asset,
        "freq": freq,
        "base_date": s.index[0].strftime("%Y-%m-%d"),
        "points": len(s),
        "labels": [d.strftime("%Y-%m") for d in s.index],
        "values": [round(float(v), 2) for v in s.values],
    })


@app.route("/api/combined")
def api_combined():
    freq = request.args.get("freq", "monthly").lower()

    s_ls80 = _normalize_100(_resample(_read_asset("ls80"), freq))
    s_gold = _normalize_100(_resample(_read_asset("gold"), freq))
    s_btc  = _normalize_100(_resample(_read_asset("btc"), freq))

    df = pd.concat(
        {
            "ls80": s_ls80,
            "gold": s_gold,
            "btc": s_btc,
        },
        axis=1
    ).dropna()

    # C = benchmark LS80
    c = df["ls80"]

    # B = portafoglio pesato
    b = (
        df["ls80"] * WEIGHTS["ls80"]
        + df["gold"] * WEIGHTS["gold"]
        + df["btc"]  * WEIGHTS["btc"]
    )

    return jsonify({
        "freq": freq,
        "base_date": df.index[0].strftime("%Y-%m-%d"),
        "labels": [d.strftime("%Y-%m") for d in df.index],
        "B": [round(float(v), 2) for v in b.values],
        "C": [round(float(v), 2) for v in c.values],
    })


if __name__ == "__main__":
    app.run(debug=True)
