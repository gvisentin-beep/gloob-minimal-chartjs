from flask import Flask, jsonify, render_template, request
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "ls80": DATA_DIR / "ls80.csv",
    "gold": DATA_DIR / "gold.csv",
    "btc": DATA_DIR / "btc.csv",
}

WEIGHTS = {"ls80": 0.80, "gold": 0.10, "btc": 0.10}


def load_series(asset: str) -> pd.Series:
    """
    Carica un CSV con separatore ';' e prende le prime 2 colonne:
    - data
    - close
    Gestisce date tipo "25/01/2026" e "25.01.2026"
    """
    path = FILES.get(asset)
    if path is None:
        raise ValueError(f"Asset non valido: {asset}")

    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    df = pd.read_csv(path, sep=";", usecols=[0, 1], engine="python")
    df.columns = ["date", "close"]

    # pulizia date
    df["date"] = (
        df["date"]
        .astype(str)
        .str.strip()
        .str.replace(".", "/", regex=False)
    )
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # pulizia prezzi
    df["close"] = (
        df["close"]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["date", "close"]).sort_values("date")
    s = df.set_index("date")["close"].astype(float)
    return s


def downsample(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode:
      - daily (default)
      - weekly
      - monthly
    """
    mode = (mode or "daily").lower()

    if mode == "daily":
        return df
    if mode == "weekly":
        return df.resample("W-FRI").last().dropna()
    if mode == "monthly":
        return df.resample("M").last().dropna()

    return df


@app.route("/")
def index():
    return render_template("index.html")


# ENDPOINT VECCHIO (debug) - lascia pure, comodo
@app.route("/api/data")
def api_data():
    asset = request.args.get("asset", "ls80").lower()
    s = load_series(asset)

    # riduci un po' i punti (mensile) per leggibilità sul browser
    mode = request.args.get("freq", "monthly")
    df = downsample(s.to_frame("v"), mode)

    labels = [d.strftime("%Y-%m") for d in df.index]
    values = df["v"].round(6).tolist()

    return jsonify({"asset": asset, "labels": labels, "values": values})


# NUOVO ENDPOINT: tutto pronto per 2 grafici (B sopra, C sotto)
@app.route("/api/combined")
def api_combined():
    # daily/weekly/monthly
    mode = request.args.get("freq", "daily")

    s_ls80 = load_series("ls80")
    s_gold = load_series("gold")
    s_btc = load_series("btc")

    df = pd.concat(
        {"ls80": s_ls80, "gold": s_gold, "btc": s_btc},
        axis=1,
        join="inner",
    ).dropna()

    df = df.sort_index()
    df = downsample(df, mode)

    # normalizza tutti a 100 alla prima data comune
    base = df.iloc[0]
    df_norm = (df / base) * 100.0

    # portafoglio (B)
    w = pd.Series(WEIGHTS)
    portfolio = df_norm.mul(w, axis=1).sum(axis=1)

    labels = [d.strftime("%Y-%m-%d") for d in df_norm.index]

    return jsonify({
        "labels": labels,
        "portfolio": portfolio.round(6).tolist(),
        "ls80": df_norm["ls80"].round(6).tolist(),
        "gold": df_norm["gold"].round(6).tolist(),
        "btc": df_norm["btc"].round(6).tolist(),
        "base_date": df_norm.index[0].strftime("%Y-%m-%d"),
        "weights": WEIGHTS,
        "freq": mode,
        "points": int(len(labels)),
    })


if __name__ == "__main__":
    app.run(debug=True)
