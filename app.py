from flask import Flask, jsonify, render_template, request
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "ls80": DATA_DIR / "ls80.csv",
    "gold": DATA_DIR / "gold.csv",
    "btc":  DATA_DIR / "btc.csv",
}

# pesi Portafoglio "B"
WEIGHTS = {"ls80": 0.80, "gold": 0.10, "btc": 0.10}


def _pick_date_col(cols):
    cols_l = [c.lower().strip() for c in cols]
    for cand in ["date", "data", "datetime", "time", "timestamp"]:
        if cand in cols_l:
            return cols[cols_l.index(cand)]
    return cols[0]  # fallback


def _pick_value_col(df):
    cols_l = [c.lower().strip() for c in df.columns]
    for cand in ["value", "close", "adj close", "adj_close", "price", "last", "nav"]:
        if cand in cols_l:
            return df.columns[cols_l.index(cand)]
    # fallback: ultima colonna numerica
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        return num_cols[-1]
    return df.columns[-1]


def _read_asset(asset: str) -> pd.Series:
    asset = (asset or "").lower().strip()
    if asset not in FILES:
        raise ValueError(f"Asset non valido: {asset}")

    path = FILES[asset]
    if not path.exists():
        raise FileNotFoundError(f"File mancante: {path}")

    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(f"CSV {asset} deve avere almeno 2 colonne (data, valore)")

    date_col = _pick_date_col(df.columns)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # prova a convertire eventuali numeri con virgola
    val_col = _pick_value_col(df)
    if df[val_col].dtype == object:
        df[val_col] = (
            df[val_col]
            .astype(str)
            .str.replace(".", "", regex=False)   # se ci sono separatori migliaia
            .str.replace(",", ".", regex=False)  # virgola decimale
        )
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])

    s = pd.Series(df[val_col].values, index=df[date_col].values).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _resample(s: pd.Series, freq: str) -> pd.Series:
    """
    freq: daily | monthly | quarterly | yearly
    Usa offset compatibili con pandas recenti: ME, QE, YE.
    """
    f = (freq or "monthly").lower().strip()
    if f == "daily":
        return s.dropna()

    if f == "monthly":
        rule = "ME"   # Month End
    elif f == "quarterly":
        rule = "QE"   # Quarter End
    elif f == "yearly":
        rule = "YE"   # Year End
    else:
        # default
        rule = "ME"

    # usiamo ultimo valore del periodo
    return s.resample(rule).last().dropna()


def _to_base100(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    base = float(s.iloc[0])
    if base == 0:
        return s
    return (s / base) * 100.0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    asset = request.args.get("asset", "ls80").lower()
    freq = request.args.get("freq", "monthly").lower()

    s = _read_asset(asset)
    s = _resample(s, freq)
    s = _to_base100(s)

    labels = [pd.to_datetime(d).strftime("%Y-%m") for d in s.index]
    values = [round(float(v), 2) for v in s.values]

    return jsonify({
        "asset": asset,
        "freq": freq,
        "base_date": pd.to_datetime(s.index[0]).strftime("%Y-%m-%d") if len(s) else None,
        "points": len(values),
        "labels": labels,
        "values": values
    })


@app.route("/api/combined")
def api_combined():
    """
    Ritorna due serie NORMALIZZATE a 100:
    - B: Portafoglio (ls80 80% + gold 10% + btc 10%)
    - C: Benchmark (ls80)
    """
    freq = request.args.get("freq", "monthly").lower()

    s_ls80 = _resample(_read_asset("ls80"), freq)
    s_gold = _resample(_read_asset("gold"), freq)
    s_btc  = _resample(_read_asset("btc"), freq)

    df = pd.concat({"ls80": s_ls80, "gold": s_gold, "btc": s_btc}, axis=1).dropna()
    if df.empty:
        return jsonify({"error": "Dati insufficienti / date non allineate"}), 400

    # Portafoglio B: calcolo su rendimenti relativi (base 1) e poi base 100
    rel = df / df.iloc[0]
    port = (
        rel["ls80"] * WEIGHTS["ls80"] +
        rel["gold"] * WEIGHTS["gold"] +
        rel["btc"]  * WEIGHTS["btc"]
    ) * 100.0

    bench = (df["ls80"] / df["ls80"].iloc[0]) * 100.0

    labels = [pd.to_datetime(d).strftime("%Y-%m") for d in df.index]
    out = {
        "freq": freq,
        "base_date": pd.to_datetime(df.index[0]).strftime("%Y-%m-%d"),
        "points": len(labels),
        "weights": WEIGHTS,
        "labels": labels,
        "portfolio": [round(float(v), 2) for v in port.values],  # B
        "benchmark": [round(float(v), 2) for v in bench.values], # C
    }
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
