from __future__ import annotations

from flask import Flask, jsonify, render_template, request
from pathlib import Path
import pandas as pd

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "ls80": DATA_DIR / "ls80.csv",
    "gold": DATA_DIR / "gold.csv",
    "btc": DATA_DIR / "btc.csv",
}

# Pesi del "Pigro" (modifica qui se vuoi)
WEIGHTS = {"ls80": 0.80, "gold": 0.10, "btc": 0.10}

# Frequenze pandas (robuste su versioni recenti)
FREQ_MAP = {
    "daily": "D",
    "weekly": "W",
    "month": "ME",
    "monthly": "ME",
    "quarter": "QE",
    "quarterly": "QE",
    "year": "YE",
    "yearly": "YE",
}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """
    Legge CSV robustamente:
    - prova autodetect separatore
    - fallback su ';' e ','
    - se finisce con 1 colonna, prova a splittare quella colonna
    """
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    # 1) autodetect separatore (engine python)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=";", engine="python")

    # 2) se ancora 1 colonna, prova fallback espliciti
    if df.shape[1] < 2:
        try:
            df = pd.read_csv(path, sep=";")
        except Exception:
            df = pd.read_csv(path, sep=",")

    # 3) se ancora 1 colonna, prova split manuale
    if df.shape[1] < 2 and df.shape[0] > 0:
        col0 = df.columns[0]
        s = df[col0].astype(str)
        # prova split su ';' oppure ','
        if s.str.contains(";").any():
            parts = s.str.split(";", n=1, expand=True)
            df = pd.DataFrame({"date": parts[0], "value": parts[1]})
        elif s.str.contains(",").any():
            parts = s.str.split(",", n=1, expand=True)
            df = pd.DataFrame({"date": parts[0], "value": parts[1]})

    return df


def _pick_date_value_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Sceglie colonna data e valore in modo tollerante.
    """
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # candidati "data"
    date_candidates = [c for c in cols if c.lower() in ("date", "data", "datetime", "giorno")]
    date_col = date_candidates[0] if date_candidates else cols[0]

    # candidati "valore"
    value_candidates = [c for c in cols if c.lower() in ("value", "valore", "close", "prezzo", "price", "nav")]
    if value_candidates:
        value_col = value_candidates[0]
    else:
        # cerca prima colonna numerica diversa dalla data
        other_cols = [c for c in cols if c != date_col]
        value_col = other_cols[0] if other_cols else cols[-1]

    return date_col, value_col


def _read_asset(asset: str) -> pd.Series:
    if asset not in FILES:
        raise ValueError(f"Asset non valido: {asset}. Usa: {', '.join(FILES.keys())}")

    df = _safe_read_csv(FILES[asset])

    if df.shape[1] < 2:
        raise ValueError(f"CSV {asset} deve avere almeno 2 colonne (data, valore)")

    date_col, value_col = _pick_date_value_columns(df)

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "value"]

    # pulizia value: virgole, spazi, ecc.
    out["value"] = (
        out["value"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    # parsing date (in Italia spesso day-first)
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)

    out = out.dropna(subset=["date", "value"]).sort_values("date")
    if out.empty:
        raise ValueError(f"CSV {asset}: non trovo righe valide (data + valore)")

    s = out.set_index("date")["value"]
    # se ci sono date duplicate, prendi l’ultima
    s = s[~s.index.duplicated(keep="last")]
    return s


def _resample(s: pd.Series, freq: str) -> pd.Series:
    f = FREQ_MAP.get(freq.lower(), "ME")

    # daily: non resamplare, ma normalizza (tieni l’ultima per giorno se serve)
    if f == "D":
        return s.resample("D").last().dropna()

    # per ME/QE/YE ecc: prendi ultimo valore del periodo
    return s.resample(f).last().dropna()


def _normalize_100(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    base = float(s.iloc[0])
    if base == 0:
        return s * 0.0
    return (s / base) * 100.0


@app.route("/")
def home():
    # se hai templates/index.html va bene così
    try:
        return render_template("index.html")
    except Exception:
        return "OK - service running"


@app.route("/api/data")
def api_data():
    asset = request.args.get("asset", "ls80").lower()
    freq = request.args.get("freq", "monthly").lower()

    try:
        s = _resample(_read_asset(asset), freq)
        idx = _normalize_100(s)
        labels = [d.strftime("%Y-%m") for d in idx.index]
        values = [round(float(v), 2) for v in idx.values]
        return jsonify(
            {
                "asset": asset,
                "freq": freq,
                "base_date": idx.index[0].strftime("%Y-%m-%d") if len(idx) else None,
                "points": len(values),
                "labels": labels,
                "values": values,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/combined")
def api_combined():
    """
    Ritorna due serie NORMALIZZATE a 100:
    - "portfolio": Portafoglio (LS80 80% + Gold 10% + BTC 10%)
    - "benchmark": LS80 (benchmark semplice)
    """
    freq = request.args.get("freq", "monthly").lower()

    try:
        s_ls80 = _resample(_read_asset("ls80"), freq)
        s_gold = _resample(_read_asset("gold"), freq)
        s_btc = _resample(_read_asset("btc"), freq)

        df = pd.concat(
            {"ls80": s_ls80, "gold": s_gold, "btc": s_btc},
            axis=1,
        ).dropna()

        if df.empty:
            return jsonify({"error": "Dati insufficienti (date comuni vuote)"}), 500

        # portfolio: somma pesata dei livelli normalizzati (base 100)
        ls80_n = _normalize_100(df["ls80"])
        gold_n = _normalize_100(df["gold"])
        btc_n = _normalize_100(df["btc"])

        portfolio = (
            ls80_n * WEIGHTS["ls80"]
            + gold_n * WEIGHTS["gold"]
            + btc_n * WEIGHTS["btc"]
        )

        labels = [d.strftime("%Y-%m") for d in df.index]
        out = {
            "base_date": df.index[0].strftime("%Y-%m-%d"),
            "freq": freq,
            "points": len(df.index),
            "labels": labels,
            "series": {
                "benchmark": [round(float(v), 2) for v in ls80_n.values],
                "portfolio": [round(float(v), 2) for v in portfolio.values],
            },
        }
        return jsonify(out)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
