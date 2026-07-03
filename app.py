# ============================================================
# Signal System V33 — Streamlit Web App
#   - All original logic preserved
#   - UI: ETF selection, volume threshold, run button
#   - Displays comparison table, ETF charts, constituent charts
#   - Downloadable PDF and CSV reports
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import datetime
import pickle
import os
import re
import requests
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import time

# ---------------- Settings ----------------
st.set_page_config(page_title="Sector Signal V33", layout="wide")
st.title("📊 SPDR Sector ETF Signal System V33")

DATA_PERIOD = "6mo"
BREAKOUT_THRESHOLD_PCT = 0.0
BIN_WIDTH_PCT = 0.5
MIN_BINS = 30
MAX_BINS = 100
OBV_WINDOWS = [5, 10, 20]
OBV_WEIGHTS = [0.5, 0.3, 0.2]
OBV_DIVERGENCE_LOOKBACK = 20
VOLUME_LOOKBACK = 60

SECTOR_ETF = [
    "XLK", "XLV", "XLF", "XLE", "XLY",
    "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"
]

TICKER_MAP = {"BRK.B": "BRK-B", "BF.B": "BF-B"}
SPECIAL_TICKER_MAP = {
    "2670549D": "XOM",
}
HOLDINGS_CACHE = "holdings_cache.pkl"
ANALYSIS_CACHE = "analysis_cache.pkl"

# ---------------- Helper functions (exactly as V33) ----------------
def malaysia_now():
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    return utc_now + datetime.timedelta(hours=8)

def get_spdr_url(etf_ticker: str) -> str:
    return (f"https://www.ssga.com/us/en/intermediary/library-content/"
            f"products/fund-data/etfs/us/holdings-daily-us-en-{etf_ticker.lower()}.xlsx")

def download_and_parse_holdings(etf_ticker: str):
    url = get_spdr_url(etf_ticker)
    tmp_file = None
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        tmp_file.write(resp.content)
        tmp_file.close()
        df_raw = pd.read_excel(tmp_file.name, sheet_name=0, header=None, dtype=str)
        date_str = None
        for idx, row in df_raw.iterrows():
            full = ' '.join([str(v) for v in row if pd.notna(v)])
            match = re.search(r'As of (\d{1,2}-[A-Za-z]{3}-\d{4})', full)
            if match:
                date_str = match.group(1)
                break
        if date_str is None:
            date_str = "Unknown"
        header_row = None
        for idx, row in df_raw.iterrows():
            if any(str(v).strip().lower() == 'ticker' for v in row if pd.notna(v)):
                header_row = idx
                break
        if header_row is None:
            raise ValueError("Could not find 'Ticker' header")
        df = df_raw.iloc[header_row+1:].copy()
        df.columns = [str(col).strip().lower() for col in df_raw.iloc[header_row]]
        df['ticker'] = df['ticker'].astype(str).str.strip()
        df['ticker'] = df['ticker'].replace(SPECIAL_TICKER_MAP)
        df = df[df['ticker'] != '']
        df = df[df['ticker'] != '-']
        df = df[~df['ticker'].str.contains(r'\d')]
        if 'sedol' in df.columns:
            df = df[df['sedol'].astype(str).str.strip() != '-']
        name_col = 'name'
        ticker_col = 'ticker'
        weight_col = next((c for c in df.columns if 'weight' in c and '%' not in c), None)
        if weight_col is None:
            weight_col = next((c for c in df.columns if 'weight' in c), None)
        if name_col not in df.columns or weight_col is None:
            raise ValueError(f"Missing columns: {df.columns.tolist()}")
        df = df[[name_col, ticker_col, weight_col]].copy()
        df[weight_col] = df[weight_col].astype(str).str.replace('%', '').str.strip()
        df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
        df = df.dropna(subset=[weight_col])
        if df[weight_col].max() < 1.0:
            df[weight_col] *= 100.0
        df[ticker_col] = df[ticker_col].replace(TICKER_MAP)
        holdings = dict(zip(df[ticker_col], df[weight_col]))
        os.unlink(tmp_file.name)
        return holdings, date_str
    except Exception as e:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
        raise RuntimeError(f"Download/parse failed for {etf_ticker}: {e}")

def normalize_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required) and len(df.columns) == 5:
        df.columns = required
    return df

def load_data(ticker, period=DATA_PERIOD):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = normalize_columns(df)
    return df.dropna()

def compute_volume_profile_params(prices, vol):
    avg_price = np.mean(prices)
    price_range = prices.max() - prices.min()
    bin_width_price = avg_price * (BIN_WIDTH_PCT / 100.0)
    bins = int(np.round(price_range / bin_width_price)) if bin_width_price != 0 else MIN_BINS
    bins = np.clip(bins, MIN_BINS, MAX_BINS)
    hist, edges = np.histogram(prices, bins=bins, weights=vol)
    idx_poc = np.argmax(hist)
    poc = (edges[idx_poc] + edges[idx_poc + 1]) / 2
    above_mask = np.arange(len(hist)) > idx_poc
    below_mask = np.arange(len(hist)) < idx_poc
    above_hist, below_hist = hist[above_mask], hist[below_mask]
    above_edges, below_edges = edges[1:][above_mask], edges[:-1][below_mask]
    if len(above_hist) > 0:
        above_sorted_idx = np.argsort(above_hist)[::-1]
    else:
        above_sorted_idx = []
    if len(below_hist) > 0:
        below_sorted_idx = np.argsort(below_hist)[::-1]
    else:
        below_sorted_idx = []
    num_above, num_below = len(above_hist), len(below_hist)
    if num_above >= 2 and num_below >= 2:
        N_up, N_down = 2, 2
    elif num_above == 1:
        N_up, N_down = 1, min(3, num_below)
    elif num_above == 0:
        N_up, N_down = 0, min(4, num_below)
    elif num_below == 1:
        N_up, N_down = min(3, num_above), 1
    elif num_below == 0:
        N_up, N_down = min(4, num_above), 0
    else:
        N_up, N_down = 0, 0
    vah = poc if N_up == 0 else np.max(above_edges[above_sorted_idx[:N_up]])
    val = poc if N_down == 0 else np.min(below_edges[below_sorted_idx[:N_down]])
    if num_above == 0 and num_below == 0:
        vah = val = poc
    return poc, vah, val, bins, edges, hist

def volume_profile_v6(df):
    prices = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"]
    poc, vah, val, _, _, _ = compute_volume_profile_params(prices, vol)
    last = df["Close"].iloc[-1]
    if last >= vah:
        vp_score = 1.0
    elif last <= val:
        vp_score = -1.0
    else:
        mid = (vah + val) / 2
        half = (vah - val) / 2
        vp_score = (last - mid) / half if half > 0 else 0.0
    return poc, vah, val, np.clip(vp_score, -1.0, 1.0)

def obv_v7(df, windows=OBV_WINDOWS, weights=OBV_WEIGHTS, divergence_lookback=OBV_DIVERGENCE_LOOKBACK):
    diff = df["Close"].diff()
    obv = np.cumsum(np.where(diff > 0, df["Volume"], np.where(diff < 0, -df["Volume"], 0)))
    if len(obv) < max(windows):
        return 0.0, obv, [None]*len(windows)
    def standardized_slope(series, window):
        y = series[-window:]
        x = np.arange(window)
        slope = np.polyfit(x, y, 1)[0]
        std = np.std(y)
        return slope / std if std > 0 else 0.0
    slopes = [standardized_slope(obv, w) for w in windows]
    raw_score = np.sum([w * s for w, s in zip(weights, slopes)])
    obv_trend_score = np.clip(raw_score, -1.0, 1.0)
    price = df["Close"].values
    obv_arr = obv
    lookback = divergence_lookback
    price_high_idx = np.argmax(price[-lookback:])
    obv_high_idx = np.argmax(obv_arr[-lookback:])
    bearish = (price[-lookback + price_high_idx] > np.max(price[-lookback-1:-lookback])) and \
              (obv_arr[-lookback + obv_high_idx] < np.max(obv_arr[-lookback-1:-lookback]))
    price_low_idx = np.argmin(price[-lookback:])
    obv_low_idx = np.argmin(obv_arr[-lookback:])
    bullish = (price[-lookback + price_low_idx] < np.min(price[-lookback-1:-lookback])) and \
              (obv_arr[-lookback + obv_low_idx] > np.min(obv_arr[-lookback-1:-lookback]))
    div_adj = -0.3 if bearish else 0.3 if bullish else 0.0
    obv_score = np.clip(obv_trend_score + div_adj, -1.0, 1.0)
    return float(obv_score), obv, slopes

def generate_signal_v8(df, ticker):
    poc, vah, val, vp_score = volume_profile_v6(df)
    obv_score, obv, slopes = obv_v7(df)
    last = df["Close"].iloc[-1]
    pct_above = ((last - vah) / vah * 100) if last > vah else 0.0
    pct_below = ((val - last) / val * 100) if last < val else 0.0
    if vp_score > 0.7 and obv_score > 0:
        action, conf, reason = "LATE_BREAKOUT_BUY", 0.70, "Strong uptrend, late stage breakout"
    elif vp_score > 0.5 and obv_score > 0.3:
        action, conf, reason = "EARLY_BREAKOUT_BUY", 0.85, "Early breakout with accumulation"
    elif vp_score > 0.5 and obv_score < -0.3:
        action, conf, reason = "EARLY_BREAKOUT_SELL", 0.85, "High price with distribution, early sell signal"
    elif vp_score < -0.7 and obv_score < 0:
        action, conf, reason = "LATE_BREAKOUT_SELL", 0.70, "Strong downtrend, panic selling"
    elif vp_score < -0.2 and obv_score > 0:
        action, conf, reason = "BUY", 0.65, "Value zone accumulation (mean reversion)"
    elif vp_score > 0.2 and obv_score < 0:
        action, conf, reason = "SELL", 0.65, "Overvalued area with selling pressure"
    else:
        action, conf, reason = "HOLD", 0.50, "No clear alignment, wait"
    signal = 1 if "BUY" in action and "SELL" not in action else (-1 if "SELL" in action else 0)
    return {
        "ticker": ticker, "action": action, "signal": signal, "confidence": conf,
        "vp_score": vp_score, "obv_score": obv_score, "POC": poc, "VAH": vah, "VAL": val,
        "Pct_Above_VAH": pct_above, "Pct_Below_VAL": pct_below,
        "reason": reason, "asset_type": "Stock"
    }

def volume_profile_plot_data(df):
    prices = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"]
    poc, vah, val, bins, edges, hist = compute_volume_profile_params(prices, vol)
    return {"prices": prices, "hist": hist, "edges": edges, "poc": poc, "vah": vah, "val": val,
            "last_price": df["Close"].iloc[-1], "bins": bins}

def obv_plot_data(df):
    diff = df["Close"].diff()
    obv = np.cumsum(np.where(diff > 0, df["Volume"], np.where(diff < 0, -df["Volume"], 0)))
    trends = {}
    for w in OBV_WINDOWS:
        if len(obv) >= w:
            y = obv[-w:]
            slope, inter = np.polyfit(np.arange(w), y, 1)
            trends[w] = {"line": slope * np.arange(w) + inter, "slope": slope}
        else:
            trends[w] = None
    return {"obv": obv, "trends": trends}

def plot_ticker_analysis(ticker, df, vp_data, obv_data, asset_type, zone_status,
                         pct_above=0.0, pct_below=0.0, avg_vol=None, weight=None):
    title = f"{ticker}  |  {asset_type} "
    if "Above" in zone_status and pct_above > 0:
        title += f"Above VAH {pct_above:.2f}%"
    elif "Below" in zone_status and pct_below > 0:
        title += f"Below VAL {pct_below:.2f}%"
    else:
        title += f"{zone_status}"
    if avg_vol is not None:
        title += f" (avg vol {format_volume(avg_vol)})"
    if weight is not None:
        title += f"  weight: {weight:.2f}%"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    edges, hist = vp_data["edges"], vp_data["hist"]
    poc, vah, val, last = vp_data["poc"], vp_data["vah"], vp_data["val"], vp_data["last_price"]
    bins = vp_data.get("bins", len(hist))
    centers = (edges[:-1] + edges[1:]) / 2
    bh = edges[1] - edges[0]
    inside = (centers >= val) & (centers <= vah)
    colors = np.where(inside, 'coral', 'steelblue')
    ax1.barh(centers, hist, height=bh*0.9, color=colors, edgecolor='white', alpha=0.9)
    ax1.axhspan(val, vah, color='coral', alpha=0.15, label='Value Area')
    ax1.set_xlabel("Volume")
    ax1.set_ylabel("Price")
    ax1.set_title(f"Volume Profile (bins={bins})")
    ax1.axhline(poc, color='gold', ls='-', lw=2, label=f'POC {poc:.2f}')
    ax1.axhline(vah, color='red', ls='--', lw=1, label=f'VAH {vah:.2f}')
    ax1.axhline(val, color='green', ls='--', lw=1, label=f'VAL {val:.2f}')
    ax1.axhline(last, color='black', ls='-', lw=1.5, label=f'Last {last:.2f}')
    ax1.legend(loc='upper right')
    ymin, ymax = edges[0], edges[-1]
    ax1.set_ylim(ymin - (ymax-ymin)*0.05, ymax + (ymax-ymin)*0.05)
    obv = obv_data["obv"]
    dates = df.index[-len(obv):]
    ax2.plot(dates, obv, color='navy', lw=1, label='OBV')
    line_styles = {5: ('orange', 'dotted'), 10: ('red', 'dashed'), 20: ('darkred', 'dashdot')}
    for w in OBV_WINDOWS:
        trend_info = obv_data["trends"].get(w)
        if trend_info is not None:
            td = dates[-w:]
            ax2.plot(td, trend_info["line"], color=line_styles[w][0],
                     linestyle=line_styles[w][1], lw=1.5, label=f'{w}d Trend')
    ax2.set_title("OBV (5/10/20 composite) with Multi‑period Trends")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("OBV")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def format_volume(vol):
    if vol >= 1e6:
        val = vol / 1e6
        if abs(val - round(val)) < 0.05:
            return f"{int(round(val))}M"
        return f"{val:.1f}M"
    val = vol / 1e3
    return f"{val:.0f}K" if val >= 10 else f"{val:.1f}K"

def create_single_summary_page(zone_stats, etf_ticker, holding_date=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    total_weight = zone_stats['Above VAH']['weight'] + zone_stats['Below VAL']['weight'] + zone_stats['Inside VA']['weight']
    total_count = zone_stats['Above VAH']['count'] + zone_stats['Below VAL']['count'] + zone_stats['Inside VA']['count']
    data = [
        ['Above VAH', zone_stats['Above VAH']['count'], f"{zone_stats['Above VAH']['weight']:.2f}%"],
        ['Below VAL', zone_stats['Below VAL']['count'], f"{zone_stats['Below VAL']['weight']:.2f}%"],
        ['Inside VA / Others', zone_stats['Inside VA']['count'], f"{zone_stats['Inside VA']['weight']:.2f}%"],
        ['Total', total_count, f"{total_weight:.2f}%"]
    ]
    table = ax.table(cellText=data,
                     colLabels=['Zone', 'Count', 'Total Weighting %'],
                     loc='center', cellLoc='center', colColours=['#f0f0f0']*3)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    title = f"ETF Constituent Volume Profile Zone Statistics ({etf_ticker})"
    if holding_date:
        title += f"\nHoldings as of {holding_date}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_sector_comparison_table(etf_zone_stats, etf_self_data, include_constituent=True, holding_dates=None):
    rows = []
    for etf in SECTOR_ETF:
        stats = etf_zone_stats.get(etf, {
            'Above VAH': {'count':0,'weight':0.0},
            'Below VAL': {'count':0,'weight':0.0},
            'Inside VA': {'count':0,'weight':0.0}
        })
        total_count = stats['Above VAH']['count'] + stats['Below VAL']['count'] + stats['Inside VA']['count']
        total_weight = stats['Above VAH']['weight'] + stats['Below VAL']['weight'] + stats['Inside VA']['weight']
        self_data = etf_self_data.get(etf, {'pct_above':0.0, 'pct_below':0.0})
        pct_above = self_data['pct_above']
        pct_below = self_data['pct_below']
        row = [etf,
               stats['Above VAH']['count'], f"{stats['Above VAH']['weight']:.2f}%",
               stats['Below VAL']['count'], f"{stats['Below VAL']['weight']:.2f}%",
               stats['Inside VA']['count'], f"{stats['Inside VA']['weight']:.2f}%",
               total_count, f"{total_weight:.2f}%",
               f"{pct_above:.2f}%" if pct_above > 0 else "0.00%",
               f"{pct_below:.2f}%" if pct_below > 0 else "0.00%"]
        rows.append(row)
    above_weights = [float(r[2].rstrip('%')) for r in rows]
    below_weights = [float(r[4].rstrip('%')) for r in rows]
    top3_above_idx = set(np.argsort(above_weights)[::-1][:3].tolist())
    top3_below_idx = set(np.argsort(below_weights)[::-1][:3].tolist())
    for i, row in enumerate(rows):
        signal = ""
        if i in top3_above_idx:
            pct_above_val = float(row[9].rstrip('%'))
            if pct_above_val > 0:
                signal = "Long"
            else:
                signal = "Wait"
        elif i in top3_below_idx:
            pct_below_val = float(row[10].rstrip('%'))
            if pct_below_val > 0:
                signal = "Short"
            else:
                signal = "Wait"
        row.append(signal)
    if include_constituent:
        total_row = ["Total"]
        sum_above_cnt = sum(r[1] for r in rows)
        sum_below_cnt = sum(r[3] for r in rows)
        sum_inside_cnt = sum(r[5] for r in rows)
        sum_total_cnt = sum(r[7] for r in rows)
        total_row.extend([sum_above_cnt, "", sum_below_cnt, "", sum_inside_cnt, "", sum_total_cnt, "", "", "", ""])
        rows.append(total_row)
    col_labels = ['ETF', 'Above VAH\nCount', 'Above VAH\nWeight',
                  'Below VAL\nCount', 'Below VAL\nWeight',
                  'Inside VA\nCount', 'Inside VA\nWeight',
                  'Total\nCount', 'Total\nWeight',
                  '% Above VAH', '% Below VAL', 'Signal']
    fig, ax = plt.subplots(figsize=(18, len(rows)*0.6 + 1))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)
    for i, row in enumerate(rows[:-1]):
        for col_idx in [9, 10]:
            cell = table[i+1, col_idx]
            val_str = cell.get_text().get_text().rstrip('%')
            try:
                val = float(val_str)
                if val > 0:
                    cell.get_text().set_color('blue')
                    cell.get_text().set_fontweight('bold')
            except:
                pass
    if include_constituent:
        weight_cols = [2, 4, 6]
        for col in weight_cols:
            values = [float(r[col].rstrip('%')) for r in rows[:-1]]
            top_indices = np.argsort(values)[::-1][:3]
            for idx in top_indices:
                cell = table[idx+1, col]
                cell.get_text().set_color('red')
                cell.get_text().set_fontweight('bold')
    signal_col = 11
    for i, row in enumerate(rows[:-1]):
        signal = row[signal_col]
        if signal:
            cell = table[i+1, signal_col]
            if signal == "Long":
                cell.get_text().set_color('darkgreen')
            elif signal == "Short":
                cell.get_text().set_color('darkred')
            elif signal == "Wait":
                cell.get_text().set_color('gray')
            cell.get_text().set_fontweight('bold')
    title = "SPDR Sector ETFs — Comparison (Constituent Zone + Self Position)"
    if holding_dates:
        unique_dates = sorted(set(holding_dates.values()))
        if unique_dates:
            title += f"\nHoldings as of {unique_dates[0]}"
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def add_etf_charts(pdf, selected_etfs):
    for etf in selected_etfs:
        df = load_data(etf)
        if df.empty or len(df) < max(120, VOLUME_LOOKBACK):
            continue
        vp_data = volume_profile_plot_data(df)
        obv_data = obv_plot_data(df)
        poc, vah, val, _ = volume_profile_v6(df)
        last = df["Close"].iloc[-1]
        if last > vah:
            zone = 'Above VAH'
            pct_above = (last - vah) / vah * 100
            pct_below = 0.0
        elif last < val:
            zone = 'Below VAL'
            pct_above = 0.0
            pct_below = (val - last) / val * 100
        else:
            zone = 'Inside VA'
            pct_above = 0.0
            pct_below = 0.0
        avg_vol = df["Volume"].tail(VOLUME_LOOKBACK).mean()
        fig = plot_ticker_analysis(etf, df, vp_data, obv_data, "Sector ETF", zone,
                                   pct_above=pct_above, pct_below=pct_below,
                                   avg_vol=avg_vol, weight=None)
        pdf.savefig(fig)
        plt.close(fig)

# ---------------- Cached data loading ----------------
@st.cache_data
def get_holdings():
    """Download and cache SPDR holdings for all sector ETFs."""
    etf_to_holdings = {}
    holding_dates = {}
    for etf in SECTOR_ETF:
        try:
            holdings, date_str = download_and_parse_holdings(etf)
            etf_to_holdings[etf] = holdings
            holding_dates[etf] = date_str
        except Exception as e:
            st.error(f"Failed to download {etf}: {e}")
            raise
    return etf_to_holdings, holding_dates

@st.cache_data
def run_full_analysis(etf_to_holdings, min_avg_volume):
    """Run the full analysis on all constituents."""
    all_signals = {}
    etf_zone_stats_all = {etf: {'Above VAH': {'count':0,'weight':0.0},
                                 'Below VAL': {'count':0,'weight':0.0},
                                 'Inside VA': {'count':0,'weight':0.0}}
                          for etf in SECTOR_ETF}
    etf_self_data = {}
    # ETF self data
    for etf in SECTOR_ETF:
        df = load_data(etf)
        if df.empty or len(df) < max(120, VOLUME_LOOKBACK):
            continue
        poc, vah, val, _ = volume_profile_v6(df)
        last = df["Close"].iloc[-1]
        pct_above = ((last - vah) / vah * 100) if last > vah else 0.0
        pct_below = ((val - last) / val * 100) if last < val else 0.0
        etf_self_data[etf] = {'last': last, 'vah': vah, 'val': val,
                              'pct_above': pct_above, 'pct_below': pct_below}
    all_tickers = set()
    for etf in SECTOR_ETF:
        all_tickers.update(etf_to_holdings[etf].keys())
    total_tickers = len(all_tickers)
    progress_bar = st.progress(0, text="Processing constituents...")
    for idx, ticker in enumerate(all_tickers):
        try:
            df = load_data(ticker)
            if df.empty or len(df) < max(120, VOLUME_LOOKBACK):
                progress_bar.progress((idx+1)/total_tickers)
                continue
            avg_vol = df["Volume"].tail(VOLUME_LOOKBACK).mean()
            res = generate_signal_v8(df, ticker)
            last = df["Close"].iloc[-1]
            vah, val = res["VAH"], res["VAL"]
            zone = 'Above VAH' if last > vah else ('Below VAL' if last < val else 'Inside VA')
            for etf in SECTOR_ETF:
                if ticker in etf_to_holdings[etf]:
                    w = etf_to_holdings[etf][ticker]
                    etf_zone_stats_all[etf][zone]['count'] += 1
                    etf_zone_stats_all[etf][zone]['weight'] += w
            all_signals[ticker] = {
                'signal': res,
                'avg_vol': avg_vol,
                'last': last,
                'zone': zone
            }
        except Exception as e:
            # skip problematic tickers silently
            pass
        progress_bar.progress((idx+1)/total_tickers)
    progress_bar.empty()
    return all_signals, etf_zone_stats_all, etf_self_data

# ---------------- UI ----------------
st.sidebar.header("Settings")
all_etfs = SECTOR_ETF
selected_etfs = st.sidebar.multiselect("Select Sector ETFs to analyze", all_etfs, default=all_etfs[:3])
vol_thresholds = [0.01e6, 0.5e6, 1e6, 2e6, 5e6, 10e6]
min_vol = st.sidebar.radio(
    "Minimum average volume threshold",
    options=vol_thresholds,
    format_func=lambda x: f"{x/1e6:.2f}M",
    index=3
)
run_button = st.sidebar.button("🚀 Run Analysis", type="primary")

# ---------------- Main logic ----------------
if run_button:
    if not selected_etfs:
        st.warning("Please select at least one ETF.")
    else:
        with st.spinner("Loading holdings data..."):
            etf_to_holdings, holding_dates = get_holdings()
        with st.spinner("Running analysis (this may take a few minutes)..."):
            all_signals, etf_zone_stats_all, etf_self_data = run_full_analysis(etf_to_holdings, min_vol)

        # Build results for selected ETFs
        selected_ticker_weight = {}
        for etf in selected_etfs:
            for t, w in etf_to_holdings[etf].items():
                selected_ticker_weight[t] = selected_ticker_weight.get(t, 0.0) + w

        results = []
        zone_entries = []
        breakout_above = []
        breakout_below = []

        for ticker, data in all_signals.items():
            if ticker not in selected_ticker_weight:
                continue
            if data['avg_vol'] < min_vol:
                continue
            res = data['signal']
            last = data['last']
            zone = data['zone']
            weight = selected_ticker_weight[ticker]
            results.append(res)
            zone_entries.append({
                "ticker": ticker, "asset_type": "Stock", "Last": last,
                "VAH": res["VAH"], "VAL": res["VAL"],
                "Pct_Above_VAH": res["Pct_Above_VAH"],
                "Pct_Below_VAL": res["Pct_Below_VAL"],
                "Weight": weight, "Avg_Vol": data['avg_vol'],
                "zone_status": zone
            })
            pct_above = res["Pct_Above_VAH"]
            pct_below = res["Pct_Below_VAL"]
            if last > res["VAH"] and pct_above > BREAKOUT_THRESHOLD_PCT:
                breakout_above.append((pct_above, ticker, None, res, zone, data['avg_vol'], weight))
            elif last < res["VAL"] and pct_below > BREAKOUT_THRESHOLD_PCT:
                breakout_below.append((pct_below, ticker, None, res, zone, data['avg_vol'], weight))

        breakout_above.sort(key=lambda x: x[0], reverse=True)
        breakout_below.sort(key=lambda x: x[0], reverse=True)

        # Load price data for breakout charts
        for idx, (pct, ticker, _, res, zone, avg_vol, weight) in enumerate(breakout_above):
            df = load_data(ticker)
            breakout_above[idx] = (pct, ticker, df, res, zone, avg_vol, weight)
        for idx, (pct, ticker, _, res, zone, avg_vol, weight) in enumerate(breakout_below):
            df = load_data(ticker)
            breakout_below[idx] = (pct, ticker, df, res, zone, avg_vol, weight)

        # ---------- Display Results ----------
        st.success("Analysis complete!")

        # 1. Comparison Table
        st.subheader("📊 Sector Comparison Table")
        comp_fig = create_sector_comparison_table(etf_zone_stats_all, etf_self_data, True, holding_dates)
        st.pyplot(comp_fig)
        plt.close(comp_fig)

        # 2. ETF Charts
        st.subheader("📈 ETF Charts")
        for etf in selected_etfs:
            df = load_data(etf)
            if df.empty or len(df) < max(120, VOLUME_LOOKBACK):
                continue
            vp_data = volume_profile_plot_data(df)
            obv_data = obv_plot_data(df)
            poc, vah, val, _ = volume_profile_v6(df)
            last = df["Close"].iloc[-1]
            if last > vah:
                zone = 'Above VAH'
                pct_above = (last - vah) / vah * 100
                pct_below = 0.0
            elif last < val:
                zone = 'Below VAL'
                pct_above = 0.0
                pct_below = (val - last) / val * 100
            else:
                zone = 'Inside VA'
                pct_above = 0.0
                pct_below = 0.0
            avg_vol = df["Volume"].tail(VOLUME_LOOKBACK).mean()
            fig = plot_ticker_analysis(etf, df, vp_data, obv_data, "Sector ETF", zone,
                                       pct_above=pct_above, pct_below=pct_below,
                                       avg_vol=avg_vol, weight=None)
            st.pyplot(fig)
            plt.close(fig)

        # 3. Constituent Summary Pages
        st.subheader("📋 ETF Constituent Zone Summaries")
        for etf in selected_etfs:
            fig = create_single_summary_page(etf_zone_stats_all[etf], etf, holding_dates.get(etf))
            st.pyplot(fig)
            plt.close(fig)

        # 4. Breakout Charts (if any)
        if breakout_above or breakout_below:
            st.subheader("🔍 Breakout Charts")
            for pct, ticker, df, res, zone, avg_vol, weight in breakout_above + breakout_below:
                if df is None:
                    continue
                vp_data = volume_profile_plot_data(df)
                obv_data = obv_plot_data(df)
                if zone == 'Above VAH':
                    fig = plot_ticker_analysis(ticker, df, vp_data, obv_data, "Stock", zone,
                                               pct_above=pct, pct_below=0, avg_vol=avg_vol, weight=weight)
                else:
                    fig = plot_ticker_analysis(ticker, df, vp_data, obv_data, "Stock", zone,
                                               pct_above=0, pct_below=pct, avg_vol=avg_vol, weight=weight)
                st.pyplot(fig)
                plt.close(fig)

        # 5. Download buttons
        st.subheader("📥 Download Reports")

        # Prepare PDF in memory
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            # Comparison table
            comp_fig = create_sector_comparison_table(etf_zone_stats_all, etf_self_data, True, holding_dates)
            pdf.savefig(comp_fig)
            plt.close(comp_fig)
            # ETF charts
            for etf in selected_etfs:
                df = load_data(etf)
                if df.empty or len(df) < max(120, VOLUME_LOOKBACK):
                    continue
                vp_data = volume_profile_plot_data(df)
                obv_data = obv_plot_data(df)
                poc, vah, val, _ = volume_profile_v6(df)
                last = df["Close"].iloc[-1]
                if last > vah:
                    zone = 'Above VAH'
                    pct_above = (last - vah) / vah * 100
                    pct_below = 0.0
                elif last < val:
                    zone = 'Below VAL'
                    pct_above = 0.0
                    pct_below = (val - last) / val * 100
                else:
                    zone = 'Inside VA'
                    pct_above = 0.0
                    pct_below = 0.0
                avg_vol = df["Volume"].tail(VOLUME_LOOKBACK).mean()
                fig = plot_ticker_analysis(etf, df, vp_data, obv_data, "Sector ETF", zone,
                                           pct_above=pct_above, pct_below=pct_below,
                                           avg_vol=avg_vol, weight=None)
                pdf.savefig(fig)
                plt.close(fig)
            # Constituent summaries
            for etf in selected_etfs:
                fig = create_single_summary_page(etf_zone_stats_all[etf], etf, holding_dates.get(etf))
                pdf.savefig(fig)
                plt.close(fig)
            # Breakouts
            for pct, ticker, df, res, zone, avg_vol, weight in breakout_above + breakout_below:
                if df is None:
                    continue
                vp_data = volume_profile_plot_data(df)
                obv_data = obv_plot_data(df)
                if zone == 'Above VAH':
                    fig = plot_ticker_analysis(ticker, df, vp_data, obv_data, "Stock", zone,
                                               pct_above=pct, pct_below=0, avg_vol=avg_vol, weight=weight)
                else:
                    fig = plot_ticker_analysis(ticker, df, vp_data, obv_data, "Stock", zone,
                                               pct_above=0, pct_below=pct, avg_vol=avg_vol, weight=weight)
                pdf.savefig(fig)
                plt.close(fig)
        pdf_buffer.seek(0)
        st.download_button(
            label="📄 Download Full PDF Report",
            data=pdf_buffer,
            file_name=f"sector_signal_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )

        # CSV downloads
        if results:
            signal_df = pd.DataFrame(results)
            signal_df['weight'] = signal_df['ticker'].map(selected_ticker_weight).fillna(0.0)
            csv_signal = signal_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Signal CSV",
                data=csv_signal,
                file_name=f"signals_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        if zone_entries:
            zone_df = pd.DataFrame(zone_entries)
            zone_df["sort_pct"] = zone_df.apply(lambda row: row["Pct_Above_VAH"] if row["zone_status"] == "Above VAH"
                                                else (row["Pct_Below_VAL"] if row["zone_status"] == "Below VAL" else 0), axis=1)
            zone_df_sorted = zone_df.sort_values(["zone_status", "sort_pct"], ascending=[True, False]).drop(columns="sort_pct")
            csv_zone = zone_df_sorted.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Zone CSV",
                data=csv_zone,
                file_name=f"zones_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
