# EMERGENCY DEPENDENCY INSTALLER — работает даже без requirements.txt
import sys
import subprocess
try:
    import yfinance
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance==0.2.37", "pandas==2.1.4", "numpy==1.26.2", "matplotlib==3.8.2"])
    import yfinance

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Multi-Timeframe Trend Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Стили
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton button {
        background-color: #2E7D32;
        color: white;
        font-size: 20px;
        padding: 15px 30px;
        border-radius: 10px;
        border: none;
        width: 100%;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #1B5E20;
        transform: scale(1.02);
    }
    h1 { color: #263238; text-align: center; font-size: 2.5em; margin-bottom: 10px; }
    .subtitle { text-align: center; color: #607D8B; font-size: 1.2em; margin-bottom: 40px; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Multi-Timeframe Trend Analysis")
st.markdown('<p class="subtitle">Анализ трендов по 16 ключевым активам — один клик и готово</p>', unsafe_allow_html=True)

# ========================================
# Универсальные функции
# ========================================
def normalize_df(df):
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def calc_regression(series, window=20):
    if len(series) < window:
        return None, None, None, None
    y = series.tail(window).values
    x = np.arange(window)
    n = len(x)
    slope = (n * np.sum(x*y) - np.sum(x)*np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    intercept = (np.sum(y) - slope * np.sum(x)) / n
    line = slope * x + intercept
    std = np.std(y - line)
    return slope, line - 2*std, line, line + 2*std

def classify_trend(pct):
    if pct is None or np.isnan(pct):
        return "NO DATA", '#9E9E9E'
    if pct > 0.0100: return "VERY BULLISH", '#2E7D32'
    elif pct >= 0.0025: return "BULLISH", '#4CAF50'
    elif pct >= -0.0025: return "NEUTRAL", '#607D8B'
    elif pct >= -0.0100: return "BEARISH", '#F44336'
    else: return "VERY BEARISH", '#B71C1C'

def generate_chart(df_1h, name, ticker):
    """Генерирует график и возвращает matplotlib figure"""
    if df_1h.empty or len(df_1h) < 20:
        return None, "NO DATA", "NO DATA", "NO DATA"
    
    # Конвертация в МСК
    if df_1h.index.tz is None:
        df_1h.index = df_1h.index.tz_localize('UTC')
    df_1h.index = df_1h.index.tz_convert('Europe/Moscow')
    
    # Агрегация
    df_4h = df_1h.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    df_1d = df_1h.resample('24H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    
    if len(df_1d) < 20 or len(df_4h) < 20 or len(df_1h) < 20:
        return None, "INSUFFICIENT DATA", "INSUFFICIENT DATA", "INSUFFICIENT DATA"
    
    # Расчёт каналов
    slope_1h, lower_1h, mid_1h, upper_1h = calc_regression(df_1h['Close'])
    slope_4h, lower_4h, mid_4h, upper_4h = calc_regression(df_4h['Close'])
    slope_1d, lower_1d, mid_1d, upper_1d = calc_regression(df_1d['Close'])
    
    if slope_1h is None or slope_4h is None or slope_1d is None:
        return None, "CALC ERROR", "CALC ERROR", "CALC ERROR"
    
    # Нормализация
    cur = df_1h['Close'].iloc[-1]
    pct_1h = (slope_1h / cur) * 100
    pct_4h = ((slope_4h / df_4h['Close'].iloc[-1]) * 100) / 4.0
    pct_1d = ((slope_1d / df_1d['Close'].iloc[-1]) * 100) / 24.0
    
    cat_1d, col_1d = classify_trend(pct_1d)
    cat_4h, col_4h = classify_trend(pct_4h)
    cat_1h, col_1h = classify_trend(pct_1h)
    
    # Визуализация
    plot_window = 100
    df_plot = df_1h.tail(plot_window * 2).dropna().tail(plot_window)
    
    if len(df_plot) < 20:
        return None, "NO DATA", "NO DATA", "NO DATA"
    
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='white')
    ax.set_facecolor('#F8F9FA')
    
    # Японские свечи
    for i, (_, row) in enumerate(df_plot.iterrows()):
        o, c, h, l = row['Open'], row['Close'], row['High'], row['Low']
        clr = '#26A69A' if c >= o else '#EF5350'
        ax.bar(i, abs(c-o), bottom=min(o,c), width=0.8, color=clr, edgecolor='black', linewidth=0.8)
        ax.plot([i,i], [h, max(o,c)], color='black', linewidth=1)
        ax.plot([i,i], [min(o,c), l], color='black', linewidth=1)
    
    # Каналы
    if len(df_1d) >= 20:
        x = np.linspace(max(0, plot_window-96), plot_window, 5)
        y_mid = np.linspace(mid_1d[-4], mid_1d[-1], 5)
        y_up = np.linspace(upper_1d[-4], upper_1d[-1], 5)
        y_low = np.linspace(lower_1d[-4], lower_1d[-1], 5)
        ax.plot(x, y_mid, color=col_1d, linewidth=2.8, alpha=0.9)
        ax.fill_between(x, y_low, y_up, color=col_1d, alpha=0.08)
    
    if len(df_4h) >= 20:
        x = np.linspace(max(0, plot_window-80), plot_window, 21)
        y_mid = np.interp(x, np.linspace(max(0, plot_window-80), plot_window, 20), mid_4h[-20:])
        y_up = np.interp(x, np.linspace(max(0, plot_window-80), plot_window, 20), upper_4h[-20:])
        y_low = np.interp(x, np.linspace(max(0, plot_window-80), plot_window, 20), lower_4h[-20:])
        ax.plot(x, y_mid, color=col_4h, linewidth=2.6, alpha=0.95)
        ax.fill_between(x, y_low, y_up, color=col_4h, alpha=0.12)
    
    if len(df_1h) >= 20:
        x = np.arange(plot_window-20, plot_window)
        ax.plot(x, mid_1h[-20:], color=col_1h, linewidth=3.2, alpha=1.0, marker='o', markersize=4)
        ax.fill_between(x, lower_1h[-20:], upper_1h[-20:], color=col_1h, alpha=0.20)
    
    # Оформление
    ax.set_title(f'{name} ({ticker}) — Multi-Timeframe Trend Analysis', fontsize=24, fontweight='bold', pad=20, color='#263238')
    ax.set_ylabel('Price', fontsize=14, fontweight='bold', labelpad=12, color='#37474F')
    ax.set_xlabel('Time (MSK)', fontsize=14, fontweight='bold', labelpad=12, color='#37474F')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    # Ось X
    x_ticks = np.arange(0, len(df_plot)+1, max(1, len(df_plot)//10))
    x_labels = [df_plot.index[int(i)].strftime('%m-%d %H:%M') for i in x_ticks if i < len(df_plot)]
    ax.set_xticks(x_ticks[:len(x_labels)])
    ax.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=10, color='#37474F')
    
    # Сводка трендов
    trend_text = f"""DAILY TREND (20d):    {cat_1d}
MID-TERM (20×4h):    {cat_4h}
LOCAL TREND (20h):   {cat_1h}"""
    bbox = dict(boxstyle='round,pad=0.9', facecolor='white', alpha=0.92, edgecolor='#E0E0E0', linewidth=1.0)
    ax.text(0.03, 0.97, trend_text, transform=ax.transAxes, fontsize=13, fontweight='bold',
            verticalalignment='top', family='monospace', color='#263238', bbox=bbox)
    
    # Период
    period = f"{df_plot.index[0].strftime('%d %b %Y %H:%M')} → {df_plot.index[-1].strftime('%d %b %Y %H:%M')} MSK"
    ax.text(0.99, 0.02, period, transform=ax.transAxes, fontsize=10, color='gray', ha='right', style='italic', alpha=0.8)
    
    plt.tight_layout()
    return fig, cat_1d, cat_4h, cat_1h

# ========================================
# Список активов
# ========================================
assets = [
    ("S&P 500", "SPY"),
    ("Hang Seng", "^HSI"),
    ("NASDAQ", "QQQ"),
    ("EURO STOXX 50", "FEZ"),
    ("MSCI WORLD", "URTH"),
    ("Bitcoin", "BTC-USD"),
    ("Ethereum", "ETH-USD"),
    ("Solana", "SOL-USD"),
    ("Gold", "GC=F"),
    ("Silver", "SI=F"),
    ("Platinum", "PL=F"),
    ("Palladium", "PA=F"),
    ("Copper", "HG=F"),
    ("Brent", "BZ=F"),
    ("Natural Gas US", "NG=F"),
    ("DXY", "DX-Y.NYB")
]

# ========================================
# Кнопка генерации
# ========================================
if st.button("🚀 СГЕНЕРИРОВАТЬ ВСЕ ГРАФИКИ", use_container_width=True):
    st.markdown("---")
    progress_bar = st.progress(0)
    status_text = st.empty()
    charts_container = st.container()
    
    success_count = 0
    failed = []
    
    for i, (name, ticker) in enumerate(assets, 1):
        progress_bar.progress(i / len(assets))
        status_text.markdown(f"⏳ {name} ({ticker})... ({i}/{len(assets)})")
        
        try:
            # Загрузка основных данных
            df = normalize_df(yf.download(ticker, period="730d", interval="1h", progress=False))
            
            # Резервные варианты
            if len(df) < 50 and ticker == "^HSI":
                df = normalize_df(yf.download("HSI=F", period="730d", interval="1h", progress=False))
            if len(df) < 50 and ticker == "DX-Y.NYB":
                df = normalize_df(yf.download("^DX-Y", period="730d", interval="1h", progress=False))
            
            if len(df) < 50:
                failed.append((name, ticker, "недостаточно данных"))
                continue
            
            # Генерация графика
            fig, cat_1d, cat_4h, cat_1h = generate_chart(df, name, ticker)
            
            if fig is None:
                failed.append((name, ticker, f"ошибка генерации: {cat_1d}"))
                continue
            
            # Отображение
            with charts_container:
                st.markdown(f"### 📊 {name} ({ticker})")
                st.pyplot(fig)
                st.markdown(f"**Тренды:** Дневной: `{cat_1d}`, Среднесрочный: `{cat_4h}`, Локальный: `{cat_1h}`")
                st.markdown("---")
            
            plt.close(fig)
            success_count += 1
            
        except Exception as e:
            failed.append((name, ticker, str(e)[:50]))
            continue
    
    # Результат
    progress_bar.empty()
    status_text.empty()
    
    if success_count > 0:
        st.success(f"✅ Успешно сгенерировано {success_count} из {len(assets)} графиков")
    
    if failed:
        with st.expander(f"⚠️ {len(failed)} активов с ошибками"):
            for name, ticker, err in failed:
                st.write(f"• **{name}** (`{ticker}`): `{err}`")

# ========================================
# Информация
# ========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #607D8B; padding: 20px;'>
    <p><strong>💡 Совет:</strong> Первый запуск может занять 2-3 минуты (загрузка данных).</p>
    <p>Последующие запуски будут быстрее благодаря кэшированию.</p>
    <p style='margin-top: 20px; font-size: 0.9em; color: #9E9E9E;'>
        Данные: Yahoo Finance | Обновление: в реальном времени
    </p>
</div>
""", unsafe_allow_html=True)
