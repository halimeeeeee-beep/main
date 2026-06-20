import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="AI 주식 분석 대시보드",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #020617 100%);
    color: white;
}
.main-title {
    font-size: 46px;
    font-weight: 900;
    text-align: center;
    color: #38bdf8;
    margin-bottom: 5px;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
[data-testid="stMetricValue"] {
    font-size: 28px;
    color: #facc15;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 AI 실시간 주식 분석 대시보드</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Yahoo Finance 실시간 데이터 + 기술적 분석 + AI 종가 예측</div>', unsafe_allow_html=True)

stock_dict = {
    "Apple 🍎": "AAPL",
    "Microsoft 💻": "MSFT",
    "NVIDIA 🤖": "NVDA",
    "Amazon 📦": "AMZN",
    "Google 🔍": "GOOGL",
    "Meta 🌐": "META",
    "Tesla 🚗": "TSLA",
    "Netflix 🎬": "NFLX",
    "AMD 🔥": "AMD",
    "Palantir 🧠": "PLTR",
    "Samsung Electronics 🇰🇷": "005930.KS",
    "SK Hynix 💾": "000660.KS",
    "Hyundai Motor 🚙": "005380.KS",
    "NAVER 🟢": "035420.KS",
    "Kakao 💬": "035720.KS"
}

st.sidebar.title("⚙️ 분석 설정")

selected_name = st.sidebar.selectbox("📌 종목 선택", list(stock_dict.keys()))
ticker = stock_dict[selected_name]

period = st.sidebar.selectbox(
    "📅 데이터 기간",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

interval = st.sidebar.selectbox(
    "⏱️ 데이터 간격",
    ["1d", "1h", "30m", "15m", "5m"],
    index=0
)

st.sidebar.success(f"선택 종목 코드: {ticker}")

@st.cache_data(ttl=300)
def load_stock_data(ticker, period, interval):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "date"})

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    needed_cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[needed_cols]
    df = df.dropna()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    return df

def to_float(value):
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    return float(value)

df = load_stock_data(ticker, period, interval)

if df.empty:
    st.error("데이터를 불러오지 못했습니다. 기간 또는 간격을 바꿔보세요.")
    st.stop()

df["MA5"] = df["close"].rolling(window=5).mean()
df["MA20"] = df["close"].rolling(window=20).mean()
df["MA60"] = df["close"].rolling(window=60).mean()

df["daily_return"] = df["close"].pct_change()
df["volatility"] = df["daily_return"].rolling(window=20).std()

latest_close = to_float(df["close"].iloc[-1])
first_close = to_float(df["close"].iloc[0])
total_return = (latest_close - first_close) / first_close * 100
avg_volume = to_float(df["volume"].mean())
volatility = to_float(df["daily_return"].std() * 100)

st.markdown(f"## 📌 {selected_name} 실시간 분석")
st.caption("Yahoo Finance 데이터를 사용하며, 데이터는 5분마다 새로고침됩니다.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 최근 종가", f"{latest_close:,.2f}")
col2.metric("📈 기간 수익률", f"{total_return:.2f}%")
col3.metric("📦 평균 거래량", f"{avg_volume:,.0f}")
col4.metric("⚡ 변동성", f"{volatility:.2f}%")

st.divider()

st.subheader("🕯️ 캔들 차트 + 이동평균선")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["date"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="캔들차트"
))

fig.add_trace(go.Scatter(x=df["date"], y=df["MA5"], mode="lines", name="MA5"))
fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"], mode="lines", name="MA20"))
fig.add_trace(go.Scatter(x=df["date"], y=df["MA60"], mode="lines", name="MA60"))

fig.update_layout(
    template="plotly_dark",
    height=650,
    xaxis_title="날짜",
    yaxis_title="가격",
    xaxis_rangeslider_visible=False,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.8)"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📦 거래량")

volume_fig = go.Figure()
volume_fig.add_trace(go.Bar(
    x=df["date"],
    y=df["volume"],
    name="거래량"
))

volume_fig.update_layout(
    template="plotly_dark",
    height=350,
    xaxis_title="날짜",
    yaxis_title="거래량",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.8)"
)

st.plotly_chart(volume_fig, use_container_width=True)

st.subheader("📉 수익률 변화")

return_fig = go.Figure()
return_fig.add_trace(go.Scatter(
    x=df["date"],
    y=df["daily_return"] * 100,
    mode="lines",
    name="수익률"
))

return_fig.update_layout(
    template="plotly_dark",
    height=350,
    xaxis_title="날짜",
    yaxis_title="수익률(%)",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.8)"
)

st.plotly_chart(return_fig, use_container_width=True)

st.divider()

st.subheader("🤖 AI 다음 종가 예측")

ai_df = df.copy()

ai_df["prev_close"] = ai_df["close"].shift(1)
ai_df["prev_volume"] = ai_df["volume"].shift(1)
ai_df["return_1d"] = ai_df["close"].pct_change()
ai_df["high_low_gap"] = ai_df["high"] - ai_df["low"]
ai_df["open_close_gap"] = ai_df["open"] - ai_df["close"]
ai_df["target"] = ai_df["close"].shift(-1)

ai_df = ai_df.dropna()

features = [
    "open", "high", "low", "close", "volume",
    "prev_close", "prev_volume", "return_1d",
    "high_low_gap", "open_close_gap", "MA5", "MA20"
]

if len(ai_df) > 80:
    X = ai_df[features]
    y = ai_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=8
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    latest_data = X.iloc[[-1]]
    next_pred = float(model.predict(latest_data)[0])

    col5, col6, col7 = st.columns(3)
    col5.metric("🔮 AI 예측 다음 종가", f"{next_pred:,.2f}")
    col6.metric("📏 MAE", f"{mae:.2f}")
    col7.metric("✅ R² Score", f"{r2:.3f}")

    result_df = pd.DataFrame({
        "date": ai_df.loc[y_test.index, "date"],
        "actual": y_test.values,
        "predicted": pred
    })

    pred_fig = go.Figure()

    pred_fig.add_trace(go.Scatter(
        x=result_df["date"],
        y=result_df["actual"],
        mode="lines",
        name="실제 종가"
    ))

    pred_fig.add_trace(go.Scatter(
        x=result_df["date"],
        y=result_df["predicted"],
        mode="lines",
        name="AI 예측 종가"
    ))

    pred_fig.update_layout(
        template="plotly_dark",
        height=450,
        title="실제 종가 vs AI 예측 종가",
        xaxis_title="날짜",
        yaxis_title="종가",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)"
    )

    st.plotly_chart(pred_fig, use_container_width=True)

else:
    st.warning("데이터가 부족해서 AI 예측을 실행할 수 없습니다. 기간을 더 길게 선택하세요.")

st.info("⚠️ 이 프로그램은 교육용 데이터 분석 예제입니다. 실제 투자 판단이나 수익을 보장하지 않습니다.")

with st.expander("📋 최근 데이터 보기"):
    st.dataframe(df.tail(20), use_container_width=True)
