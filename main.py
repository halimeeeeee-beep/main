import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="AI 실시간 주식 분석 대시보드",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI 실시간 주식 분석 대시보드")
st.write("Yahoo Finance 데이터를 실시간으로 불러와 주가 차트와 AI 예측을 제공합니다.")

# -----------------------------
# 종목 목록
# -----------------------------
stock_dict = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Meta": "META",
    "Tesla": "TSLA",
    "Netflix": "NFLX",
    "AMD": "AMD",
    "Palantir": "PLTR",
    "Samsung Electronics": "005930.KS",
    "SK Hynix": "000660.KS",
    "Hyundai Motor": "005380.KS",
    "NAVER": "035420.KS",
    "Kakao": "035720.KS"
}

st.sidebar.header("⚙️ 분석 설정")

selected_name = st.sidebar.selectbox("종목 선택", list(stock_dict.keys()))
ticker = stock_dict[selected_name]

period = st.sidebar.selectbox(
    "데이터 기간",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

interval = st.sidebar.selectbox(
    "데이터 간격",
    ["1d", "1h", "30m", "15m", "5m"],
    index=0
)

st.sidebar.info(f"선택 종목 코드: {ticker}")

@st.cache_data(ttl=300)
def load_stock_data(ticker, period, interval):
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    data = data.reset_index()

    if "Date" in data.columns:
        data = data.rename(columns={"Date": "date"})
    elif "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "date"})

    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    data = data.dropna()
    return data

df = load_stock_data(ticker, period, interval)

if df.empty:
    st.error("데이터를 불러오지 못했습니다. 기간 또는 간격을 변경해 보세요.")
    st.stop()

# -----------------------------
# 기술적 지표 계산
# -----------------------------
df["MA5"] = df["close"].rolling(window=5).mean()
df["MA20"] = df["close"].rolling(window=20).mean()
df["MA60"] = df["close"].rolling(window=60).mean()

df["daily_return"] = df["close"].pct_change()
df["volatility"] = df["daily_return"].rolling(window=20).std()

# -----------------------------
# 핵심 지표
# -----------------------------
st.subheader(f"📌 {selected_name} ({ticker}) 실시간 분석")

latest_close = df["close"].iloc[-1]
first_close = df["close"].iloc[0]
total_return = (latest_close - first_close) / first_close * 100
avg_volume = df["volume"].mean()
volatility = df["daily_return"].std() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("최근 종가", f"{latest_close:,.2f}")
col2.metric("기간 수익률", f"{total_return:.2f}%")
col3.metric("평균 거래량", f"{avg_volume:,.0f}")
col4.metric("변동성", f"{volatility:.2f}%")

st.caption("데이터는 Yahoo Finance에서 가져오며, 5분마다 자동 갱신됩니다.")

# -----------------------------
# 캔들 차트
# -----------------------------
st.subheader("📊 주가 차트 + 이동평균선")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["date"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="캔들차트"
))

fig.add_trace(go.Scatter(
    x=df["date"],
    y=df["MA5"],
    mode="lines",
    name="MA5"
))

fig.add_trace(go.Scatter(
    x=df["date"],
    y=df["MA20"],
    mode="lines",
    name="MA20"
))

fig.add_trace(go.Scatter(
    x=df["date"],
    y=df["MA60"],
    mode="lines",
    name="MA60"
))

fig.update_layout(
    height=600,
    xaxis_title="날짜",
    yaxis_title="가격",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 거래량
# -----------------------------
st.subheader("📦 거래량 분석")

volume_fig = go.Figure()
volume_fig.add_trace(go.Bar(
    x=df["date"],
    y=df["volume"],
    name="거래량"
))

volume_fig.update_layout(
    height=350,
    xaxis_title="날짜",
    yaxis_title="거래량"
)

st.plotly_chart(volume_fig, use_container_width=True)

# -----------------------------
# 일별 수익률
# -----------------------------
st.subheader("📉 수익률 분석")

return_fig = go.Figure()
return_fig.add_trace(go.Scatter(
    x=df["date"],
    y=df["daily_return"] * 100,
    mode="lines",
    name="수익률"
))

return_fig.update_layout(
    height=350,
    xaxis_title="날짜",
    yaxis_title="수익률(%)"
)

st.plotly_chart(return_fig, use_container_width=True)

# -----------------------------
# AI 예측
# -----------------------------
st.divider()
st.subheader("🤖 AI 다음 데이터 종가 예측")

ai_df = df.copy()

ai_df["prev_close"] = ai_df["close"].shift(1)
ai_df["prev_volume"] = ai_df["volume"].shift(1)
ai_df["return_1d"] = ai_df["close"].pct_change()
ai_df["high_low_gap"] = ai_df["high"] - ai_df["low"]
ai_df["open_close_gap"] = ai_df["open"] - ai_df["close"]
ai_df["target"] = ai_df["close"].shift(-1)

ai_df = ai_df.dropna()

features = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "prev_close",
    "prev_volume",
    "return_1d",
    "high_low_gap",
    "open_close_gap",
    "MA5",
    "MA20"
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
    next_pred = model.predict(latest_data)[0]

    col5, col6, col7 = st.columns(3)
    col5.metric("AI 예측 다음 종가", f"{next_pred:,.2f}")
    col6.metric("MAE", f"{mae:.2f}")
    col7.metric("R² Score", f"{r2:.3f}")

    result_df = pd.DataFrame({
        "date": ai_df.loc[y_test.index, "date"],
        "actual": y_test,
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
        height=450,
        title="실제 종가 vs AI 예측 종가",
        xaxis_title="날짜",
        yaxis_title="종가"
    )

    st.plotly_chart(pred_fig, use_container_width=True)

else:
    st.warning("데이터가 부족해서 AI 예측을 실행할 수 없습니다. 기간을 더 길게 선택하세요.")

st.info("이 프로그램은 교육용 데이터 분석 예제입니다. 실제 투자 판단에 사용하면 안 됩니다.")

st.subheader("📋 최근 데이터")
st.dataframe(df.tail(20), use_container_width=True)
