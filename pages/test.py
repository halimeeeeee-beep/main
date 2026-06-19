import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="🌍 글로벌 시가총액 TOP10 대시보드",
    page_icon="📈",
    layout="wide"
)

st.title("🌍 글로벌 시가총액 TOP10 주식 대시보드")
st.caption("📡 Yahoo Finance에서 실시간으로 데이터를 가져옵니다.")

# 글로벌 시가총액 TOP10 (필요 시 수정 가능)
stocks = {
    "🟢 NVIDIA": "NVDA",
    "🍎 Apple": "AAPL",
    "💻 Microsoft": "MSFT",
    "🔍 Alphabet": "GOOGL",
    "📦 Amazon": "AMZN",
    "📱 Meta": "META",
    "🚗 Tesla": "TSLA",
    "💿 Broadcom": "AVGO",
    "🏭 TSMC": "TSM",
    "🛢️ Saudi Aramco": "2222.SR",
}

selected = st.multiselect(
    "비교할 기업을 선택하세요",
    options=list(stocks.keys()),
    default=list(stocks.keys())
)

@st.cache_data(ttl=300)   # 5분마다 최신 데이터 갱신
def load_data(ticker):
    df = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    return df

fig = go.Figure()

for company in selected:
    ticker = stocks[company]
    df = load_data(ticker)

    if len(df) > 0:
        # 시작 가격을 100으로 정규화
        normalized = df["Close"] / df["Close"].iloc[0] * 100

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=normalized,
                mode="lines",
                name=company,
                hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"
            )
        )

fig.update_layout(
    title="📈 최근 1년간 주가 변화 (시작일 = 100)",
    xaxis_title="날짜",
    yaxis_title="주가 지수",
    hovermode="x unified",
    template="plotly_dark",
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# 현재 가격 표시
st.subheader("💰 현재 주가")

cols = st.columns(5)

for i, company in enumerate(selected):
    ticker = stocks[company]
    info = yf.Ticker(ticker)

    try:
        price = info.fast_info["lastPrice"]
        cols[i % 5].metric(company, f"${price:,.2f}")
    except:
        cols[i % 5].metric(company, "조회 실패")

st.success("✅ 데이터는 Yahoo Finance에서 실시간으로 가져옵니다.")
