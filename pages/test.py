import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="글로벌 시가총액 TOP10 주식 대시보드",
    page_icon="📈",
    layout="wide"
)

st.title("🌍 글로벌 시가총액 TOP10 주식 대시보드")
st.caption("📡 Yahoo Finance 데이터를 실시간으로 가져와 시각화합니다.")

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
    "📌 비교할 기업을 선택하세요",
    options=list(stocks.keys()),
    default=list(stocks.keys())
)

@st.cache_data(ttl=300)
def load_data(ticker):
    df = yf.Ticker(ticker).history(
        period="1y",
        interval="1d",
        auto_adjust=True
    )
    return df

fig = go.Figure()

if selected:
    for company in selected:
        ticker = stocks[company]
        df = load_data(ticker)

        if df.empty:
            st.warning(f"{company} 데이터를 가져오지 못했습니다.")
            continue

        close = df["Close"].dropna()

        if len(close) == 0:
            st.warning(f"{company} 종가 데이터가 없습니다.")
            continue

        normalized = close / close.iloc[0] * 100

        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized.values,
                mode="lines",
                name=company,
                hovertemplate="%{x}<br>주가 지수: %{y:.2f}<extra></extra>"
            )
        )

    fig.update_layout(
        title="📈 최근 1년간 주가 변화 비교 (시작일 = 100)",
        xaxis_title="날짜",
        yaxis_title="주가 지수",
        hovermode="x unified",
        template="plotly_white",
        height=700,
        legend_title_text="기업"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💰 현재 주가")

    cols = st.columns(5)

    for i, company in enumerate(selected):
        ticker = stocks[company]
        df = load_data(ticker)

        if not df.empty and "Close" in df.columns:
            current_price = df["Close"].dropna().iloc[-1]
            first_price = df["Close"].dropna().iloc[0]
            change_rate = ((current_price - first_price) / first_price) * 100

            cols[i % 5].metric(
                label=company,
                value=f"${current_price:,.2f}",
                delta=f"{change_rate:.2f}%"
            )
        else:
            cols[i % 5].metric(company, "조회 실패")

else:
    st.warning("기업을 1개 이상 선택해주세요.")
