# -----------------------------
# 차트 분석 기법 선택
# -----------------------------
st.divider()
st.subheader("📊 원하는 주식 차트 분석 기법 선택")

chart_options = st.multiselect(
    "보고 싶은 차트 분석 기법을 선택하세요",
    [
        "🕯️ 캔들차트",
        "📈 이동평균선",
        "📦 볼린저 밴드",
        "💪 RSI",
        "📉 MACD",
        "📊 거래량",
        "⚡ 일별 수익률",
        "🚀 누적 수익률"
    ],
    default=["🕯️ 캔들차트", "📈 이동평균선"]
)

# -----------------------------
# 추가 지표 계산
# -----------------------------

# 볼린저 밴드
df["BB_Middle"] = df["close"].rolling(window=20).mean()
df["BB_Std"] = df["close"].rolling(window=20).std()
df["BB_Upper"] = df["BB_Middle"] + (df["BB_Std"] * 2)
df["BB_Lower"] = df["BB_Middle"] - (df["BB_Std"] * 2)

# RSI
delta = df["close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# MACD
ema12 = df["close"].ewm(span=12, adjust=False).mean()
ema26 = df["close"].ewm(span=26, adjust=False).mean()

df["MACD"] = ema12 - ema26
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Hist"] = df["MACD"] - df["Signal"]

# 누적 수익률
df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1

# -----------------------------
# 1. 캔들차트 / 이동평균선 / 볼린저 밴드
# -----------------------------
if (
    "🕯️ 캔들차트" in chart_options
    or "📈 이동평균선" in chart_options
    or "📦 볼린저 밴드" in chart_options
):
    st.subheader("🕯️ 가격 차트")

    price_fig = go.Figure()

    if "🕯️ 캔들차트" in chart_options:
        price_fig.add_trace(go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="캔들차트"
        ))
    else:
        price_fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["close"],
            mode="lines",
            name="종가"
        ))

    if "📈 이동평균선" in chart_options:
        price_fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["MA5"],
            mode="lines",
            name="MA5"
        ))

        price_fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["MA20"],
            mode="lines",
            name="MA20"
        ))

        price_fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["MA60"],
            mode="lines",
            name="MA60"
        ))

    if "📦 볼린저 밴드" in chart_options:
        price_fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["BB_Upper"],
            mode="lines",
            name="볼린저 상단"
        ))

        price_fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["BB_Middle"],
            mode="lines",
            name="볼린저 중심선"
        ))

        price_fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["BB_Lower"],
            mode="lines",
            name="볼린저 하단"
        ))

    price_fig.update_layout(
        template="plotly_dark",
        height=650,
        xaxis_title="날짜",
        yaxis_title="가격",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)"
    )

    st.plotly_chart(price_fig, use_container_width=True)

# -----------------------------
# 2. RSI
# -----------------------------
if "💪 RSI" in chart_options:
    st.subheader("💪 RSI 분석")

    rsi_fig = go.Figure()

    rsi_fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["RSI"],
        mode="lines",
        name="RSI"
    ))

    rsi_fig.add_hline(y=70, line_dash="dash", annotation_text="과매수 구간")
    rsi_fig.add_hline(y=30, line_dash="dash", annotation_text="과매도 구간")

    rsi_fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="날짜",
        yaxis_title="RSI",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)"
    )

    st.plotly_chart(rsi_fig, use_container_width=True)

    st.info("RSI가 70 이상이면 과매수, 30 이하이면 과매도로 해석하는 경우가 많습니다.")

# -----------------------------
# 3. MACD
# -----------------------------
if "📉 MACD" in chart_options:
    st.subheader("📉 MACD 분석")

    macd_fig = go.Figure()

    macd_fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["MACD"],
        mode="lines",
        name="MACD"
    ))

    macd_fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["Signal"],
        mode="lines",
        name="Signal"
    ))

    macd_fig.add_trace(go.Bar(
        x=df["date"],
        y=df["MACD_Hist"],
        name="MACD Histogram"
    ))

    macd_fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="날짜",
        yaxis_title="MACD",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)"
    )

    st.plotly_chart(macd_fig, use_container_width=True)

    st.info("MACD선이 Signal선을 위로 돌파하면 상승 신호, 아래로 돌파하면 하락 신호로 해석하기도 합니다.")

# -----------------------------
# 4. 거래량
# -----------------------------
if "📊 거래량" in chart_options:
    st.subheader("📊 거래량 분석")

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

# -----------------------------
# 5. 일별 수익률
# -----------------------------
if "⚡ 일별 수익률" in chart_options:
    st.subheader("⚡ 일별 수익률")

    return_fig = go.Figure()

    return_fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["daily_return"] * 100,
        mode="lines",
        name="일별 수익률"
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

# -----------------------------
# 6. 누적 수익률
# -----------------------------
if "🚀 누적 수익률" in chart_options:
    st.subheader("🚀 누적 수익률")

    cumulative_fig = go.Figure()

    cumulative_fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["cumulative_return"] * 100,
        mode="lines",
        name="누적 수익률"
    ))

    cumulative_fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="날짜",
        yaxis_title="누적 수익률(%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)"
    )

    st.plotly_chart(cumulative_fig, use_container_width=True)
