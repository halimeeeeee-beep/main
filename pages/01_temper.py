import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="서울 기온 변화 대시보드",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 1907~2026 기온 변화 대시보드")
st.caption("📊 서울의 장기 기온 데이터를 활용한 기후변화 시각화 웹앱")

@st.cache_data
def load_data():
    df = pd.read_csv("ta_20260619190504.csv", encoding="utf-8-sig")

    # 날짜 앞에 붙은 탭 문자 제거
    df["날짜"] = df["날짜"].astype(str).str.strip()

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 빈 행 제거
    df = df.dropna(subset=["날짜", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"])

    # 연도, 월 만들기
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month

    return df

df = load_data()

st.success("✅ 데이터 불러오기 완료!")

col1, col2, col3, col4 = st.columns(4)

col1.metric("📅 시작 연도", int(df["연도"].min()))
col2.metric("📅 마지막 연도", int(df["연도"].max()))
col3.metric("🌡️ 전체 평균기온", f"{df['평균기온(℃)'].mean():.2f}℃")
col4.metric("🔥 최고기온 기록", f"{df['최고기온(℃)'].max():.2f}℃")

st.divider()

yearly = df.groupby("연도")[["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]].mean().reset_index()

st.subheader("📈 연도별 평균기온 변화")

fig1 = px.line(
    yearly,
    x="연도",
    y="평균기온(℃)",
    markers=True,
    title="1907~2026 서울 연도별 평균기온 변화"
)

fig1.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온(℃)",
    height=500
)

st.plotly_chart(fig1, use_container_width=True)

st.subheader("🌡️ 평균기온 · 최저기온 · 최고기온 비교")

fig2 = px.line(
    yearly,
    x="연도",
    y=["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"],
    title="연도별 평균기온, 최저기온, 최고기온 변화"
)

fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="기온(℃)",
    legend_title="기온 종류",
    height=550
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.sidebar.header("🔍 연도 선택")

years = sorted(df["연도"].unique())
selected_year = st.sidebar.selectbox(
    "확인할 연도를 선택하세요",
    years,
    index=len(years) - 1
)

year_df = df[df["연도"] == selected_year]

monthly = year_df.groupby("월")[["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]].mean().reset_index()

st.subheader(f"📅 {selected_year}년 월별 기온 변화")

fig3 = px.line(
    monthly,
    x="월",
    y=["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"],
    markers=True,
    title=f"{selected_year}년 월별 기온 변화"
)

fig3.update_layout(
    xaxis_title="월",
    yaxis_title="기온(℃)",
    legend_title="기온 종류",
    height=500
)

st.plotly_chart(fig3, use_container_width=True)

st.divider()

df["폭염일"] = df["최고기온(℃)"] >= 33
df["한파일"] = df["최저기온(℃)"] <= -12

hot_days = df.groupby("연도")["폭염일"].sum().reset_index()
cold_days = df.groupby("연도")["한파일"].sum().reset_index()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 연도별 폭염일 수")
    fig4 = px.bar(
        hot_days,
        x="연도",
        y="폭염일",
        title="최고기온 33℃ 이상인 날"
    )
    fig4.update_layout(
        xaxis_title="연도",
        yaxis_title="폭염일 수"
    )
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    st.subheader("❄️ 연도별 한파일 수")
    fig5 = px.bar(
        cold_days,
        x="연도",
        y="한파일",
        title="최저기온 -12℃ 이하인 날"
    )
    fig5.update_layout(
        xaxis_title="연도",
        yaxis_title="한파일 수"
    )
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 가장 더웠던 날 TOP10")
    hottest = df.sort_values("최고기온(℃)", ascending=False).head(10)
    st.dataframe(
        hottest[["날짜", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]],
        use_container_width=True
    )

with col2:
    st.subheader("❄️ 가장 추웠던 날 TOP10")
    coldest = df.sort_values("최저기온(℃)", ascending=True).head(10)
    st.dataframe(
        coldest[["날짜", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]],
        use_container_width=True
    )

st.divider()

min_year = int(df["연도"].min())
max_year = int(df["연도"].max())

early = df[(df["연도"] >= min_year) & (df["연도"] < min_year + 10)]
recent = df[(df["연도"] > max_year - 10) & (df["연도"] <= max_year)]

early_mean = early["평균기온(℃)"].mean()
recent_mean = recent["평균기온(℃)"].mean()
diff = recent_mean - early_mean

st.subheader("🌍 초기 10년 vs 최근 10년 평균기온 비교")

col1, col2, col3 = st.columns(3)

col1.metric("초기 10년 평균기온", f"{early_mean:.2f}℃")
col2.metric("최근 10년 평균기온", f"{recent_mean:.2f}℃")
col3.metric("기온 변화", f"{diff:.2f}℃")

if diff > 0:
    st.warning(f"🌡️ 최근 10년 평균기온은 초기 10년보다 약 {diff:.2f}℃ 높습니다.")
else:
    st.info(f"❄️ 최근 10년 평균기온은 초기 10년보다 약 {abs(diff):.2f}℃ 낮습니다.")

st.caption("자료: 서울 기온 관측 데이터")
