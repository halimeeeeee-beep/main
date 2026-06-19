import streamlit as st
import random
from datetime import date

st.set_page_config(
    page_title="MBTI 오늘의 운세",
    page_icon="🔮",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ffe4f2, #e0f7ff, #fff7cc);
}
.title {
    text-align: center;
    font-size: 52px;
    font-weight: 900;
    color: #8e24aa;
}
.subtitle {
    text-align: center;
    font-size: 22px;
    color: #444;
}
.card {
    background: rgba(255,255,255,0.88);
    padding: 26px;
    border-radius: 28px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.16);
    margin: 18px 0;
    border: 3px solid #f8bbd0;
}
.big {
    font-size: 34px;
    font-weight: 800;
    color: #d81b60;
}
.text {
    font-size: 19px;
    color: #333;
    line-height: 1.7;
}
.score {
    font-size: 26px;
    font-weight: bold;
    color: #3949ab;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔮✨ MBTI 오늘의 운세 ✨🔮</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">MBTI와 출생년도를 입력하면 오늘의 행운 메시지를 알려드려요 🌈🍀💖</div>', unsafe_allow_html=True)

st.write("")
st.write("")

mbti_list = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]

fortune_messages = [
    "오늘은 작은 도전이 큰 기회로 이어질 수 있어요 🚀",
    "말 한마디가 좋은 인연을 만들어 줄 수 있는 날이에요 💬💕",
    "조급해하지 않으면 원하는 결과에 가까워질 수 있어요 🌱",
    "새로운 아이디어가 떠오르기 좋은 하루예요 💡✨",
    "평소보다 자신감을 가지고 행동해도 좋은 날이에요 🔥",
    "주변 사람의 도움을 기분 좋게 받아들이면 행운이 커져요 🤝🍀",
    "정리정돈을 하면 마음도 함께 가벼워질 수 있어요 🧹🌈",
    "오늘은 감정보다 차분한 판단이 행운을 불러와요 🧠⭐",
    "웃는 얼굴이 좋은 기운을 끌어당기는 하루예요 😊💖",
    "미뤄둔 일을 하나만 끝내도 만족감이 커질 거예요 ✅🎉"
]

lucky_items = [
    "파란색 볼펜 🖊️", "딸기 우유 🍓", "하얀 운동화 👟",
    "노란 메모지 📝", "초콜릿 🍫", "작은 거울 🪞",
    "이어폰 🎧", "분홍색 소품 💗", "따뜻한 차 🍵", "책 한 권 📚"
]

lucky_colors = [
    "핑크 💗", "하늘색 🩵", "노랑 💛", "보라 💜",
    "민트 💚", "흰색 🤍", "주황 🧡", "빨강 ❤️"
]

advice_by_mbti = {
    "I": "혼자만의 시간을 조금 가지면 에너지가 충전돼요 🌙",
    "E": "사람들과 대화할수록 좋은 기운이 생겨요 🎉",
    "S": "눈앞의 할 일을 차근차근 해내면 운이 좋아져요 📌",
    "N": "상상력과 아이디어를 마음껏 펼쳐보세요 🌟",
    "T": "논리적인 판단이 좋은 결과를 가져와요 🧠",
    "F": "따뜻한 말과 배려가 행운을 불러와요 💕",
    "J": "계획표를 세우면 하루가 더 편안해져요 📅",
    "P": "즉흥적인 선택이 의외의 즐거움을 줄 수 있어요 🎈"
}

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    mbti = st.selectbox("🌟 MBTI를 선택하세요", mbti_list)
    birth_year = st.number_input(
        "🎂 출생년도를 입력하세요",
        min_value=1950,
        max_value=date.today().year,
        value=2008,
        step=1
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔮 오늘의 운세 보기 🔮"):
        today = date.today()

        seed_value = f"{mbti}-{birth_year}-{today}"
        random.seed(seed_value)

        total_score = random.randint(70, 100)
        love_score = random.randint(60, 100)
        study_score = random.randint(60, 100)
        money_score = random.randint(60, 100)
        health_score = random.randint(60, 100)

        fortune = random.choice(fortune_messages)
        item = random.choice(lucky_items)
        color = random.choice(lucky_colors)

        mbti_advice = [
            advice_by_mbti[mbti[0]],
            advice_by_mbti[mbti[1]],
            advice_by_mbti[mbti[2]],
            advice_by_mbti[mbti[3]]
        ]

        st.balloons()

        st.markdown(f"""
        <div class="card">
            <div class="big">🌈 {today} 오늘의 운세 🌈</div>
            <p class="text">
            ✨ <b>{mbti}</b> 유형, <b>{birth_year}년생</b>의 오늘 운세입니다! ✨<br><br>
            🔮 {fortune}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <div class="big">🍀 종합 운세 점수</div>
            <p class="score">⭐ {total_score}점 / 100점 ⭐</p>
        </div>
        """, unsafe_allow_html=True)

        a, b, c, d = st.columns(4)

        with a:
            st.metric("💕 애정운", f"{love_score}점")
        with b:
            st.metric("📚 공부운", f"{study_score}점")
        with c:
            st.metric("💰 금전운", f"{money_score}점")
        with d:
            st.metric("💪 건강운", f"{health_score}점")

        st.markdown(f"""
        <div class="card">
            <div class="big">🎁 오늘의 행운 아이템</div>
            <p class="text">🍀 {item}</p>
            <div class="big">🎨 오늘의 행운 색깔</div>
            <p class="text">🌈 {color}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="big">💌 MBTI 맞춤 조언</div>
        """, unsafe_allow_html=True)

        for advice in mbti_advice:
            st.write(f"✨ {advice}")

        st.markdown("</div>", unsafe_allow_html=True)

        st.success("🌟 오늘 하루도 반짝반짝 빛나는 하루 보내세요! ✨💖🌈")

st.write("---")
st.caption("🔮 이 앱은 재미와 진로·상담 활동용 예시입니다. 실제 운세와는 관련이 없어요 😊")
