import streamlit as st
import random

st.set_page_config(
    page_title="MBTI 진로 추천 앱",
    page_icon="🌈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #FFE5EC, #E0F7FA, #EDE7F6);
}
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #7B1FA2;
}
.subtitle {
    text-align: center;
    font-size: 22px;
    color: #444;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 25px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
    margin: 15px 0;
}
.job {
    font-size: 25px;
    font-weight: bold;
    color: #D81B60;
}
.desc {
    font-size: 18px;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌈✨ MBTI 진로 추천 웹앱 ✨🌈</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">나의 성향에 어울리는 직업을 찾아보세요! 💼🚀🎨📚</div>', unsafe_allow_html=True)

st.write("")
st.write("")

mbti_jobs = {
    "ISTJ": {
        "emoji": "📋🧠",
        "name": "현실적이고 책임감 있는 관리자형",
        "jobs": ["공무원", "회계사", "데이터 분석가", "법무사", "품질관리 전문가"],
        "reason": "꼼꼼하고 책임감이 강해서 정확성과 신뢰가 중요한 직업에 잘 어울려요."
    },
    "ISFJ": {
        "emoji": "🤝🌷",
        "name": "따뜻하고 성실한 보호자형",
        "jobs": ["간호사", "교사", "사회복지사", "상담사", "행정직"],
        "reason": "다른 사람을 돕고 배려하는 능력이 뛰어나 사람을 지원하는 직업에 적합해요."
    },
    "INFJ": {
        "emoji": "🌙📖",
        "name": "통찰력 있는 조언자형",
        "jobs": ["심리상담사", "작가", "교사", "기획자", "사회운동가"],
        "reason": "사람의 마음을 잘 이해하고 의미 있는 일을 추구하는 성향이 강해요."
    },
    "INTJ": {
        "emoji": "♟️🔬",
        "name": "전략적인 설계자형",
        "jobs": ["연구원", "개발자", "전략기획자", "건축가", "AI 전문가"],
        "reason": "논리적이고 장기적인 계획을 잘 세워 분석과 전략이 필요한 직업에 어울려요."
    },
    "ISTP": {
        "emoji": "🛠️🏍️",
        "name": "실용적인 문제해결자형",
        "jobs": ["엔지니어", "파일럿", "정비사", "경찰관", "응급구조사"],
        "reason": "손으로 직접 다루고 문제를 해결하는 능력이 뛰어나요."
    },
    "ISFP": {
        "emoji": "🎨🌿",
        "name": "감성적인 예술가형",
        "jobs": ["디자이너", "플로리스트", "요리사", "음악가", "동물관련 직업"],
        "reason": "감각적이고 섬세해서 아름다움과 감성을 표현하는 직업에 잘 맞아요."
    },
    "INFP": {
        "emoji": "🦋✍️",
        "name": "이상적인 중재자형",
        "jobs": ["작가", "상담사", "콘텐츠 크리에이터", "예술가", "비영리단체 활동가"],
        "reason": "자신만의 가치관이 뚜렷하고 창의적인 표현을 잘해요."
    },
    "INTP": {
        "emoji": "🧪💡",
        "name": "논리적인 사색가형",
        "jobs": ["프로그래머", "과학자", "교수", "데이터 사이언티스트", "철학자"],
        "reason": "호기심이 많고 복잡한 문제를 분석하는 데 강점이 있어요."
    },
    "ESTP": {
        "emoji": "🔥🏆",
        "name": "활동적인 도전가형",
        "jobs": ["창업가", "영업 전문가", "스포츠 지도자", "경찰관", "마케터"],
        "reason": "에너지가 넘치고 순발력이 좋아 빠르게 움직이는 직업에 잘 맞아요."
    },
    "ESFP": {
        "emoji": "🎤🎉",
        "name": "사교적인 연예인형",
        "jobs": ["배우", "방송인", "이벤트 기획자", "승무원", "서비스직"],
        "reason": "사람들과 어울리는 것을 좋아하고 분위기를 밝게 만드는 능력이 있어요."
    },
    "ENFP": {
        "emoji": "🌟🚀",
        "name": "재기발랄한 활동가형",
        "jobs": ["광고기획자", "크리에이터", "상담사", "교사", "마케터"],
        "reason": "창의적이고 열정적이라 새로운 아이디어를 내는 직업에 어울려요."
    },
    "ENTP": {
        "emoji": "⚡🗣️",
        "name": "논쟁을 즐기는 발명가형",
        "jobs": ["창업가", "변호사", "기획자", "컨설턴트", "개발자"],
        "reason": "토론과 아이디어 발상이 뛰어나 새로운 도전을 즐기는 직업에 적합해요."
    },
    "ESTJ": {
        "emoji": "📊👔",
        "name": "체계적인 관리자형",
        "jobs": ["경영자", "공무원", "군인", "프로젝트 매니저", "금융 전문가"],
        "reason": "조직을 관리하고 목표를 달성하는 능력이 뛰어나요."
    },
    "ESFJ": {
        "emoji": "💖🏫",
        "name": "친절한 협력자형",
        "jobs": ["교사", "간호사", "상담사", "인사담당자", "서비스 매니저"],
        "reason": "사람들과 협력하고 분위기를 조화롭게 만드는 능력이 좋아요."
    },
    "ENFJ": {
        "emoji": "🌻🎓",
        "name": "따뜻한 리더형",
        "jobs": ["교사", "강연가", "상담사", "HR 전문가", "사회복지사"],
        "reason": "사람을 이끌고 성장시키는 데 큰 보람을 느끼는 유형이에요."
    },
    "ENTJ": {
        "emoji": "👑🚀",
        "name": "대담한 통솔자형",
        "jobs": ["CEO", "경영컨설턴트", "변호사", "정치인", "프로젝트 리더"],
        "reason": "목표지향적이고 리더십이 강해서 조직을 이끄는 직업에 잘 맞아요."
    }
}

mbti_list = list(mbti_jobs.keys())

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    selected_mbti = st.selectbox(
        "🌟 나의 MBTI를 선택하세요!",
        mbti_list
    )

    if st.button("💖 직업 추천 받기 💖"):
        info = mbti_jobs[selected_mbti]

        st.balloons()

        st.markdown(f"""
        <div class="card">
            <h2>{info["emoji"]} {selected_mbti}</h2>
            <h3>✨ {info["name"]} ✨</h3>
            <p class="desc">{info["reason"]}</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("💼 추천 직업 TOP 5")

        for i, job in enumerate(info["jobs"], start=1):
            st.markdown(f"""
            <div class="card">
                <div class="job">🌟 {i}. {job}</div>
                <div class="desc">이 직업은 {selected_mbti} 유형의 성향과 잘 어울릴 수 있어요! 💕</div>
            </div>
            """, unsafe_allow_html=True)

        st.success("🎉 진로 선택은 MBTI만으로 결정하지 말고, 흥미·능력·가치관도 함께 생각해보세요!")

st.write("")
st.write("---")

st.markdown("### 🎯 진로 탐색 꿀팁")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.info("💡 내가 좋아하는 활동을 적어보기")

with col_b:
    st.warning("📚 관련 학과와 과목 찾아보기")

with col_c:
    st.success("🚀 실제 직업인의 인터뷰 찾아보기")

st.write("---")
st.caption("🌈 이 앱은 진로교육 활동용 예시 앱입니다. MBTI 결과는 참고용으로만 활용하세요.")
