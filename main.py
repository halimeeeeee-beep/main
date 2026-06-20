import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="🧠 Brain Cell AI Explorer",
    page_icon="🧠",
    layout="wide"
)

# =========================
# 🎨 Pastel Pink + Sky Blue Theme
# =========================
st.markdown("""
<style>

/* 전체 앱 배경 */
.stApp,
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #FFF1F7 0%, #FFF7FB 45%, #FFF0F6 100%) !important;
}

/* 메인 컨테이너 */
.main .block-container {
    background: rgba(255, 246, 250, 0.95) !important;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    border-radius: 24px;
}

/* 상단 헤더 투명 */
[data-testid="stHeader"] {
    background: rgba(255,255,255,0) !important;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFD6E8 0%, #FFEAF3 100%) !important;
    border-right: 2px solid #FFC1DC;
}

/* 사이드바 안쪽 글자 */
[data-testid="stSidebar"] * {
    color: #333333 !important;
}

/* 제목 */
.big-title {
    font-size: 54px;
    font-weight: 1000;
    text-align: center;
    color: #EC407A !important;
    text-shadow: 2px 2px 0px rgba(255,255,255,0.9);
    margin-bottom: 0.5rem;
}

/* 부제목 */
.sub-title {
    text-align: center;
    font-size: 23px;
    color: #333333 !important;
    font-weight: 800;
    margin-bottom: 2rem;
}

/* 전체 글씨 */
html, body, p, div, span, label, h1, h2, h3, h4, h5, h6 {
    color: #2F2F2F !important;
}

/* 구분선 */
hr {
    border: none;
    border-top: 2px solid #FFC1DC;
}

/* Metric 카드 */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.92) !important;
    padding: 24px;
    border-radius: 22px;
    border: 1.5px solid #FFD1E4;
    box-shadow: 0 8px 22px rgba(236, 64, 122, 0.13);
}

/* Metric 라벨 */
[data-testid="stMetricLabel"] {
    color: #333333 !important;
    font-weight: 800 !important;
    font-size: 18px !important;
}

/* Metric 숫자 */
[data-testid="stMetricValue"] {
    color: #EC407A !important;
    font-size: 40px !important;
    font-weight: 1000 !important;
}

/* 검정 상자였던 selectbox를 하늘색으로 */
div[data-baseweb="select"] > div {
    background: linear-gradient(180deg, #EAF7FF 0%, #DFF2FF 100%) !important;
    border: 2px solid #9BD4FF !important;
    border-radius: 14px !important;
    color: #1F2937 !important;
    min-height: 58px !important;
}

/* selectbox 내부 글자 */
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #1F2937 !important;
    font-weight: 700 !important;
}

/* selectbox 화살표 */
div[data-baseweb="select"] svg {
    fill: #1F2937 !important;
}

/* dropdown 목록 */
div[role="listbox"],
ul[role="listbox"] {
    background: #EAF7FF !important;
    border: 1px solid #9BD4FF !important;
}

/* dropdown 항목 */
div[role="option"] {
    background: #EAF7FF !important;
    color: #1F2937 !important;
}

/* multiselect도 하늘색 */
[data-baseweb="tag"] {
    background-color: #BDE7FF !important;
    color: #1F2937 !important;
}

/* 파일 업로드 검정 박스를 하늘색으로 */
[data-testid="stFileUploader"] {
    background: linear-gradient(180deg, #EAF7FF 0%, #DFF2FF 100%) !important;
    border: 2px solid #9BD4FF !important;
    border-radius: 16px !important;
    padding: 16px !important;
}

/* 파일 업로드 내부 박스 */
[data-testid="stFileUploaderDropzone"] {
    background: #EAF7FF !important;
    border: 2px dashed #8CCEFF !important;
    border-radius: 14px !important;
}

/* 파일 업로드 버튼 */
[data-testid="stFileUploader"] button {
    background: #EAF7FF !important;
    color: #1976D2 !important;
    border: 2px solid #8CCEFF !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
}

/* 일반 버튼 */
.stButton > button {
    background: linear-gradient(135deg, #FF69A6 0%, #FF8FC4 100%) !important;
    color: white !important;
    border-radius: 14px !important;
    border: none !important;
    font-weight: 900 !important;
    box-shadow: 0 5px 12px rgba(236, 64, 122, 0.25);
}

/* slider */
[data-testid="stSlider"] * {
    color: #333333 !important;
}

/* radio 버튼 */
.stRadio label {
    color: #333333 !important;
    font-weight: 700 !important;
}

/* expander, info, success */
[data-testid="stAlert"] {
    border-radius: 16px !important;
}

/* dataframe */
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
}

/* 카드 컨테이너용 */
.pretty-card {
    background: rgba(255,255,255,0.92);
    border: 1.5px solid #FFD1E4;
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 8px 22px rgba(236, 64, 122, 0.12);
    margin-bottom: 22px;
}

/* 섹션 제목 */
.section-title {
    font-size: 38px;
    font-weight: 1000;
    color: #333333 !important;
    margin-top: 22px;
    margin-bottom: 18px;
}

</style>
""", unsafe_allow_html=True)


st.markdown('<div class="big-title">🧠 ✨ 뇌세포 AI 선별 프로그램 ✨ 🧬</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">이미지 업로드 + 군집 분석 + 분류 + 회귀 + UMAP 2D/3D + Spatial 시각화</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


REQUIRED_FILES = {
    "cell_metadata": "cell_metadata.csv",
    "embeddings_spatial": "embeddings_spatial.csv",
    "embeddings_umap2d": "embeddings_umap2d.csv",
    "embeddings_umap3d": "embeddings_umap3d.csv",
    "gene_metadata": "gene_metadata.csv",
    "ground_truth_regulators": "ground_truth_regulators.csv"
}


@st.cache_data
def load_csv_safe(file_name):
    return pd.read_csv(file_name)


missing_files = [file for file in REQUIRED_FILES.values() if not os.path.exists(file)]

if missing_files:
    st.error("🚨 CSV 파일을 찾을 수 없습니다!")
    st.write("아래 파일들이 `main.py`와 같은 GitHub 폴더에 있어야 합니다.")
    st.code("\n".join(missing_files))
    st.stop()

try:
    cell_metadata = load_csv_safe(REQUIRED_FILES["cell_metadata"])
    spatial = load_csv_safe(REQUIRED_FILES["embeddings_spatial"])
    umap2d = load_csv_safe(REQUIRED_FILES["embeddings_umap2d"])
    umap3d = load_csv_safe(REQUIRED_FILES["embeddings_umap3d"])
    gene_metadata = load_csv_safe(REQUIRED_FILES["gene_metadata"])
    regulators = load_csv_safe(REQUIRED_FILES["ground_truth_regulators"])
except Exception as e:
    st.error("🚨 CSV 파일을 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


def find_id_column(df):
    candidates = ["cell_id", "cell", "Cell", "barcode", "id", "ID"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]


def merge_by_cell(base, emb):
    base_id = find_id_column(base)
    emb_id = find_id_column(emb)

    if base_id in base.columns and emb_id in emb.columns:
        return pd.merge(base, emb, left_on=base_id, right_on=emb_id, how="inner")

    n = min(len(base), len(emb))
    return pd.concat(
        [base.iloc[:n].reset_index(drop=True), emb.iloc[:n].reset_index(drop=True)],
        axis=1
    )


def numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(exclude=[np.number]).columns.tolist()

def friendly_cell_name(value):
    """데이터 안의 영어 세포 라벨을 일반인이 이해하기 쉬운 한국어 이름으로 변환"""
    if pd.isna(value):
        return "알 수 없음"

    raw = str(value)
    key = raw.strip().replace(" ", "_")

    name_map = {
        "NPC": "신경전구세포",
        "Neural_Progenitor": "신경전구세포",
        "Excitatory_Neuron": "흥분성 뉴런",
        "Inhibitory_Neuron": "억제성 뉴런",
        "Neuron": "뉴런",
        "Astrocyte": "별아교세포",
        "Oligodendrocyte": "희소돌기아교세포",
        "OPC": "희소돌기아교 전구세포",
        "Microglia": "미세아교세포",
    }
    return name_map.get(key, raw)


def friendly_cell_description(value):
    """세포 이름에 대한 쉬운 설명"""
    name = friendly_cell_name(value)

    desc_map = {
        "신경전구세포": "아직 완전히 성숙하지 않았고, 여러 종류의 뇌세포로 발달할 수 있는 준비 단계 세포입니다.",
        "흥분성 뉴런": "다른 세포에 신호를 보내 뇌 활동을 활발하게 만드는 신경세포입니다.",
        "억제성 뉴런": "신호가 과도하게 퍼지지 않도록 조절하는 브레이크 역할의 신경세포입니다.",
        "뉴런": "정보를 전달하고 처리하는 대표적인 뇌 신경세포입니다.",
        "별아교세포": "뉴런을 보호하고 영양 공급, 주변 환경 조절을 돕는 지원 세포입니다.",
        "희소돌기아교세포": "신경섬유를 감싸 신호가 빠르게 이동하도록 돕는 세포입니다.",
        "희소돌기아교 전구세포": "희소돌기아교세포로 발달할 수 있는 미성숙 단계의 세포입니다.",
        "미세아교세포": "뇌 안의 면역세포처럼 손상이나 염증을 감시하고 청소하는 세포입니다.",
    }
    return desc_map.get(name, "데이터의 메타데이터 라벨을 바탕으로 추정된 세포 그룹입니다.")


def broad_cell_group(value):
    """더 큰 범주의 세포 그룹명"""
    name = friendly_cell_name(value)
    if name in ["흥분성 뉴런", "억제성 뉴런", "뉴런"]:
        return "신경세포 그룹"
    if name in ["별아교세포", "희소돌기아교세포", "미세아교세포"]:
        return "아교세포 그룹"
    if name in ["신경전구세포", "희소돌기아교 전구세포"]:
        return "전구세포 그룹"
    return "기타 세포 그룹"


def best_label_column(df):
    """세포 유형 해석에 가장 적합한 열 자동 선택"""
    preferred = ["lineage", "cell_type", "celltype", "type", "label", "annotation", "class"]
    lower_map = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in lower_map:
            return lower_map[p.lower()]

    cats = categorical_columns(df)
    if len(cats) > 0:
        # ID처럼 종류가 너무 많은 열은 피하기
        reasonable = [c for c in cats if df[c].nunique() <= 30]
        if reasonable:
            return reasonable[0]
        return cats[0]
    return None


def friendly_numeric_meaning(col):
    """숫자형 열 이름을 쉬운 표현으로 바꾸기"""
    c = str(col).lower()
    if "pseudotime" in c:
        return "세포 발달 단계 점수"
    if "branch" in c:
        return "세포 발달 갈림길 점수"
    if "umap" in c and ("1" in c or "x" in c):
        return "UMAP 가로 위치"
    if "umap" in c and ("2" in c or "y" in c):
        return "UMAP 세로 위치"
    if "umap" in c and "3" in c:
        return "UMAP 입체 높이 위치"
    if c in ["x", "spatial_x", "coord_x"] or "spatial" in c and "x" in c:
        return "조직 안 가로 위치"
    if c in ["y", "spatial_y", "coord_y"] or "spatial" in c and "y" in c:
        return "조직 안 세로 위치"
    return str(col)


def regression_quality_text(r2, mae):
    if r2 >= 0.8:
        quality = "매우 잘 맞는 예측"
    elif r2 >= 0.5:
        quality = "어느 정도 설명력이 있는 예측"
    elif r2 >= 0.2:
        quality = "약한 예측"
    else:
        quality = "설명력이 낮은 예측"

    return (
        f"이 회귀모델은 '{quality}'입니다. "
        f"R²={r2:.3f}는 모델이 실제 값을 얼마나 잘 설명하는지 나타내며, 1에 가까울수록 좋습니다. "
        f"MAE={mae:.3f}는 예측값이 실제값과 평균적으로 얼마나 차이 나는지를 뜻합니다."
    )


def umap_easy_explanation(mode="2D"):
    if mode == "2D":
        return (
            "UMAP 2D는 복잡한 세포 데이터를 평면 지도처럼 펼친 그림입니다. "
            "점 하나가 세포 하나이고, 가까이 모인 점들은 서로 비슷한 세포일 가능성이 큽니다."
        )
    return (
        "UMAP 3D는 세포 사이의 관계를 입체 공간에서 보는 그림입니다. "
        "2D에서 겹쳐 보이던 세포 그룹도 3D에서는 더 잘 분리되어 보일 수 있습니다."
    )


def spatial_easy_explanation():
    return (
        "Spatial 시각화는 세포가 실제 조직 안에서 어느 위치에 있는지 보여주는 지도입니다. "
        "특정 색의 세포가 한곳에 몰려 있으면, 그 세포 그룹이 특정 영역에 많이 분포한다는 뜻입니다."
    )


def describe_gene_or_regulator_row(row):
    text = " / ".join([f"{c}: {row[c]}" for c in row.index[:4]])
    return text


def style_plotly(fig):
    fig.update_layout(
        paper_bgcolor="rgba(255,246,250,0.95)",
        plot_bgcolor="rgba(255,255,255,0.88)",
        font=dict(color="#333333", size=14),
        title_font=dict(size=22, color="#C2185B"),
        legend=dict(bgcolor="rgba(255,255,255,0.65)", bordercolor="#FFD1E4", borderwidth=1),
        margin=dict(l=40, r=40, t=70, b=40),
    )
    fig.update_xaxes(gridcolor="#F8BBD0", zerolinecolor="#F48FB1")
    fig.update_yaxes(gridcolor="#F8BBD0", zerolinecolor="#F48FB1")
    return fig


def style_umap2d_publication(fig, x_col, y_col):
    """논문 그림처럼 보이는 UMAP 2D 스타일"""
    fig.update_traces(
        marker=dict(
            size=5,
            opacity=0.72,
            line=dict(width=0)
        )
    )
    fig.update_layout(
        title=dict(
            text="2D Projection: embeddings_umap2d.csv",
            x=0.02,
            xanchor="left",
            font=dict(size=28, color="#111111")
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111", size=16),
        legend=dict(
            title="",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#DDDDDD",
            borderwidth=1,
            font=dict(size=13, color="#111111")
        ),
        margin=dict(l=70, r=40, t=80, b=70),
        height=720
    )
    fig.update_xaxes(
        title_text="UMAP_1",
        title_font=dict(size=24, color="#111111"),
        tickfont=dict(size=16, color="#111111"),
        showgrid=True,
        gridcolor="#DDDDDD",
        zeroline=False,
        showline=True,
        linewidth=2,
        linecolor="#111111",
        mirror=True
    )
    fig.update_yaxes(
        title_text="UMAP_2",
        title_font=dict(size=24, color="#111111"),
        tickfont=dict(size=16, color="#111111"),
        showgrid=True,
        gridcolor="#DDDDDD",
        zeroline=False,
        showline=True,
        linewidth=2,
        linecolor="#111111",
        mirror=True,
        scaleanchor="x",
        scaleratio=1
    )
    return fig


data2d = merge_by_cell(cell_metadata, umap2d)
data3d = merge_by_cell(cell_metadata, umap3d)
datasp = merge_by_cell(cell_metadata, spatial)

num2d = numeric_columns(data2d)
num3d = numeric_columns(data3d)
numsp = numeric_columns(datasp)
cat_cols = categorical_columns(cell_metadata)


with st.sidebar:
    st.markdown("## 🧠 분석 메뉴")
    st.markdown("### 🖼️ 뇌세포 이미지 업로드")
    uploaded_image = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    st.markdown("### 🔍 분석 방식 선택")
    analysis_mode = st.radio(
        "",
        [
            "🎨 K-Means 군집 분석",
            "🌳 의사결정트리 분류",
            "📈 회귀 분석",
            "🌈 UMAP 2D 시각화",
            "🧊 UMAP 3D 시각화",
            "📍 Spatial 시각화",
            "🧬 유전자/조절인자 탐색"
        ],
        label_visibility="collapsed"
    )

    st.success("✅ 데이터 로드 완료")


col1, col2, col3 = st.columns(3)
col1.metric("🌐 세포 메타데이터", f"{len(cell_metadata):,}개")
col2.metric("🧬 유전자 메타데이터", f"{len(gene_metadata):,}개")
col3.metric("🎯 조절인자 데이터", f"{len(regulators):,}개")

if uploaded_image is not None:
    st.markdown("### 🖼️ 업로드한 뇌세포 이미지")
    st.image(uploaded_image, use_container_width=True)

    st.success("🔍 참고 이미지 업로드 완료!")
    st.info(
        "🤖 이 앱은 업로드한 이미지를 직접 판별하는 딥러닝 모델은 아닙니다. "
        "대신 업로드한 이미지를 참고 자료로 보여주고, CSV에 들어 있는 뇌세포 데이터를 바탕으로 AI 분석을 실행합니다."
    )

    st.markdown("### 📋 CSV 기반 AI 분석 결과 요약")

    # 일반인이 이해하기 쉬운 대표 세포 그룹 자동 요약
    label_col_preview = best_label_column(data2d)
    if label_col_preview:
        easy_counts = data2d[label_col_preview].dropna().apply(friendly_cell_name).value_counts().head(5).reset_index()
        easy_counts.columns = ["세포 이름", "데이터 안의 세포 수"]
        easy_counts["쉬운 설명"] = easy_counts["세포 이름"].apply(friendly_cell_description)

        st.write("🧠 현재 데이터에서 많이 관찰되는 대표 뇌세포 그룹입니다.")
        st.dataframe(easy_counts, use_container_width=True)

        top_cell = easy_counts.iloc[0]["세포 이름"]
        st.success(
            f"🌟 현재 CSV 데이터 기준으로 가장 많이 보이는 세포 그룹은 **{top_cell}** 입니다. "
            "왼쪽 메뉴에서 K-Means, 의사결정트리, UMAP, Spatial 분석을 선택하면 더 자세한 결과를 볼 수 있습니다."
        )
    else:
        st.warning("세포 이름으로 해석할 수 있는 메타데이터 열을 찾지 못했습니다. 그래도 UMAP/Spatial 좌표 기반 분석은 가능합니다.")

    st.markdown("""
    #### 🧭 이 앱에서 확인할 수 있는 것
    - 🎨 **K-Means 군집 분석**: 비슷한 세포끼리 자동으로 묶고, 쉬운 세포 이름으로 해석
    - 🌳 **의사결정트리 분류**: 세포가 어떤 종류인지 예측하고 쉬운 설명 출력
    - 📈 **회귀 분석**: 세포 발달 단계나 숫자형 특징을 예측
    - 🌈 **UMAP 2D/3D**: 비슷한 세포들이 지도 위에서 어디에 모이는지 확인
    - 📍 **Spatial 시각화**: 세포가 조직 안에서 어느 위치에 있는지 확인
    - 🧬 **유전자/조절인자 탐색**: 세포 특징과 관련된 유전자 정보 확인
    """)

else:
    st.warning("🖼️ 왼쪽 사이드바에서 뇌세포 이미지를 업로드하면 참고 이미지와 CSV 기반 AI 분석 요약이 표시됩니다.")

st.markdown("<hr>", unsafe_allow_html=True)


if analysis_mode == "🎨 K-Means 군집 분석":
    st.markdown('<div class="section-title">🎨 K-Means 군집 분석</div>', unsafe_allow_html=True)

    if len(num2d) < 2:
        st.error("숫자형 좌표 열이 2개 이상 필요합니다.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        x_col = st.selectbox("X축 선택", num2d, index=0)
    with c2:
        y_col = st.selectbox("Y축 선택", num2d, index=1)
    with c3:
        k = st.selectbox("군집 개수 K 선택", list(range(2, 11)), index=2)

    label_col_auto = best_label_column(data2d)
    label_candidates = categorical_columns(data2d)
    label_candidates = [c for c in label_candidates if c not in [x_col, y_col]]

    if label_col_auto and label_col_auto in label_candidates:
        default_idx = label_candidates.index(label_col_auto)
    else:
        default_idx = 0 if len(label_candidates) > 0 else None

    if len(label_candidates) > 0:
        label_col = st.selectbox(
            "🧠 군집 결과를 어떤 세포 정보로 해석할까요?",
            label_candidates,
            index=default_idx,
            help="추천: lineage. K-Means는 좌표상 가까운 세포를 묶고, 선택한 메타데이터 열을 이용해 각 군집을 쉬운 세포 이름으로 해석합니다."
        )
    else:
        label_col = None
        st.warning("해석에 사용할 문자형 메타데이터 열이 없어 cluster 번호만 출력합니다.")

    use_cols = [x_col, y_col] + ([label_col] if label_col else [])
    df_kmeans = data2d[use_cols].dropna()

    X = df_kmeans[[x_col, y_col]]
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    plot_df = df_kmeans.copy()
    plot_df["cluster"] = labels.astype(str)

    if label_col:
        summary = (
            plot_df.groupby("cluster")[label_col]
            .agg(lambda s: s.value_counts().index[0])
            .reset_index()
            .rename(columns={label_col: "대표 원본 라벨"})
        )

        counts = plot_df.groupby("cluster").size().reset_index(name="세포 수")
        purity = (
            plot_df.groupby("cluster")[label_col]
            .agg(lambda s: round(s.value_counts(normalize=True).iloc[0] * 100, 1))
            .reset_index()
            .rename(columns={label_col: "대표 유형 비율(%)"})
        )

        cluster_result = counts.merge(summary, on="cluster").merge(purity, on="cluster")
        cluster_result["예측 세포 이름"] = cluster_result["대표 원본 라벨"].apply(friendly_cell_name)
        cluster_result["쉬운 세포 그룹"] = cluster_result["대표 원본 라벨"].apply(broad_cell_group)
        cluster_result["쉬운 설명"] = cluster_result["대표 원본 라벨"].apply(friendly_cell_description)

        cluster_name_map = dict(zip(cluster_result["cluster"], cluster_result["예측 세포 이름"]))
        cluster_group_map = dict(zip(cluster_result["cluster"], cluster_result["쉬운 세포 그룹"]))
        cluster_desc_map = dict(zip(cluster_result["cluster"], cluster_result["쉬운 설명"]))

        plot_df["예측 세포 이름"] = plot_df["cluster"].map(cluster_name_map)
        plot_df["쉬운 세포 그룹"] = plot_df["cluster"].map(cluster_group_map)
        plot_df["쉬운 설명"] = plot_df["cluster"].map(cluster_desc_map)
        color_target = "예측 세포 이름"
    else:
        cluster_result = plot_df["cluster"].value_counts().sort_index().reset_index()
        cluster_result.columns = ["cluster", "세포 수"]
        color_target = "cluster"

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_target,
        hover_data=["cluster"] + ([label_col, "예측 세포 이름", "쉬운 세포 그룹"] if label_col else []),
        title=f"🎨 K-Means 군집 결과: K={k}",
        template="plotly_white",
        opacity=0.82
    )
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 군집 결과를 쉬운 이름으로 해석")
    st.info(
        "K-Means는 먼저 좌표상 가까운 세포끼리 자동으로 묶습니다. "
        "그다음 각 cluster 안에서 가장 많이 등장한 세포 라벨을 찾아서 "
        "일반인이 이해하기 쉬운 이름으로 바꿔 표시합니다."
    )
    st.dataframe(cluster_result, use_container_width=True)

    if label_col:
        st.markdown("### 🔎 세포별 군집 분류 결과")
        output_cols = [x_col, y_col, "cluster", "예측 세포 이름", "쉬운 세포 그룹", "쉬운 설명", label_col]
        st.dataframe(plot_df[output_cols].head(200), use_container_width=True)

        csv = plot_df[output_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 군집 분류 결과 CSV 다운로드",
            data=csv,
            file_name="kmeans_cell_classification_result.csv",
            mime="text/csv"
        )


elif analysis_mode == "🌳 의사결정트리 분류":
    st.markdown('<div class="section-title">🌳 의사결정트리 분류</div>', unsafe_allow_html=True)

    if len(cat_cols) == 0:
        st.warning("분류에 사용할 문자형 라벨 열이 없습니다.")
        st.stop()

    target_col = st.selectbox("예측할 세포 라벨 선택", cat_cols)

    feature_cols = numeric_columns(data2d)
    if len(feature_cols) < 2:
        st.error("분류에 사용할 숫자형 특성이 부족합니다.")
        st.stop()

    selected_features = st.multiselect(
        "학습에 사용할 숫자 특성 선택",
        feature_cols,
        default=feature_cols[:min(3, len(feature_cols))]
    )

    if len(selected_features) < 1:
        st.warning("특성을 1개 이상 선택하세요.")
        st.stop()

    df = data2d[selected_features + [target_col]].dropna()

    if df[target_col].nunique() < 2:
        st.error("분류하려면 라벨 종류가 2개 이상이어야 합니다.")
        st.stop()

    le = LabelEncoder()
    y = le.fit_transform(df[target_col].astype(str))
    X = df[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    st.metric("🌟 분류 정확도", f"{acc:.3f}")

    result_df = X_test.copy()
    result_df["실제 원본 라벨"] = le.inverse_transform(y_test)
    result_df["예측 원본 라벨"] = le.inverse_transform(pred)
    result_df["실제 세포 이름"] = result_df["실제 원본 라벨"].apply(friendly_cell_name)
    result_df["예측 세포 이름"] = result_df["예측 원본 라벨"].apply(friendly_cell_name)
    result_df["예측 세포 그룹"] = result_df["예측 원본 라벨"].apply(broad_cell_group)
    result_df["쉬운 설명"] = result_df["예측 원본 라벨"].apply(friendly_cell_description)

    st.markdown("### 🧠 의사결정트리 세포 예측 결과")
    st.info(
        "의사결정트리는 이미 알려진 세포 라벨을 학습한 뒤, 새로운 세포가 어떤 세포인지 예측합니다. "
        "아래 표에서는 영어 라벨 대신 일반인이 이해하기 쉬운 세포 이름과 설명을 함께 표시합니다."
    )
    show_cols = ["실제 세포 이름", "예측 세포 이름", "예측 세포 그룹", "쉬운 설명", "실제 원본 라벨", "예측 원본 라벨"] + selected_features
    st.dataframe(result_df[show_cols].head(80), use_container_width=True)

    pred_summary = (
        result_df["예측 세포 이름"]
        .value_counts()
        .reset_index()
    )
    pred_summary.columns = ["예측 세포 이름", "예측 개수"]
    pred_summary["세포 그룹"] = pred_summary["예측 세포 이름"].apply(broad_cell_group)
    pred_summary["쉬운 설명"] = pred_summary["예측 세포 이름"].apply(friendly_cell_description)

    st.markdown("### 📊 예측된 세포 종류 요약")
    st.dataframe(pred_summary, use_container_width=True)

    importance = pd.DataFrame({
        "feature": selected_features,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)

    fig = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        title="🌳 의사결정트리 특성 중요도",
        template="plotly_white"
    )
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)


elif analysis_mode == "📈 회귀 분석":
    st.markdown('<div class="section-title">📈 회귀 분석</div>', unsafe_allow_html=True)

    all_numeric = numeric_columns(data2d)

    if len(all_numeric) < 2:
        st.error("회귀 분석에는 숫자형 열이 2개 이상 필요합니다.")
        st.stop()

    target = st.selectbox("예측할 숫자값 선택", all_numeric, index=len(all_numeric)-1)
    features = st.multiselect(
        "입력 특성 선택",
        [c for c in all_numeric if c != target],
        default=[c for c in all_numeric if c != target][:min(3, len(all_numeric)-1)]
    )

    if len(features) < 1:
        st.warning("입력 특성을 1개 이상 선택하세요.")
        st.stop()

    df = data2d[features + [target]].dropna()

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    col1, col2 = st.columns(2)
    col1.metric("📊 R² 결정계수", f"{r2:.3f}")
    col2.metric("📉 MAE 평균절대오차", f"{mae:.3f}")

    st.markdown("### 🧠 회귀 분석 결과를 쉽게 해석하기")
    st.info(regression_quality_text(r2, mae))
    st.write(f"🎯 예측하려는 값: **{friendly_numeric_meaning(target)}** (`{target}`)")
    st.write("📌 입력으로 사용한 정보: " + ", ".join([f"**{friendly_numeric_meaning(c)}** (`{c}`)" for c in features]))

    result = pd.DataFrame({
        "실제값": y_test,
        "예측값": pred
    })
    result["차이"] = (result["실제값"] - result["예측값"]).abs()
    result["쉬운 해석"] = result["차이"].apply(
        lambda v: "예측이 실제값과 매우 비슷함" if v <= mae else "평균보다 차이가 큰 예측"
    )

    fig = px.scatter(
        result,
        x="실제값",
        y="예측값",
        title="📈 실제값 vs 예측값",
        template="plotly_white",
        hover_data=["차이", "쉬운 해석"]
    )
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 세포별 예측 결과")
    st.dataframe(result.head(80), use_container_width=True)


elif analysis_mode == "🌈 UMAP 2D 시각화":
    st.markdown('<div class="section-title">🌈 UMAP 2D 시각화</div>', unsafe_allow_html=True)
    st.info("💡 " + umap_easy_explanation("2D"))
    st.caption("가까운 점끼리는 비슷한 세포입니다. 색이 다르게 나뉘면 서로 다른 세포 그룹으로 볼 수 있습니다.")

    if len(num2d) < 2:
        st.error("UMAP 2D 좌표 열이 부족합니다.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        x_col = st.selectbox("UMAP_1로 사용할 열", num2d, index=0)
    with c2:
        y_col = st.selectbox("UMAP_2로 사용할 열", num2d, index=1)
    with c3:
        color_options = ["K-Means 자동 군집"] + list(data2d.columns)
        color_col = st.selectbox("색상 기준", color_options)

    plot_df = data2d[[x_col, y_col]].copy()

    if color_col == "K-Means 자동 군집":
        k_umap = st.slider("🎨 UMAP 색상 군집 수", 2, 12, 8)
        X_umap = plot_df[[x_col, y_col]].dropna()
        km = KMeans(n_clusters=k_umap, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(X_umap)

        plot_df = X_umap.copy()
        plot_df["cluster"] = cluster_labels.astype(str)
        color_target = "cluster"
    else:
        temp = data2d[[x_col, y_col, color_col]].dropna()
        plot_df = temp.copy()
        color_target = color_col

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_target,
        title="2D Projection: embeddings_umap2d.csv",
        template="plotly_white",
        opacity=0.72,
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig = style_umap2d_publication(fig, x_col, y_col)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧬 UMAP 2D 결과 해석")
    if color_col == "K-Means 자동 군집":
        umap_summary = plot_df["cluster"].value_counts().sort_index().reset_index()
        umap_summary.columns = ["그룹 번호", "세포 수"]
        umap_summary["쉬운 해석"] = umap_summary["그룹 번호"].apply(
            lambda x: f"UMAP 지도에서 서로 가까운 세포들의 {x}번 그룹입니다."
        )
        st.dataframe(umap_summary, use_container_width=True)
    else:
        st.write(f"색상은 **{color_col}** 기준입니다. 같은 색으로 모인 점들은 비슷한 특징을 가진 세포로 해석할 수 있습니다.")

    st.markdown("### 🧬 UMAP 2D 데이터 미리보기")
    st.dataframe(plot_df.head(100), use_container_width=True)


elif analysis_mode == "🧊 UMAP 3D 시각화":
    st.markdown('<div class="section-title">🧊 UMAP 3D 시각화</div>', unsafe_allow_html=True)
    st.info("💡 " + umap_easy_explanation("3D"))

    if len(num3d) < 3:
        st.error("UMAP 3D 좌표 열이 부족합니다.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_col = st.selectbox("3D X축", num3d, index=0)
    with c2:
        y_col = st.selectbox("3D Y축", num3d, index=1)
    with c3:
        z_col = st.selectbox("3D Z축", num3d, index=2)
    with c4:
        color_options = ["없음"] + list(data3d.columns)
        color_col = st.selectbox("색상 기준", color_options)

    fig = px.scatter_3d(
        data3d,
        x=x_col,
        y=y_col,
        z=z_col,
        color=None if color_col == "없음" else color_col,
        title="🧊 UMAP 3D 뇌세포 공간",
        template="plotly_white",
        opacity=0.75
    )
    fig.update_layout(
        paper_bgcolor="rgba(255,246,250,0.95)",
        scene=dict(
            bgcolor="rgba(255,255,255,0.85)",
            xaxis=dict(backgroundcolor="#FFF7FB", gridcolor="#F8BBD0"),
            yaxis=dict(backgroundcolor="#FFF7FB", gridcolor="#F8BBD0"),
            zaxis=dict(backgroundcolor="#FFF7FB", gridcolor="#F8BBD0"),
        ),
        font=dict(color="#333333"),
        title_font=dict(color="#C2185B", size=22),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 UMAP 3D 결과 해석")
    st.write("점 하나는 세포 하나입니다. 입체 공간에서 가까운 점들은 비슷한 세포이고, 멀리 떨어진 점들은 서로 다른 특성을 가진 세포일 가능성이 큽니다.")
    if color_col != "없음":
        st.write(f"현재 색상은 **{color_col}** 기준입니다. 같은 색이 한 공간에 모이면 그 세포들이 비슷한 그룹을 이룬다는 뜻입니다.")

    st.markdown("### 🧬 UMAP 3D 데이터 미리보기")
    st.dataframe(data3d.head(100), use_container_width=True)


elif analysis_mode == "📍 Spatial 시각화":
    st.markdown('<div class="section-title">📍 Spatial 시각화</div>', unsafe_allow_html=True)
    st.info("💡 " + spatial_easy_explanation())

    if len(numsp) < 2:
        st.error("Spatial 좌표 열이 부족합니다.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        x_col = st.selectbox("Spatial X축", numsp, index=0)
    with c2:
        y_col = st.selectbox("Spatial Y축", numsp, index=1)
    with c3:
        color_options = ["없음"] + list(datasp.columns)
        color_col = st.selectbox("색상 기준", color_options)

    fig = px.scatter(
        datasp,
        x=x_col,
        y=y_col,
        color=None if color_col == "없음" else color_col,
        title="📍 Spatial 뇌세포 위치 지도",
        template="plotly_white",
        opacity=0.8
    )
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📍 Spatial 결과 해석")
    st.write("이 그림은 세포가 조직 안에서 어디에 위치하는지 보여줍니다.")
    if color_col != "없음":
        st.write(f"현재 색상은 **{color_col}** 기준입니다. 같은 색이 특정 영역에 몰려 있으면 그 세포 그룹이 그 위치에 많이 존재한다는 뜻입니다.")
    st.write("예를 들어 특정 세포가 한쪽 영역에만 많이 모여 있다면, 그 영역에서 중요한 역할을 할 가능성이 있습니다.")

    st.markdown("### 🧬 Spatial 데이터 미리보기")
    st.dataframe(datasp.head(100), use_container_width=True)


elif analysis_mode == "🧬 유전자/조절인자 탐색":
    st.markdown('<div class="section-title">🧬 유전자 / 조절인자 탐색</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🧬 Gene Metadata", "🎯 Regulators"])

    with tab1:
        st.write("### 🧬 유전자 메타데이터")
        st.info("유전자 메타데이터는 각 유전자가 어떤 이름과 특징을 가지는지 정리한 표입니다. 세포의 성격을 이해하는 데 사용됩니다.")

        gene_view = gene_metadata.copy()
        gene_view["쉬운 설명"] = "이 유전자는 세포 특징을 설명하는 데 사용될 수 있는 후보 정보입니다."
        st.dataframe(gene_view.head(300), use_container_width=True)

        if len(gene_metadata.columns) > 0:
            search_gene = st.text_input("🔎 유전자 검색어 입력")
            if search_gene:
                mask = gene_metadata.astype(str).apply(
                    lambda row: row.str.contains(search_gene, case=False, na=False).any(),
                    axis=1
                )
                result_gene = gene_metadata[mask].copy()
                result_gene["쉬운 해석"] = "검색한 유전자와 관련된 정보입니다. 특정 세포 유형을 구분하는 단서가 될 수 있습니다."
                st.markdown("### 🔍 검색된 유전자 결과")
                st.dataframe(result_gene, use_container_width=True)

    with tab2:
        st.write("### 🎯 조절인자 탐색")
        st.info("조절인자는 다른 유전자의 활동을 켜거나 끄는 스위치 같은 역할을 합니다. 어떤 세포가 어떤 방향으로 발달하는지 이해하는 데 중요합니다.")

        reg_view = regulators.copy()
        reg_view["쉬운 설명"] = "이 조절인자는 세포 발달이나 세포 유형 결정에 영향을 줄 수 있는 후보입니다."
        st.dataframe(reg_view.head(300), use_container_width=True)

        if len(regulators.columns) > 0:
            search_reg = st.text_input("🔎 조절인자 검색어 입력")
            if search_reg:
                mask = regulators.astype(str).apply(
                    lambda row: row.str.contains(search_reg, case=False, na=False).any(),
                    axis=1
                )
                result_reg = regulators[mask].copy()
                result_reg["쉬운 해석"] = "검색한 조절인자는 세포의 특징이나 발달 방향을 결정하는 데 관여할 수 있습니다."
                st.markdown("### 🔍 검색된 조절인자 결과")
                st.dataframe(result_reg, use_container_width=True)


st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🧠 Brain Cell AI Explorer | CSV 기반 뇌세포 선별 및 시각화 Streamlit App")
