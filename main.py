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
    st.info("💡 현재 이미지는 분석 화면에 표시용으로 사용됩니다. 실제 선별은 CSV 데이터 기반으로 수행됩니다.")

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

    X = data2d[[x_col, y_col]].dropna()
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    plot_df = X.copy()
    plot_df["cluster"] = labels.astype(str)

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color="cluster",
        title=f"🎨 K-Means 군집 결과: K={k}",
        template="plotly_white",
        opacity=0.82
    )
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🏆 군집별 세포 수")
    cluster_count = plot_df["cluster"].value_counts().sort_index().reset_index()
    cluster_count.columns = ["cluster", "count"]
    st.dataframe(cluster_count, use_container_width=True)


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
    result_df["실제 라벨"] = le.inverse_transform(y_test)
    result_df["예측 라벨"] = le.inverse_transform(pred)

    st.markdown("### 🧠 세포 분류 결과")
    st.dataframe(result_df.head(50), use_container_width=True)

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

    result = pd.DataFrame({
        "실제값": y_test,
        "예측값": pred
    })

    fig = px.scatter(
        result,
        x="실제값",
        y="예측값",
        title="📈 실제값 vs 예측값",
        template="plotly_white"
    )
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(result.head(50), use_container_width=True)


elif analysis_mode == "🌈 UMAP 2D 시각화":
    st.markdown('<div class="section-title">🌈 UMAP 2D 시각화</div>', unsafe_allow_html=True)

    if len(num2d) < 2:
        st.error("UMAP 2D 좌표 열이 부족합니다.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        x_col = st.selectbox("UMAP X축", num2d, index=0)
    with c2:
        y_col = st.selectbox("UMAP Y축", num2d, index=1)
    with c3:
        color_options = ["없음"] + list(data2d.columns)
        color_col = st.selectbox("색상 기준", color_options)

    fig = px.scatter(
        data2d,
        x=x_col,
        y=y_col,
        color=None if color_col == "없음" else color_col,
        title="🌈 UMAP 2D 뇌세포 지도",
        template="plotly_white",
        opacity=0.8
    )
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(data2d.head(100), use_container_width=True)


elif analysis_mode == "🧊 UMAP 3D 시각화":
    st.markdown('<div class="section-title">🧊 UMAP 3D 시각화</div>', unsafe_allow_html=True)

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

    st.dataframe(data3d.head(100), use_container_width=True)


elif analysis_mode == "📍 Spatial 시각화":
    st.markdown('<div class="section-title">📍 Spatial 시각화</div>', unsafe_allow_html=True)

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

    st.dataframe(datasp.head(100), use_container_width=True)


elif analysis_mode == "🧬 유전자/조절인자 탐색":
    st.markdown('<div class="section-title">🧬 유전자 / 조절인자 탐색</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🧬 Gene Metadata", "🎯 Regulators"])

    with tab1:
        st.write("### 🧬 유전자 메타데이터")
        st.dataframe(gene_metadata.head(300), use_container_width=True)

        if len(gene_metadata.columns) > 0:
            search_gene = st.text_input("🔎 유전자 검색어 입력")
            if search_gene:
                mask = gene_metadata.astype(str).apply(
                    lambda row: row.str.contains(search_gene, case=False, na=False).any(),
                    axis=1
                )
                st.dataframe(gene_metadata[mask], use_container_width=True)

    with tab2:
        st.write("### 🎯 Ground Truth Regulators")
        st.dataframe(regulators.head(300), use_container_width=True)

        if len(regulators.columns) > 0:
            search_reg = st.text_input("🔎 조절인자 검색어 입력")
            if search_reg:
                mask = regulators.astype(str).apply(
                    lambda row: row.str.contains(search_reg, case=False, na=False).any(),
                    axis=1
                )
                st.dataframe(regulators[mask], use_container_width=True)


st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🧠 Brain Cell AI Explorer | CSV 기반 뇌세포 선별 및 시각화 Streamlit App")
