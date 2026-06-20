
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


st.set_page_config(
    page_title="🧠 Brain Cell AI Explorer",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #FFF1F7 0%, #FFF7FB 45%, #FFF0F6 100%) !important;
}
.main .block-container {
    background: rgba(255, 246, 250, 0.95) !important;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    border-radius: 24px;
}
[data-testid="stHeader"] { background: rgba(255,255,255,0) !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFD6E8 0%, #FFEAF3 100%) !important;
    border-right: 2px solid #FFC1DC;
}
[data-testid="stSidebar"] * { color: #333333 !important; }

.big-title {
    font-size: 52px;
    font-weight: 1000;
    text-align: center;
    color: #EC407A !important;
    text-shadow: 2px 2px 0px rgba(255,255,255,0.9);
}
.sub-title {
    text-align: center;
    font-size: 22px;
    color: #333333 !important;
    font-weight: 800;
    margin-bottom: 2rem;
}
html, body, p, div, span, label, h1, h2, h3, h4, h5, h6 {
    color: #2F2F2F !important;
}
hr { border: none; border-top: 2px solid #FFC1DC; }

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.92) !important;
    padding: 22px;
    border-radius: 22px;
    border: 1.5px solid #FFD1E4;
    box-shadow: 0 8px 22px rgba(236, 64, 122, 0.13);
}
[data-testid="stMetricLabel"] {
    color: #333333 !important;
    font-weight: 800 !important;
    font-size: 17px !important;
}
[data-testid="stMetricValue"] {
    color: #EC407A !important;
    font-size: 34px !important;
    font-weight: 1000 !important;
}

/* 하늘색 입력창 */
div[data-baseweb="select"] > div {
    background: linear-gradient(180deg, #EAF7FF 0%, #DFF2FF 100%) !important;
    border: 2px solid #9BD4FF !important;
    border-radius: 14px !important;
    color: #1F2937 !important;
    min-height: 54px !important;
}
div[data-baseweb="select"] span, div[data-baseweb="select"] input {
    color: #1F2937 !important;
    font-weight: 700 !important;
}
div[data-baseweb="select"] svg { fill: #1F2937 !important; }
div[role="listbox"], ul[role="listbox"] {
    background: #EAF7FF !important;
    border: 1px solid #9BD4FF !important;
}
div[role="option"] {
    background: #EAF7FF !important;
    color: #1F2937 !important;
}
[data-baseweb="tag"] {
    background-color: #BDE7FF !important;
    color: #1F2937 !important;
}
[data-testid="stFileUploader"] {
    background: linear-gradient(180deg, #EAF7FF 0%, #DFF2FF 100%) !important;
    border: 2px solid #9BD4FF !important;
    border-radius: 16px !important;
    padding: 14px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #EAF7FF !important;
    border: 2px dashed #8CCEFF !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] button {
    background: #EAF7FF !important;
    color: #1976D2 !important;
    border: 2px solid #8CCEFF !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #FF69A6 0%, #FF8FC4 100%) !important;
    color: white !important;
    border-radius: 14px !important;
    border: none !important;
    font-weight: 900 !important;
}
.stRadio label { color: #333333 !important; font-weight: 700 !important; }

.section-title {
    font-size: 34px;
    font-weight: 1000;
    color: #333333 !important;
    margin-top: 20px;
    margin-bottom: 14px;
}
.result-card {
    background: rgba(255,255,255,0.92);
    border: 1.5px solid #FFD1E4;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 8px 22px rgba(236, 64, 122, 0.12);
    margin-bottom: 18px;
}
.sky-card {
    background: #EAF7FF;
    border: 2px solid #9BD4FF;
    border-radius: 20px;
    padding: 18px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧠✨ 뇌세포 이미지 기반 CSV AI 분석기 ✨🧬</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">이미지 업로드 → CSV 기반 후보 세포 선택 → 항목별 예측 결과와 근거 출력</div>', unsafe_allow_html=True)
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
    candidates = ["cell_id", "cell", "Cell", "barcode", "id", "ID", "Unnamed: 0"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]


def merge_by_cell(base, emb):
    base_id = find_id_column(base)
    emb_id = find_id_column(emb)

    if base_id in base.columns and emb_id in emb.columns:
        return pd.merge(base, emb, left_on=base_id, right_on=emb_id, how="inner", suffixes=("", "_emb"))

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
    name = friendly_cell_name(value)
    desc_map = {
        "신경전구세포": "여러 종류의 뇌세포로 발달할 수 있는 준비 단계의 세포입니다.",
        "흥분성 뉴런": "다른 신경세포에 신호를 보내 뇌 활동을 활발하게 만드는 신경세포입니다.",
        "억제성 뉴런": "신경 신호가 과도하게 퍼지지 않도록 조절하는 브레이크 역할의 신경세포입니다.",
        "뉴런": "정보를 전달하고 처리하는 대표적인 뇌 신경세포입니다.",
        "별아교세포": "뉴런을 보호하고 영양 공급과 주변 환경 조절을 돕는 지원 세포입니다.",
        "희소돌기아교세포": "신경섬유를 감싸 신호가 빠르게 이동하도록 돕는 세포입니다.",
        "희소돌기아교 전구세포": "희소돌기아교세포로 발달할 수 있는 미성숙 단계의 세포입니다.",
        "미세아교세포": "뇌 안의 면역세포처럼 손상이나 염증을 감시하고 청소하는 세포입니다.",
    }
    return desc_map.get(name, "CSV 데이터의 메타데이터 라벨을 바탕으로 추정된 세포 그룹입니다.")


def broad_cell_group(value):
    name = friendly_cell_name(value)
    if name in ["흥분성 뉴런", "억제성 뉴런", "뉴런"]:
        return "신경세포 그룹"
    if name in ["별아교세포", "희소돌기아교세포", "미세아교세포"]:
        return "아교세포 그룹"
    if name in ["신경전구세포", "희소돌기아교 전구세포"]:
        return "전구세포 그룹"
    return "기타 세포 그룹"


def best_label_column(df):
    preferred = ["lineage", "cell_type", "celltype", "type", "label", "annotation", "class"]
    lower_map = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in lower_map:
            return lower_map[p.lower()]

    cats = categorical_columns(df)
    if len(cats) > 0:
        reasonable = [c for c in cats if df[c].nunique() <= 30]
        if reasonable:
            return reasonable[0]
        return cats[0]
    return None


def friendly_numeric_meaning(col):
    c = str(col).lower()
    if "pseudotime" in c:
        return "세포 발달 단계 점수"
    if "branch_gate" in c:
        return "세포 발달 방향 전환 점수"
    if "branch_index" in c:
        return "세포 발달 가지 번호"
    if "library" in c:
        return "측정된 유전자 양 보정값"
    if "cell_cycle" in c:
        return "세포분열 활동 점수"
    if "stress" in c:
        return "세포 스트레스 점수"
    if "spatial" in c and "x" in c:
        return "조직 안 가로 위치"
    if "spatial" in c and "y" in c:
        return "조직 안 세로 위치"
    if "umap" in c and ("1" in c or "x" in c):
        return "UMAP 가로 위치"
    if "umap" in c and ("2" in c or "y" in c):
        return "UMAP 세로 위치"
    if "umap" in c and "3" in c:
        return "UMAP 입체 높이"
    return str(col)


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


def extract_image_features(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    arr = np.asarray(image).astype(float) / 255.0

    brightness = float(arr.mean())
    contrast = float(arr.std())
    red = float(arr[:, :, 0].mean())
    green = float(arr[:, :, 1].mean())
    blue = float(arr[:, :, 2].mean())

    return image, {
        "밝기": brightness,
        "대비": contrast,
        "빨강 평균": red,
        "초록 평균": green,
        "파랑 평균": blue,
    }


def choose_candidate_cell_by_image(image_features, df):
    """이미지의 밝기/색상/대비를 CSV 숫자 특징과 맞춰 가장 가까운 참고 세포를 선택"""
    num_cols = numeric_columns(df)
    safe_cols = [c for c in num_cols if df[c].notna().sum() > 0]

    preferred = [
        "pseudotime",
        "branch_gate",
        "library_size_factor",
        "cell_cycle_score",
        "stress_score",
        "spatial_x",
        "spatial_y",
        "UMAP_1",
        "UMAP_2",
    ]
    feature_cols = [c for c in preferred if c in safe_cols]
    if len(feature_cols) < 3:
        feature_cols = safe_cols[:min(5, len(safe_cols))]

    values = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].median(numeric_only=True))
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    img_vec = np.array([
        image_features["밝기"],
        image_features["대비"],
        image_features["빨강 평균"],
        image_features["초록 평균"],
        image_features["파랑 평균"],
    ])

    if scaled.shape[1] < len(img_vec):
        img_vec = img_vec[:scaled.shape[1]]
    elif scaled.shape[1] > len(img_vec):
        img_vec = np.pad(img_vec, (0, scaled.shape[1] - len(img_vec)), constant_values=image_features["밝기"])

    distances = np.linalg.norm(scaled - img_vec, axis=1)
    idx = int(np.argmin(distances))
    confidence = max(0.0, min(99.0, (1 - distances[idx] / (np.sqrt(len(img_vec)) + 1e-9)) * 100))

    return idx, feature_cols, float(distances[idx]), confidence


def get_candidate_label(df, idx):
    label_col = best_label_column(df)
    if label_col and label_col in df.columns:
        raw = df.iloc[idx][label_col]
        return label_col, raw, friendly_cell_name(raw), friendly_cell_description(raw), broad_cell_group(raw)
    return None, "알 수 없음", "알 수 없음", "세포 라벨 열을 찾지 못했습니다.", "알 수 없음"


data2d = merge_by_cell(cell_metadata, umap2d)
data3d = merge_by_cell(cell_metadata, umap3d)
datasp = merge_by_cell(cell_metadata, spatial)
datareg = merge_by_cell(cell_metadata, regulators)

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

if uploaded_image is None:
    st.warning("🖼️ 왼쪽 사이드바에서 뇌세포 이미지를 먼저 업로드하세요.")
    st.info("이미지를 업로드하면 CSV 데이터와 비교해 참고 세포 후보를 고르고, 선택한 분석 항목별로 예측 결과와 근거를 보여줍니다.")
    st.stop()

image, image_features = extract_image_features(uploaded_image)
candidate_idx, candidate_basis_cols, candidate_distance, candidate_confidence = choose_candidate_cell_by_image(image_features, data2d)
label_col, raw_label, predicted_name, predicted_desc, predicted_group = get_candidate_label(data2d, candidate_idx)

st.markdown("### 🖼️ 업로드한 뇌세포 이미지")
left, right = st.columns([1, 1.2])
with left:
    st.image(image, use_container_width=True)
with right:
    st.markdown('<div class="sky-card">', unsafe_allow_html=True)
    st.markdown("### 🔮 이미지 기반 CSV 참고 예측")
    st.success(f"예상 세포 그룹: **{predicted_name}**")
    st.write(f"🧩 큰 분류: **{predicted_group}**")
    st.write(f"📖 설명: {predicted_desc}")
    st.write(f"📌 근거: 이미지의 밝기·대비·색상 평균을 CSV의 숫자형 세포 특징과 비교해 가장 가까운 참고 세포를 선택했습니다.")
    st.write(f"📊 참고 신뢰도: **{candidate_confidence:.1f}%**")
    st.caption("주의: 이 앱은 실제 현미경 이미지를 학습한 딥러닝 모델이 아니라, 업로드 이미지와 CSV 특징값을 연결한 교육용 참고 예측입니다.")
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("🔍 이미지에서 추출한 간단한 특징 보기"):
    st.dataframe(pd.DataFrame([image_features]), use_container_width=True)
    st.write("CSV 비교에 사용한 열:", ", ".join(candidate_basis_cols))

st.markdown("<hr>", unsafe_allow_html=True)


if analysis_mode == "🎨 K-Means 군집 분석":
    st.markdown('<div class="section-title">🎨 K-Means 군집 분석 결과</div>', unsafe_allow_html=True)
    st.info("비슷한 세포끼리 자동으로 묶은 뒤, 업로드 이미지와 가장 가까운 CSV 세포가 어느 cluster에 들어가는지 예측합니다.")

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

    label_candidates = categorical_columns(data2d)
    label_candidates = [c for c in label_candidates if c not in [x_col, y_col]]
    label_col_k = best_label_column(data2d) if best_label_column(data2d) in label_candidates else (label_candidates[0] if label_candidates else None)

    use_cols = [x_col, y_col] + ([label_col_k] if label_col_k else [])
    df_kmeans = data2d[use_cols].dropna().copy()

    X = df_kmeans[[x_col, y_col]]
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    df_kmeans["cluster"] = labels.astype(str)

    candidate_xy = data2d.iloc[candidate_idx][[x_col, y_col]].astype(float).values.reshape(1, -1)
    candidate_cluster = str(model.predict(candidate_xy)[0])
    center = model.cluster_centers_[int(candidate_cluster)]
    center_distance = float(np.linalg.norm(candidate_xy[0] - center))

    if label_col_k:
        summary = (
            df_kmeans.groupby("cluster")[label_col_k]
            .agg(lambda s: s.value_counts().index[0])
            .reset_index()
            .rename(columns={label_col_k: "대표 원본 라벨"})
        )
        counts = df_kmeans.groupby("cluster").size().reset_index(name="세포 수")
        purity = (
            df_kmeans.groupby("cluster")[label_col_k]
            .agg(lambda s: round(s.value_counts(normalize=True).iloc[0] * 100, 1))
            .reset_index()
            .rename(columns={label_col_k: "대표 유형 비율(%)"})
        )
        cluster_result = counts.merge(summary, on="cluster").merge(purity, on="cluster")
        cluster_result["예측 세포 이름"] = cluster_result["대표 원본 라벨"].apply(friendly_cell_name)
        cluster_result["쉬운 세포 그룹"] = cluster_result["대표 원본 라벨"].apply(broad_cell_group)
        cluster_result["근거 설명"] = cluster_result.apply(
            lambda r: f"{r['cluster']}번 군집 안에서 {r['대표 원본 라벨']} 라벨이 가장 많아서 {r['예측 세포 이름']}로 해석했습니다.",
            axis=1
        )
        name_map = dict(zip(cluster_result["cluster"], cluster_result["예측 세포 이름"]))
        group_map = dict(zip(cluster_result["cluster"], cluster_result["쉬운 세포 그룹"]))
        df_kmeans["예측 세포 이름"] = df_kmeans["cluster"].map(name_map)
        color_target = "예측 세포 이름"
        candidate_cluster_name = name_map.get(candidate_cluster, "알 수 없음")
        candidate_cluster_group = group_map.get(candidate_cluster, "알 수 없음")
    else:
        cluster_result = df_kmeans["cluster"].value_counts().sort_index().reset_index()
        cluster_result.columns = ["cluster", "세포 수"]
        color_target = "cluster"
        candidate_cluster_name = candidate_cluster
        candidate_cluster_group = "군집 번호 기준"

    fig = px.scatter(
        df_kmeans, x=x_col, y=y_col, color=color_target,
        title=f"🎨 K-Means 군집 결과: K={k}",
        template="plotly_white", opacity=0.78
    )
    fig.add_scatter(
        x=[candidate_xy[0][0]], y=[candidate_xy[0][1]],
        mode="markers+text",
        marker=dict(size=18, symbol="star", color="red", line=dict(width=2, color="black")),
        text=["업로드 이미지 후보"],
        textposition="top center",
        name="업로드 이미지 후보"
    )
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔮 업로드 이미지의 K-Means 예측")
    st.success(f"업로드 이미지는 **Cluster {candidate_cluster} → {candidate_cluster_name}** 으로 해석됩니다.")
    st.write(f"🧩 쉬운 그룹: **{candidate_cluster_group}**")
    st.write(f"📌 근거 1: 이미지 특징과 가장 가까운 CSV 세포의 좌표가 Cluster {candidate_cluster} 중심에 가장 가깝습니다.")
    st.write(f"📌 근거 2: 해당 cluster 중심까지의 거리 = **{center_distance:.3f}**")
    st.write("📌 근거 3: 아래 표에서 해당 cluster에 가장 많이 포함된 세포 라벨을 대표 이름으로 사용했습니다.")

    st.markdown("### 📊 Cluster별 쉬운 해석표")
    st.dataframe(cluster_result, use_container_width=True)


elif analysis_mode == "🌳 의사결정트리 분류":
    st.markdown('<div class="section-title">🌳 의사결정트리 분류 결과</div>', unsafe_allow_html=True)
    st.info("이미 알려진 세포 라벨을 학습해서, 업로드 이미지와 연결된 CSV 후보 세포가 어떤 세포인지 예측합니다.")

    if len(cat_cols) == 0:
        st.warning("분류에 사용할 문자형 라벨 열이 없습니다.")
        st.stop()

    target_col = st.selectbox("예측할 세포 라벨 선택", cat_cols, index=cat_cols.index(best_label_column(cell_metadata)) if best_label_column(cell_metadata) in cat_cols else 0)

    feature_cols = numeric_columns(data2d)
    selected_features = st.multiselect(
        "학습에 사용할 숫자 특성 선택",
        feature_cols,
        default=feature_cols[:min(5, len(feature_cols))]
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    candidate_features = data2d.iloc[[candidate_idx]][selected_features].fillna(df[selected_features].median(numeric_only=True))
    candidate_pred = clf.predict(candidate_features)[0]
    candidate_raw = le.inverse_transform([candidate_pred])[0]
    candidate_easy = friendly_cell_name(candidate_raw)
    candidate_desc = friendly_cell_description(candidate_raw)
    candidate_group = broad_cell_group(candidate_raw)

    c1, c2 = st.columns(2)
    c1.metric("🌟 모델 정확도", f"{acc:.3f}")
    c2.metric("🔮 예측 세포", candidate_easy)

    st.markdown("### 🔮 업로드 이미지의 의사결정트리 예측")
    st.success(f"예측 결과: **{candidate_easy}**")
    st.write(f"🧩 큰 분류: **{candidate_group}**")
    st.write(f"📖 설명: {candidate_desc}")
    st.write(f"📌 근거 1: 의사결정트리가 `{target_col}` 라벨을 학습했습니다.")
    st.write(f"📌 근거 2: 후보 세포의 숫자형 특징({', '.join(selected_features)})을 입력하여 예측했습니다.")
    st.write(f"📌 근거 3: 테스트 데이터 기준 정확도는 **{acc:.3f}** 입니다.")

    importance = pd.DataFrame({
        "특성": selected_features,
        "중요도": clf.feature_importances_
    }).sort_values("중요도", ascending=False)
    importance["쉬운 뜻"] = importance["특성"].apply(friendly_numeric_meaning)

    fig = px.bar(importance, x="중요도", y="쉬운 뜻", orientation="h", title="🌳 예측에 많이 사용된 근거", template="plotly_white")
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(importance, use_container_width=True)


elif analysis_mode == "📈 회귀 분석":
    st.markdown('<div class="section-title">📈 회귀 분석 결과</div>', unsafe_allow_html=True)
    st.info("세포의 여러 숫자 정보를 이용해서 하나의 숫자값을 예측합니다. 예를 들어 세포 발달 단계나 스트레스 점수 등을 예측할 수 있습니다.")

    all_numeric = numeric_columns(data2d)
    if len(all_numeric) < 2:
        st.error("회귀 분석에는 숫자형 열이 2개 이상 필요합니다.")
        st.stop()

    default_target = all_numeric.index("pseudotime") if "pseudotime" in all_numeric else len(all_numeric) - 1
    target = st.selectbox("예측할 숫자값 선택", all_numeric, index=default_target)
    features = st.multiselect(
        "입력 특성 선택",
        [c for c in all_numeric if c != target],
        default=[c for c in all_numeric if c != target][:min(5, len(all_numeric)-1)]
    )

    if len(features) < 1:
        st.warning("입력 특성을 1개 이상 선택하세요.")
        st.stop()

    df = data2d[features + [target]].dropna()
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    candidate_features = data2d.iloc[[candidate_idx]][features].fillna(df[features].median(numeric_only=True))
    candidate_value = float(reg.predict(candidate_features)[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("🔮 예측값", f"{candidate_value:.3f}")
    c2.metric("📊 R²", f"{r2:.3f}")
    c3.metric("📉 MAE", f"{mae:.3f}")

    st.markdown("### 🔮 업로드 이미지의 회귀 예측")
    st.success(f"예측 결과: **{friendly_numeric_meaning(target)} = {candidate_value:.3f}**")
    st.write(f"📌 근거 1: 후보 세포의 `{', '.join(features)}` 값을 이용해 `{target}`을 예측했습니다.")
    st.write(f"📌 근거 2: R²는 **{r2:.3f}**입니다. 1에 가까울수록 실제 값을 잘 설명합니다.")
    st.write(f"📌 근거 3: MAE는 **{mae:.3f}**입니다. 예측값이 실제값과 평균적으로 이 정도 차이 난다는 뜻입니다.")

    result = pd.DataFrame({"실제값": y_test, "예측값": pred})
    result["차이"] = (result["실제값"] - result["예측값"]).abs()
    fig = px.scatter(result, x="실제값", y="예측값", title="📈 실제값 vs 예측값", template="plotly_white", hover_data=["차이"])
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)


elif analysis_mode == "🌈 UMAP 2D 시각화":
    st.markdown('<div class="section-title">🌈 UMAP 2D 시각화 결과</div>', unsafe_allow_html=True)
    st.info("세포들을 2차원 지도에 펼친 결과입니다. 가까운 점일수록 서로 비슷한 세포입니다.")

    if len(num2d) < 2:
        st.error("UMAP 2D 좌표 열이 부족합니다.")
        st.stop()

    x_col = st.selectbox("UMAP_1로 사용할 열", num2d, index=0)
    y_col = st.selectbox("UMAP_2로 사용할 열", num2d, index=1)
    color_options = ["예측 세포 이름"] + list(data2d.columns)
    color_col = st.selectbox("색상 기준", color_options)

    plot_df = data2d.copy()
    label_col2 = best_label_column(plot_df)
    if color_col == "예측 세포 이름" and label_col2:
        plot_df["예측 세포 이름"] = plot_df[label_col2].apply(friendly_cell_name)
        color_target = "예측 세포 이름"
    else:
        color_target = color_col

    candidate_x = float(data2d.iloc[candidate_idx][x_col])
    candidate_y = float(data2d.iloc[candidate_idx][y_col])

    fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_target, title="🌈 UMAP 2D 뇌세포 지도", template="plotly_white", opacity=0.72)
    fig.add_scatter(
        x=[candidate_x], y=[candidate_y],
        mode="markers+text",
        marker=dict(size=20, symbol="star", color="red", line=dict(width=2, color="black")),
        text=["업로드 이미지 후보"],
        textposition="top center",
        name="업로드 이미지 후보"
    )
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔮 업로드 이미지의 UMAP 2D 결과")
    st.success(f"업로드 이미지는 UMAP 2D 지도에서 **({candidate_x:.3f}, {candidate_y:.3f})** 위치와 가장 가깝게 배치됩니다.")
    st.write(f"📌 해석: 이 위치 주변에 있는 세포들과 비슷한 특징을 가진 것으로 볼 수 있습니다.")
    st.write(f"📌 예측 세포 이름: **{predicted_name}**")
    st.write(f"📌 근거: 이미지 특징과 가장 가까운 CSV 세포의 UMAP 좌표를 빨간 별(★)로 표시했습니다.")


elif analysis_mode == "🧊 UMAP 3D 시각화":
    st.markdown('<div class="section-title">🧊 UMAP 3D 시각화 결과</div>', unsafe_allow_html=True)
    st.info("세포들의 관계를 3차원 공간에서 입체적으로 보여줍니다.")

    if len(num3d) < 3:
        st.error("UMAP 3D 좌표 열이 부족합니다.")
        st.stop()

    x_col = st.selectbox("3D X축", num3d, index=0)
    y_col = st.selectbox("3D Y축", num3d, index=1)
    z_col = st.selectbox("3D Z축", num3d, index=2)

    plot_df = data3d.copy()
    label_col3 = best_label_column(plot_df)
    if label_col3:
        plot_df["예측 세포 이름"] = plot_df[label_col3].apply(friendly_cell_name)
        color_target = "예측 세포 이름"
    else:
        color_target = None

    candidate_x = float(data3d.iloc[candidate_idx][x_col])
    candidate_y = float(data3d.iloc[candidate_idx][y_col])
    candidate_z = float(data3d.iloc[candidate_idx][z_col])

    fig = px.scatter_3d(plot_df, x=x_col, y=y_col, z=z_col, color=color_target, title="🧊 UMAP 3D 뇌세포 공간", template="plotly_white", opacity=0.72)
    fig.add_scatter3d(
        x=[candidate_x], y=[candidate_y], z=[candidate_z],
        mode="markers+text",
        marker=dict(size=8, symbol="diamond", color="red"),
        text=["업로드 이미지 후보"],
        name="업로드 이미지 후보"
    )
    fig.update_layout(
        paper_bgcolor="rgba(255,246,250,0.95)",
        scene=dict(bgcolor="rgba(255,255,255,0.85)"),
        font=dict(color="#333333"),
        title_font=dict(color="#C2185B", size=22),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔮 업로드 이미지의 UMAP 3D 결과")
    st.success(f"업로드 이미지는 3D 공간에서 **({candidate_x:.3f}, {candidate_y:.3f}, {candidate_z:.3f})** 위치와 연결됩니다.")
    st.write(f"📌 예측 세포 이름: **{predicted_name}**")
    st.write("📌 근거: 이미지 특징과 가장 가까운 CSV 세포의 3D UMAP 위치를 빨간 표시로 나타냈습니다.")
    st.write("📌 가까운 공간에 모인 세포들은 서로 비슷한 세포일 가능성이 큽니다.")


elif analysis_mode == "📍 Spatial 시각화":
    st.markdown('<div class="section-title">📍 Spatial 시각화 결과</div>', unsafe_allow_html=True)
    st.info("세포가 조직 안에서 어느 위치에 있는지 보여줍니다.")

    if len(numsp) < 2:
        st.error("Spatial 좌표 열이 부족합니다.")
        st.stop()

    x_col = st.selectbox("Spatial X축", numsp, index=0)
    y_col = st.selectbox("Spatial Y축", numsp, index=1)

    plot_df = datasp.copy()
    label_col_sp = best_label_column(plot_df)
    if label_col_sp:
        plot_df["예측 세포 이름"] = plot_df[label_col_sp].apply(friendly_cell_name)
        color_target = "예측 세포 이름"
    else:
        color_target = None

    candidate_x = float(datasp.iloc[candidate_idx][x_col])
    candidate_y = float(datasp.iloc[candidate_idx][y_col])

    fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_target, title="📍 Spatial 뇌세포 위치 지도", template="plotly_white", opacity=0.75)
    fig.add_scatter(
        x=[candidate_x], y=[candidate_y],
        mode="markers+text",
        marker=dict(size=20, symbol="star", color="red", line=dict(width=2, color="black")),
        text=["업로드 이미지 후보"],
        textposition="top center",
        name="업로드 이미지 후보"
    )
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    x_mid = datasp[x_col].median()
    y_mid = datasp[y_col].median()
    horizontal = "오른쪽" if candidate_x >= x_mid else "왼쪽"
    vertical = "위쪽" if candidate_y >= y_mid else "아래쪽"

    st.markdown("### 🔮 업로드 이미지의 Spatial 결과")
    st.success(f"업로드 이미지는 조직 지도에서 **{horizontal} {vertical} 영역**에 가까운 세포로 연결됩니다.")
    st.write(f"📌 좌표: ({candidate_x:.3f}, {candidate_y:.3f})")
    st.write(f"📌 예측 세포 이름: **{predicted_name}**")
    st.write("📌 근거: 이미지 특징과 가장 가까운 CSV 세포의 공간 좌표를 빨간 별(★)로 표시했습니다.")
    st.write("📌 같은 색 세포가 특정 영역에 모이면 그 세포 그룹이 그 위치에 많이 분포한다는 뜻입니다.")


elif analysis_mode == "🧬 유전자/조절인자 탐색":
    st.markdown('<div class="section-title">🧬 유전자/조절인자 탐색 결과</div>', unsafe_allow_html=True)
    st.info("유전자와 조절인자는 세포가 어떤 성격을 가지는지 알려주는 단서입니다.")

    tab1, tab2 = st.tabs(["🧬 유전자 탐색", "🎯 조절인자 탐색"])

    with tab1:
        st.markdown("### 🔮 예측 세포 그룹과 관련된 유전자 후보")
        if "module" in gene_metadata.columns:
            module_key = str(raw_label).lower()
            gene_view = gene_metadata.copy()
            gene_view["쉬운 설명"] = gene_view["module"].apply(lambda m: f"{m} 특성과 관련된 유전자 후보입니다.")
            related = gene_view[gene_view["module"].astype(str).str.lower().str.contains(module_key.split("_")[0], na=False)]
            if len(related) == 0:
                related = gene_view.sort_values("mean_expression", ascending=False).head(20) if "mean_expression" in gene_view.columns else gene_view.head(20)
        else:
            gene_view = gene_metadata.copy()
            gene_view["쉬운 설명"] = "세포 특징을 설명하는 데 사용될 수 있는 유전자 후보입니다."
            related = gene_view.head(20)

        st.write(f"📌 예측 세포 이름: **{predicted_name}**")
        st.write("📌 근거: 유전자 메타데이터에서 발현량 또는 module 정보를 이용해 관련 후보를 보여줍니다.")
        st.dataframe(related.head(50), use_container_width=True)

    with tab2:
        st.markdown("### 🎯 업로드 이미지 후보 세포의 조절인자 점수")
        reg_row = regulators.iloc[[candidate_idx]].copy()
        id_col = find_id_column(reg_row)
        score_cols = [c for c in reg_row.columns if c != id_col and pd.api.types.is_numeric_dtype(regulators[c])]
        long_reg = reg_row[score_cols].T.reset_index()
        long_reg.columns = ["조절인자", "점수"]
        long_reg = long_reg.sort_values("점수", ascending=False).head(8)
        long_reg["쉬운 설명"] = long_reg["조절인자"].apply(lambda x: f"{x} 점수가 높을수록 해당 세포 특징이나 발달 방향과 관련이 클 수 있습니다.")

        fig = px.bar(long_reg, x="점수", y="조절인자", orientation="h", title="🎯 후보 세포의 주요 조절인자", template="plotly_white")
        fig = style_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.write(f"📌 예측 세포 이름: **{predicted_name}**")
        st.write("📌 근거: 업로드 이미지와 가장 가까운 CSV 세포의 조절인자 점수를 가져와 높은 순서로 정렬했습니다.")
        st.dataframe(long_reg, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🧠 Brain Cell AI Explorer | 이미지 참고 + CSV 기반 뇌세포 예측 및 근거 출력 Streamlit App")
