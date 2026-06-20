import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🧠 뇌세포 AI 선별 연구소", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #fff5fb 0%, #eef7ff 45%, #f4fff1 100%);} 
.big-title {font-size: 44px; font-weight: 900; text-align: center; color: #5b21b6;}
.sub-title {font-size: 19px; text-align: center; color: #475569;}
.card {padding: 18px; border-radius: 22px; background: rgba(255,255,255,0.78); box-shadow: 0 8px 26px rgba(80,80,120,0.12); border: 1px solid rgba(255,255,255,0.8);}
.good {color:#059669; font-weight:800;}
.warn {color:#ea580c; font-weight:800;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧠✨ 뇌세포 AI 선별 연구소 🧬🔬</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">뇌세포 이미지와 유전자 발현 데이터를 활용해 군집 · 분류 · 회귀 알고리즘으로 세포를 탐색해요 🚀</div>', unsafe_allow_html=True)
st.divider()

@st.cache_data
def load_data():
    expr = pd.read_csv("expression_matrix_normalized.csv").rename(columns={"Unnamed: 0":"cell_id"})
    meta = pd.read_csv("cell_metadata.csv").rename(columns={"Unnamed: 0":"cell_id"})
    gene_meta = pd.read_csv("gene_metadata.csv").rename(columns={"Unnamed: 0":"gene"})
    regs = pd.read_csv("ground_truth_regulators.csv").rename(columns={"Unnamed: 0":"cell_id"})
    umap2 = pd.read_csv("embeddings_umap2d.csv").rename(columns={"Unnamed: 0":"cell_id"})
    umap3 = pd.read_csv("embeddings_umap3d.csv").rename(columns={"Unnamed: 0":"cell_id"})
    spatial = pd.read_csv("embeddings_spatial.csv").rename(columns={"Unnamed: 0":"cell_id"})

    data = meta.merge(umap2, on="cell_id", how="left")
    data = data.merge(spatial, on="cell_id", how="left")
    data = data.merge(regs, on="cell_id", how="left")
    return expr, meta, gene_meta, regs, umap2, umap3, spatial, data

try:
    expr, meta, gene_meta, regs, umap2, umap3, spatial, data = load_data()
except FileNotFoundError:
    st.error("CSV 파일들이 main.py와 같은 GitHub 폴더에 있어야 합니다. 파일명을 그대로 업로드해주세요! 📁")
    st.stop()

cell_id_col = "cell_id"
gene_cols = [c for c in expr.columns if c != cell_id_col]

with st.sidebar:
    st.header("🎛️ 분석 설정")
    uploaded_img = st.file_uploader("🖼️ 뇌세포 이미지 업로드", type=["png", "jpg", "jpeg"])
    mode = st.radio("🤖 분석 방식 선택", ["🧩 군집 분석", "🏷️ 분류 분석", "📈 회귀 분석"])
    max_genes = st.slider("🧬 사용할 유전자 수", 20, 300, 80, 10)
    point_size = st.slider("🔎 점 크기", 4, 14, 7)
    st.caption("유전자는 분산이 큰 순서로 자동 선택됩니다.")

col_img, col_info = st.columns([1, 2])
with col_img:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🖼️ 입력 이미지")
    if uploaded_img:
        st.image(uploaded_img, caption="업로드한 뇌세포 이미지", use_container_width=True)
        st.success("이미지가 입력되었습니다! 아래 데이터 분석 결과와 함께 세포를 선별합니다. ✅")
    else:
        st.info("이미지를 넣으면 앱이 더 예쁘게 보여요. 현재는 CSV 데이터 기반으로 분석합니다. 🌈")
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📦 데이터 요약")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("세포 수 🧫", f"{len(meta):,}")
    m2.metric("유전자 수 🧬", f"{len(gene_cols):,}")
    m3.metric("세포 유형 🏷️", meta["lineage"].nunique())
    m4.metric("조절인자 점수 ⚡", regs.shape[1]-1)
    st.write("세포 유형:", " · ".join(sorted(meta["lineage"].dropna().unique().astype(str))))
    st.markdown('</div>', unsafe_allow_html=True)

# feature selection
variances = expr[gene_cols].var().sort_values(ascending=False)
selected_genes = variances.head(max_genes).index.tolist()
X = expr[selected_genes].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.divider()

if mode == "🧩 군집 분석":
    st.header("🧩 군집 분석으로 비슷한 뇌세포끼리 묶기 🌌")
    algo = st.selectbox("알고리즘 선택", ["K-Means", "DBSCAN", "Gaussian Mixture"])

    if algo == "K-Means":
        k = st.slider("군집 개수 K", 2, 10, 5)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    elif algo == "DBSCAN":
        eps = st.slider("eps: 가까운 정도", 0.5, 8.0, 3.0, 0.1)
        min_samples = st.slider("min_samples", 3, 30, 8)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
    else:
        k = st.slider("군집 개수", 2, 10, 5)
        model = GaussianMixture(n_components=k, random_state=42)
        labels = model.fit_predict(X_scaled)

    result = data.copy()
    result["AI_cluster"] = labels.astype(str)

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.scatter(result, x="UMAP_1", y="UMAP_2", color="AI_cluster", hover_data=["cell_id", "lineage", "pseudotime"],
                         title=f"🌈 {algo} 군집 결과", template="plotly_white")
        fig.update_traces(marker=dict(size=point_size, opacity=0.82))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("📊 군집별 세포 수")
        st.dataframe(result["AI_cluster"].value_counts().reset_index().rename(columns={"AI_cluster":"cluster", "count":"cell_count"}), use_container_width=True)
        valid = len(set(labels)) > 1 and -1 not in set(labels)
        if valid:
            st.metric("실루엣 점수 ✨", f"{silhouette_score(X_scaled, labels):.3f}")
        else:
            st.info("DBSCAN의 -1은 잡음 세포 후보입니다. 🧹")

    selected_cluster = st.selectbox("🔍 선별할 군집", sorted(result["AI_cluster"].unique()))
    picked = result[result["AI_cluster"] == selected_cluster]
    st.success(f"선택된 군집 {selected_cluster}의 뇌세포 후보: {len(picked):,}개 🧠")
    st.dataframe(picked.head(200), use_container_width=True)

elif mode == "🏷️ 분류 분석":
    st.header("🏷️ 분류 분석으로 뇌세포 유형 예측하기 🎯")
    algo = st.selectbox("알고리즘 선택", ["Random Forest", "KNN", "SVM"])
    target = st.selectbox("예측할 세포 라벨", ["lineage", "batch", "branch_index"])

    y = meta[target].astype(str)
    test_size = st.slider("테스트 데이터 비율", 0.1, 0.4, 0.25, 0.05)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X_scaled, y, meta[cell_id_col], test_size=test_size, random_state=42, stratify=y if y.nunique() < len(y)*0.5 else None
    )

    if algo == "Random Forest":
        n = st.slider("트리 개수", 50, 300, 150, 50)
        clf = RandomForestClassifier(n_estimators=n, random_state=42)
    elif algo == "KNN":
        k = st.slider("이웃 수 K", 3, 25, 7, 2)
        clf = KNeighborsClassifier(n_neighbors=k)
    else:
        c = st.slider("C: 분류 강도", 0.1, 10.0, 1.0, 0.1)
        clf = SVC(C=c, probability=True, random_state=42)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    st.metric("분류 정확도 🎯", f"{acc:.3f}")
    pred_df = pd.DataFrame({"cell_id": id_test.values, "actual": y_test.values, "predicted": pred})
    pred_df = pred_df.merge(umap2, on="cell_id", how="left").merge(meta, on="cell_id", how="left", suffixes=("", "_meta"))
    pred_df["correct"] = np.where(pred_df["actual"] == pred_df["predicted"], "✅ 정답", "❌ 오답")

    fig = px.scatter(pred_df, x="UMAP_1", y="UMAP_2", color="predicted", symbol="correct",
                     hover_data=["cell_id", "actual", "predicted"], title=f"🧠 {algo} 예측 결과", template="plotly_white")
    fig.update_traces(marker=dict(size=point_size, opacity=0.85))
    st.plotly_chart(fig, use_container_width=True)

    wanted = st.selectbox("🔍 선별할 예측 세포 유형", sorted(pred_df["predicted"].unique()))
    picked = pred_df[pred_df["predicted"] == wanted]
    st.success(f"{wanted}로 예측된 뇌세포 후보: {len(picked):,}개 🧬")
    st.dataframe(picked.head(200), use_container_width=True)

else:
    st.header("📈 회귀 분석으로 세포 점수 예측하기 🚀")
    algo = st.selectbox("알고리즘 선택", ["Random Forest Regressor", "Linear Regression"])
    reg_targets = ["pseudotime", "cell_cycle_score", "stress_score", "branch_gate"] + [c for c in regs.columns if c != "cell_id"]
    target = st.selectbox("예측할 연속값", reg_targets)

    if target in meta.columns:
        y = meta[target].astype(float)
    else:
        y = regs[target].astype(float)

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X_scaled, y, meta[cell_id_col], test_size=0.25, random_state=42)

    if algo == "Random Forest Regressor":
        n = st.slider("트리 개수", 50, 300, 150, 50)
        reg = RandomForestRegressor(n_estimators=n, random_state=42)
    else:
        reg = LinearRegression()

    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    mae = mean_absolute_error(y_test, pred)

    a, b, c = st.columns(3)
    a.metric("R² 설명력 📌", f"{r2:.3f}")
    b.metric("RMSE 오차 📉", f"{rmse:.3f}")
    c.metric("MAE 평균오차 🧮", f"{mae:.3f}")

    pred_df = pd.DataFrame({"cell_id": id_test.values, "actual": y_test.values, "predicted": pred})
    pred_df = pred_df.merge(umap2, on="cell_id", how="left").merge(meta, on="cell_id", how="left")

    fig1 = px.scatter(pred_df, x="actual", y="predicted", color="lineage", hover_data=["cell_id"],
                      title="🎯 실제값 vs 예측값", template="plotly_white", trendline="ols")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(pred_df, x="UMAP_1", y="UMAP_2", color="predicted", hover_data=["cell_id", "lineage"],
                      title=f"🌡️ UMAP 위의 {target} 예측 점수", template="plotly_white", color_continuous_scale="Turbo")
    fig2.update_traces(marker=dict(size=point_size, opacity=0.85))
    st.plotly_chart(fig2, use_container_width=True)

    threshold = st.slider("🔍 선별 기준: 예측 점수 이상", float(pred_df["predicted"].min()), float(pred_df["predicted"].max()), float(pred_df["predicted"].quantile(0.75)))
    picked = pred_df[pred_df["predicted"] >= threshold].sort_values("predicted", ascending=False)
    st.success(f"예측 점수가 높은 뇌세포 후보: {len(picked):,}개 🌟")
    st.dataframe(picked.head(200), use_container_width=True)

st.divider()
st.subheader("🧬 유전자 메타데이터 탐색")
module = st.multiselect("관심 유전자 모듈", sorted(gene_meta["module"].dropna().unique()), default=list(sorted(gene_meta["module"].dropna().unique()))[:2])
show_gene = gene_meta[gene_meta["module"].isin(module)] if module else gene_meta
fig_gene = px.scatter(show_gene, x="mean_expression", y="dispersions_norm", color="module", hover_data=["gene", "highly_variable"],
                      title="🌈 유전자 평균 발현량과 변동성", template="plotly_white")
st.plotly_chart(fig_gene, use_container_width=True)

st.caption("⚠️ 참고: 업로드 이미지는 앱에서 표시용으로 사용됩니다. 실제 이미지 픽셀을 분석하는 딥러닝 모델은 별도 학습 데이터가 필요합니다. 현재 앱은 제공된 CSV의 유전자 발현·세포 메타데이터 기반으로 뇌세포 후보를 선별합니다.")
