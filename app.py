# ============================================================================
# CROP YIELD PREDICTION - STREAMLIT WEB APPLICATION
# ============================================================================
# Dataset : India Crop Wise Area, Production and Yield (1997-2023)
# Source  : India Data Portal / Ministry of Agriculture & Farmers Welfare
# Target  : yield (Tonnes / Hectare)
# ============================================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Crop Yield Predictor | India",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Hero section */
  .hero-box {
    background: linear-gradient(135deg, #1a6b3c 0%, #2d9e5f 50%, #1a6b3c 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(26, 107, 60, 0.35);
  }
  .hero-box h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
  .hero-box p  { font-size: 1.05rem; opacity: 0.9; margin-top: 0.5rem; }

  /* Metric cards */
  .metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s ease;
  }
  .metric-card:hover { transform: translateY(-3px); }
  .metric-card .val  { font-size: 2rem; font-weight: 700; color: #1a6b3c; }
  .metric-card .lbl  { font-size: 0.85rem; color: #64748b; margin-top: 0.2rem; }

  /* Prediction box */
  .pred-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 2px solid #16a34a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
  }
  .pred-box .big-number { font-size: 3.5rem; font-weight: 800; color: #16a34a; }
  .pred-box .unit       { font-size: 1rem; color: #4b5563; }

  /* Best model badge */
  .best-badge {
    display: inline-block;
    background: #16a34a;
    color: white;
    border-radius: 20px;
    padding: 2px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
  }

  /* Info cards */
  .info-card {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
  }

  /* Section headers */
  .section-header {
    color: #1a6b3c;
    font-weight: 700;
    font-size: 1.3rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #dcfce7;
  }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "crop-wise-area-production-yield.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_FILES = {
    "Best Model (Auto)":   os.path.join(MODELS_DIR, "best_model.pkl"),
    "Linear Regression":   os.path.join(MODELS_DIR, "linear_regression_pipeline.pkl"),
    "Decision Tree":       os.path.join(MODELS_DIR, "decision_tree_pipeline.pkl"),
    "Random Forest":       os.path.join(MODELS_DIR, "random_forest_pipeline.pkl"),
    "LinearSVR":           os.path.join(MODELS_DIR, "linearsvr_pipeline.pkl"),
}

RESULTS_CSV  = os.path.join(RESULTS_DIR, "model_results.csv")
BEST_JSON    = os.path.join(RESULTS_DIR, "best_model_info.json")
COMPARE_IMG  = os.path.join(RESULTS_DIR, "model_comparison.png")

# ============================================================================
# CACHED DATA LOADERS
# ============================================================================
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["year_start"] = df["year"].str[:4].astype(int)
    return df

@st.cache_resource(show_spinner="Loading model...")
def load_model(model_key):
    path = MODEL_FILES[model_key]
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_CSV):
        return None
    return pd.read_csv(RESULTS_CSV)

@st.cache_data
def load_best_info():
    if not os.path.exists(BEST_JSON):
        return None
    with open(BEST_JSON) as f:
        return json.load(f)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/200px-Emblem_of_India.svg.png", width=80)
    st.title("Navigation")
    tab_choice = st.radio(
        "Go to",
        ["🏠 Dashboard", "📊 EDA", "🏆 Model Comparison", "🌾 Predict Yield"],
        index=0,
    )
    st.markdown("---")
    st.caption("**Dataset:** India Data Portal")
    st.caption("**Source:** Ministry of Agriculture & Farmers Welfare, GoI")
    st.caption("**Target Variable:** Yield (Tonnes/Hectare)")

# ============================================================================
# TAB 1 - DASHBOARD
# ============================================================================
if tab_choice == "🏠 Dashboard":
    st.markdown("""
    <div class='hero-box'>
      <h1>🌾 Crop Yield Prediction System</h1>
      <p>Machine Learning-powered agricultural yield forecasting for India<br>
      Powered by historical district-level data from the Ministry of Agriculture &amp; Farmers Welfare, GoI</p>
    </div>
    """, unsafe_allow_html=True)

    data = load_data()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='val'>{len(data):,}</div><div class='lbl'>Total Records</div></div>""",
            unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='val'>{data['state_name'].nunique()}</div><div class='lbl'>States / UTs</div></div>""",
            unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='val'>{data['district_name'].nunique()}</div><div class='lbl'>Districts</div></div>""",
            unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='val'>{data['crop_name'].nunique()}</div><div class='lbl'>Crop Types</div></div>""",
            unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        yr_min = data['year_start'].min()
        yr_max = data['year_start'].max()
        st.markdown(f"""<div class='metric-card'>
            <div class='val'>{yr_min}–{yr_max}</div><div class='lbl'>Year Range</div></div>""",
            unsafe_allow_html=True)
    with col6:
        st.markdown(f"""<div class='metric-card'>
            <div class='val'>{data['season'].nunique()}</div><div class='lbl'>Seasons</div></div>""",
            unsafe_allow_html=True)
    with col7:
        st.markdown(f"""<div class='metric-card'>
            <div class='val'>{data['crop_type'].nunique()}</div><div class='lbl'>Crop Categories</div></div>""",
            unsafe_allow_html=True)
    with col8:
        med_yield = data['yield'].median()
        st.markdown(f"""<div class='metric-card'>
            <div class='val'>{med_yield:.2f}</div><div class='lbl'>Median Yield (T/ha)</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Project Overview</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Objective**
        Build a regression system to predict agricultural crop yield (tonnes/hectare)
        using historical Indian district-level agriculture data.

        **Dataset Source**
        India Data Portal (indiadataportal.com) — underlying government source:
        Ministry of Agriculture & Farmers Welfare, Government of India.

        **Year Coverage:** 1997-1998 to 2022-2023 (26 agricultural years)
        """)
    with col_b:
        st.markdown("""
        **Machine Learning Models Used**
        - 📈 **Linear Regression** — Baseline linear model
        - 🌳 **Decision Tree** — Non-linear relationships
        - 🌲 **Random Forest** — Ensemble (best performer)
        - ⚡ **LinearSVR** — Support Vector Regression

        **Key Design Decisions**
        - Target transformation: log1p → expm1 (handles skewness)
        - OneHotEncoder for categorical features
        - GridSearchCV on Random Forest (memory-safe subset)
        """)

    st.markdown("<div class='section-header'>Dataset Structure</div>", unsafe_allow_html=True)
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.dataframe(data.head(10), use_container_width=True)
    with col_info2:
        missing = data.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing"]
        missing["% Missing"] = (missing["Missing"] / len(data) * 100).round(3)
        st.dataframe(missing, use_container_width=True, height=300)

# ============================================================================
# TAB 2 - EDA
# ============================================================================
elif tab_choice == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    data = load_data()

    st.markdown("<div class='section-header'>Yield Statistics</div>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Min Yield",    f"{data['yield'].min():.3f}")
    col2.metric("Median Yield", f"{data['yield'].median():.3f}")
    col3.metric("Mean Yield",   f"{data['yield'].mean():.2f}")
    col4.metric("Max Yield",    f"{data['yield'].max():.1f}")
    col5.metric("Skewness",     f"{data['yield'].skew():.2f}")

    st.markdown("""
    <div class='info-card'>
    ⚠️ <strong>High skewness (~16.33)</strong>: The yield distribution is extremely right-skewed due to a small number of
    high-yield crops (e.g., coconut). We used <code>log1p</code> transformation on the target variable during model training
    to improve regression performance. Extreme values are NOT removed as they represent legitimate agricultural data.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Yield Distribution</div>", unsafe_allow_html=True)
    eda_filter = st.slider("Filter: Show yield below percentile", 90, 100, 99, key="eda_pct")
    threshold = float(data['yield'].quantile(eda_filter / 100))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    filtered = data[data['yield'] <= threshold]['yield']
    axes[0].hist(filtered, bins=80, color='#2d9e5f', edgecolor='white', alpha=0.8)
    axes[0].set_title(f'Yield Distribution (≤ {eda_filter}th percentile)')
    axes[0].set_xlabel('Yield (Tonnes/Hectare)')
    axes[0].set_ylabel('Frequency')

    log_yield = np.log1p(data['yield'])
    axes[1].hist(log_yield, bins=80, color='#1a6b3c', edgecolor='white', alpha=0.8)
    axes[1].set_title('Log(1+Yield) Distribution')
    axes[1].set_xlabel('log1p(Yield)')
    axes[1].set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<div class='section-header'>Crop & Category Analysis</div>", unsafe_allow_html=True)
    col_e1, col_e2 = st.columns(2)

    with col_e1:
        top_crops = data.groupby('crop_name')['yield'].median().nlargest(15)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        top_crops.sort_values().plot(kind='barh', ax=ax2, color='#2d9e5f')
        ax2.set_title('Top 15 Crops by Median Yield')
        ax2.set_xlabel('Median Yield (T/ha)')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_e2:
        crop_type_counts = data['crop_type'].value_counts()
        fig3, ax3 = plt.subplots(figsize=(7, 6))
        ax3.pie(crop_type_counts, labels=crop_type_counts.index, autopct='%1.1f%%',
                colors=plt.cm.Set3.colors[:len(crop_type_counts)], startangle=140)
        ax3.set_title('Records by Crop Type')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.markdown("<div class='section-header'>Temporal & Seasonal Trends</div>", unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        year_trend = data.groupby('year_start')['yield'].median()
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.plot(year_trend.index, year_trend.values, marker='o', color='#1a6b3c', linewidth=2)
        ax4.fill_between(year_trend.index, year_trend.values, alpha=0.15, color='#2d9e5f')
        ax4.set_title('Median Yield Trend Over Years')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Median Yield (T/ha)')
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    with col_t2:
        season_yield = data.groupby('season')['yield'].median().sort_values(ascending=False)
        fig5, ax5 = plt.subplots(figsize=(9, 5))
        season_yield.plot(kind='bar', ax=ax5, color='#2d9e5f', edgecolor='white')
        ax5.set_title('Median Yield by Season')
        ax5.set_xlabel('Season')
        ax5.set_ylabel('Median Yield (T/ha)')
        ax5.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    st.markdown("<div class='section-header'>State-wise Analysis</div>", unsafe_allow_html=True)
    state_yield = data.groupby('state_name')['yield'].median().sort_values(ascending=False)
    fig6, ax6 = plt.subplots(figsize=(14, 6))
    state_yield.plot(kind='bar', ax=ax6, color='#2d9e5f', edgecolor='white')
    ax6.set_title('Median Yield by State/UT')
    ax6.set_xlabel('State')
    ax6.set_ylabel('Median Yield (T/ha)')
    ax6.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

# ============================================================================
# TAB 3 - MODEL COMPARISON
# ============================================================================
elif tab_choice == "🏆 Model Comparison":
    st.title("🏆 Model Comparison")

    results = load_results()
    best_info = load_best_info()

    if results is None:
        st.warning("Model results not found. Please run `crop_yield_model.py` first to train and save models.")
    else:
        if best_info:
            bm = best_info.get("best_model", "Random Forest")
            st.success(f"🏆 Best Performing Model: **{bm}**")

        st.markdown("<div class='section-header'>Performance Metrics</div>", unsafe_allow_html=True)

        display_df = results.copy()
        display_df['MAE']  = display_df['MAE'].round(4)
        display_df['MSE']  = display_df['MSE'].round(2)
        display_df['RMSE'] = display_df['RMSE'].round(4)
        display_df['R2']   = display_df['R2'].round(6)

        best_model_name = best_info.get("best_model", "") if best_info else ""
        st.dataframe(
            display_df.style.highlight_max(subset=['R2'], color='#bbf7d0')
                            .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='#bbf7d0'),
            use_container_width=True,
        )

        st.markdown("<div class='section-header'>Visual Comparison</div>", unsafe_allow_html=True)

        if os.path.exists(COMPARE_IMG):
            st.image(COMPARE_IMG, use_container_width=True)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            palette = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
            models  = results['Model'].tolist()

            axes[0].bar(models, results['RMSE'], color=palette)
            axes[0].set_title('RMSE (lower is better)')
            axes[0].set_ylabel('RMSE')
            axes[0].tick_params(axis='x', rotation=20)

            axes[1].bar(models, results['MAE'], color=palette)
            axes[1].set_title('MAE (lower is better)')
            axes[1].set_ylabel('MAE')
            axes[1].tick_params(axis='x', rotation=20)

            axes[2].bar(models, results['R2'], color=palette)
            axes[2].set_title('R2 Score (higher is better)')
            axes[2].set_ylabel('R2')
            axes[2].tick_params(axis='x', rotation=20)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if best_info:
            st.markdown("<div class='section-header'>GridSearchCV Results</div>", unsafe_allow_html=True)
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.metric("Best CV RMSE",  f"{best_info.get('best_cv_rmse', 0):.4f}")
                bp = best_info.get("best_params", {})
                st.json(bp)
            with col_g2:
                fm = best_info.get("final_metrics", {})
                if fm:
                    st.metric("Final MAE",  f"{fm.get('MAE', 0):.4f}")
                    st.metric("Final RMSE", f"{fm.get('RMSE', 0):.4f}")
                    st.metric("Final R2",   f"{fm.get('R2', 0):.6f}")

            st.info("""
            **GridSearchCV Methodology Note:**
            Due to the large dataset size (455,359 rows) and computational memory constraints,
            GridSearchCV was performed using a representative 10,000-row subset of the training data
            with 3-fold cross-validation (4 parameter combinations × 3 folds = 12 model trainings).
            The selected best hyperparameters were then used to retrain the Random Forest on the
            complete 80% training dataset (364,287 rows). The test set (91,072 rows) was never
            used during hyperparameter tuning.
            """)

        st.markdown("<div class='section-header'>Metric Explanations</div>", unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            **MAE (Mean Absolute Error)**
            Average absolute difference between predicted and actual yield.
            Measured in Tonnes/Hectare. Lower is better.

            **MSE (Mean Squared Error)**
            Average squared difference. Penalizes large errors more heavily.
            Lower is better.
            """)
        with col_m2:
            st.markdown("""
            **RMSE (Root Mean Squared Error)**
            Square root of MSE — same unit as yield (T/ha).
            Lower is better.

            **R² Score**
            Proportion of variance explained by the model.
            Range: 0 to 1. Higher is better (1 = perfect fit).
            """)

# ============================================================================
# TAB 4 - PREDICTION
# ============================================================================
elif tab_choice == "🌾 Predict Yield":
    st.title("🌾 Crop Yield Prediction")

    data = load_data()

    # Check if models are available
    models_available = os.path.exists(MODEL_FILES["Best Model (Auto)"])

    if not models_available:
        st.error("""
        **Models not found!**
        Please run `crop_yield_model.py` first to train and save the models:
        ```
        python crop_yield_model.py
        ```
        """)
    else:
        col_form, col_result = st.columns([1, 1])

        with col_form:
            st.markdown("<div class='section-header'>Enter Crop Details</div>", unsafe_allow_html=True)

            # Model selection
            best_info = load_best_info()
            best_name = best_info.get("best_model", "Random Forest") if best_info else "Random Forest"

            model_choice = st.selectbox(
                "Select Model",
                list(MODEL_FILES.keys()),
                index=0,
                help=f"Best model is: {best_name}",
            )

            # Get unique values for dropdowns
            states    = sorted(data['state_name'].unique().tolist())
            districts = sorted(data['district_name'].unique().tolist())
            crops     = sorted(data['crop_name'].unique().tolist())
            crop_types = sorted(data['crop_type'].unique().tolist())
            seasons   = sorted(data['season'].unique().tolist())

            state    = st.selectbox("State / UT",     states,    index=states.index("Uttar Pradesh") if "Uttar Pradesh" in states else 0)
            district = st.selectbox("District",       districts, index=0)
            crop     = st.selectbox("Crop",           crops,     index=crops.index("Rice") if "Rice" in crops else 0)
            crop_type = st.selectbox("Crop Type",     crop_types, index=crop_types.index("Cereals") if "Cereals" in crop_types else 0)
            season   = st.selectbox("Season",         seasons,   index=seasons.index("Kharif") if "Kharif" in seasons else 0)

            year_min = int(data['year_start'].min())
            year_max = int(data['year_start'].max())
            year     = st.slider("Year", year_min, year_max + 2, year_max)

            area = st.number_input(
                "Area (Hectares)",
                min_value=0.001,
                max_value=10_000_000.0,
                value=500.0,
                step=10.0,
                format="%.3f",
            )

            predict_btn = st.button("🌾 Predict Yield", type="primary", use_container_width=True)

        with col_result:
            st.markdown("<div class='section-header'>Prediction Result</div>", unsafe_allow_html=True)

            if predict_btn:
                with st.spinner("Loading model and predicting..."):
                    model = load_model(model_choice)

                if model is None:
                    st.error(f"Could not load model: {model_choice}")
                else:
                    input_df = pd.DataFrame([{
                        'state_name':    state,
                        'district_name': district,
                        'crop_name':     crop,
                        'crop_type':     crop_type,
                        'season':        season,
                        'area':          area,
                        'year_start':    year,
                    }])

                    prediction = float(model.predict(input_df)[0])
                    prediction = max(0, prediction)  # Ensure non-negative

                    st.markdown(f"""
                    <div class='pred-box'>
                        <div style='font-size:1.1rem; color:#374151; margin-bottom:0.5rem;'>Predicted Crop Yield</div>
                        <div class='big-number'>{prediction:.3f}</div>
                        <div class='unit'>Tonnes per Hectare (T/ha)</div>
                        <div style='margin-top:1rem; font-size:0.85rem; color:#6b7280;'>Model: {model_choice}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Estimated production
                    est_prod = prediction * area
                    st.metric("Estimated Total Production", f"{est_prod:,.1f} Tonnes", help="Prediction × Area")

                    # Context from dataset
                    similar = data[
                        (data['crop_name'] == crop) &
                        (data['season'] == season)
                    ]['yield']
                    if len(similar) > 0:
                        st.markdown("---")
                        st.markdown("**Historical Context (same crop & season)**")
                        hcol1, hcol2, hcol3 = st.columns(3)
                        hcol1.metric("Historical Median", f"{similar.median():.3f} T/ha")
                        hcol2.metric("Historical Mean",   f"{similar.mean():.3f} T/ha")
                        hcol3.metric("Historical Max",    f"{similar.max():.3f} T/ha")
            else:
                st.markdown("""
                <div style='text-align:center; padding:3rem; color:#9ca3af;'>
                    <div style='font-size:3rem;'>🌱</div>
                    <div style='font-size:1rem; margin-top:1rem;'>Fill in the crop details and click<br><strong>Predict Yield</strong> to get your forecast.</div>
                </div>
                """, unsafe_allow_html=True)

        # Show input summary
        if predict_btn:
            st.markdown("---")
            st.markdown("<div class='section-header'>Input Summary</div>", unsafe_allow_html=True)
            summary_df = pd.DataFrame({
                'Parameter': ['State', 'District', 'Crop', 'Crop Type', 'Season', 'Year', 'Area (ha)'],
                'Value': [state, district, crop, crop_type, season, str(year), str(area)],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
