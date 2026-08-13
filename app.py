import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="DengAI | Dengue Forecasting",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom Styling
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0E1117;
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #171923;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #171B24;
        border: 1px solid #292E39;
        padding: 20px;
        border-radius: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #AAB2C0;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
    }

    /* Titles */
    h1 {
        color: #E1BEE7 !important;
    }

    h2, h3 {
        color: #E1BEE7 !important;
    }

    /* Info boxes */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


FEATURES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "dengue_features_train.csv"
)


LABELS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "dengue_labels_train.csv"
)


SJ_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_san_juan_final.joblib"
)


IQ_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_iquitos_final.joblib"
)


SJ_IMPUTER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "imputer_san_juan_final.joblib"
)


IQ_IMPUTER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "imputer_iquitos_final.joblib"
)


# ============================================================
# Load Data
# ============================================================

@st.cache_data
def load_data():

    features = pd.read_csv(
        FEATURES_PATH
    )

    labels = pd.read_csv(
        LABELS_PATH
    )

    df = features.merge(
        labels,
        on=[
            "city",
            "year",
            "weekofyear"
        ],
        how="inner"
    )

    df["week_start_date"] = pd.to_datetime(
        df["week_start_date"],
        errors="coerce"
    )

    return df


# ============================================================
# Load Models + Imputers
# ============================================================

@st.cache_resource
def load_models_and_imputers():

    sj_model = joblib.load(
        SJ_MODEL_PATH
    )

    iq_model = joblib.load(
        IQ_MODEL_PATH
    )

    sj_imputer = joblib.load(
        SJ_IMPUTER_PATH
    )

    iq_imputer = joblib.load(
        IQ_IMPUTER_PATH
    )

    return (
        sj_model,
        iq_model,
        sj_imputer,
        iq_imputer
    )


# ============================================================
# Feature Engineering
# ============================================================

def create_features(df):

    data = df.copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    data = data.sort_values(
        [
            "city",
            "year",
            "weekofyear"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Month
    # --------------------------------------------------------

    data["month"] = (
        data["week_start_date"].dt.month
    )

    # --------------------------------------------------------
    # NDVI Mean
    # --------------------------------------------------------

    ndvi_columns = [
        "ndvi_ne",
        "ndvi_nw",
        "ndvi_se",
        "ndvi_sw"
    ]

    data["ndvi_mean"] = (
        data[ndvi_columns].mean(axis=1)
    )

    # --------------------------------------------------------
    # Temperature Range
    # --------------------------------------------------------

    data["temp_range"] = (
        data["reanalysis_max_air_temp_k"]
        -
        data["reanalysis_min_air_temp_k"]
    )

    # --------------------------------------------------------
    # Lag Features
    # --------------------------------------------------------

    for lag in [1, 2, 4, 12]:

        data[f"cases_lag_{lag}"] = (
            data.groupby("city")[
                "total_cases"
            ].shift(lag)
        )

    return data


# ============================================================
# Model Feature Columns
# ============================================================

FEATURE_COLUMNS = [

    "year",
    "weekofyear",

    "ndvi_ne",
    "ndvi_nw",
    "ndvi_se",
    "ndvi_sw",

    "precipitation_amt_mm",

    "reanalysis_air_temp_k",
    "reanalysis_avg_temp_k",
    "reanalysis_dew_point_temp_k",
    "reanalysis_max_air_temp_k",
    "reanalysis_min_air_temp_k",
    "reanalysis_precip_amt_kg_per_m2",
    "reanalysis_relative_humidity_percent",
    "reanalysis_sat_precip_amt_mm",
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_tdtr_k",

    "station_avg_temp_c",
    "station_diur_temp_rng_c",
    "station_max_temp_c",
    "station_min_temp_c",
    "station_precip_mm",

    "month",
    "ndvi_mean",
    "temp_range",

    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_4",
    "cases_lag_12"
]


# ============================================================
# Prepare City Data
# ============================================================

def prepare_city_data(
    df,
    city
):

    city_df = df[
        df["city"] == city
    ].copy()

    city_df = city_df.sort_values(
        [
            "year",
            "weekofyear"
        ]
    ).reset_index(drop=True)

    city_df = create_features(
        city_df
    )

    missing_columns = [
        col
        for col in FEATURE_COLUMNS
        if col not in city_df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing features for {city}: "
            f"{missing_columns}"
        )

    return city_df


# ============================================================
# Generate Predictions
# ============================================================

def generate_predictions(
    city_df,
    model,
    imputer
):

    prediction_df = city_df.copy()

    # --------------------------------------------------------
    # Lag features required by model
    # --------------------------------------------------------

    lag_columns = [
        "cases_lag_1",
        "cases_lag_2",
        "cases_lag_4",
        "cases_lag_12"
    ]

    # Remove rows without enough historical information
    valid_prediction_df = (
        prediction_df
        .dropna(
            subset=lag_columns
        )
        .copy()
    )

    if valid_prediction_df.empty:

        return valid_prediction_df

    # --------------------------------------------------------
    # Get exact features expected by saved imputer
    # --------------------------------------------------------

    if hasattr(
        imputer,
        "feature_names_in_"
    ):

        imputer_features = list(
            imputer.feature_names_in_
        )

    else:

        imputer_features = FEATURE_COLUMNS

    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    missing_features = [
        col
        for col in imputer_features
        if col not in valid_prediction_df.columns
    ]

    if missing_features:

        raise ValueError(
            "The following features required by "
            "the saved imputer are missing: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Select exact features
    # --------------------------------------------------------

    X = valid_prediction_df[
        imputer_features
    ].copy()

    # --------------------------------------------------------
    # Apply saved imputer
    # --------------------------------------------------------

    X_imputed = imputer.transform(
        X
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X_imputed
    )

    # Dengue cases cannot be negative
    valid_prediction_df[
        "predicted_cases"
    ] = np.maximum(
        predictions,
        0
    )

    return valid_prediction_df


# ============================================================
# Load Project
# ============================================================

try:

    df = load_data()

    (
        rf_sj,
        rf_iq,
        imputer_sj,
        imputer_iq
    ) = load_models_and_imputers()

except Exception as e:

    st.error(
        "Failed to load project data, models, or imputers."
    )

    st.exception(e)

    st.stop()


# ============================================================
# Header
# ============================================================

st.title("🦟 DengAI")

st.write(
    "### Dengue Cases Forecasting using Environmental, "
    "Temporal and Historical Epidemiological Data"
)

st.caption(
    "Explore historical dengue cases and Random Forest "
    "predictions for San Juan and Iquitos."
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title(
    "🦟 DengAI Controls"
)

st.sidebar.write(
    "Explore historical dengue cases and "
    "Random Forest predictions."
)


# ============================================================
# City Selection
# ============================================================

city_display = st.sidebar.selectbox(
    "Select City",
    [
        "San Juan",
        "Iquitos"
    ]
)


# Dataset uses city codes
city_mapping = {
    "San Juan": "sj",
    "Iquitos": "iq"
}

city = city_mapping[
    city_display
]


# ============================================================
# Select Model + Imputer
# ============================================================

if city == "sj":

    city_df = prepare_city_data(
        df,
        "sj"
    )

    model = rf_sj
    imputer = imputer_sj

else:

    city_df = prepare_city_data(
        df,
        "iq"
    )

    model = rf_iq
    imputer = imputer_iq


# ============================================================
# Validate City Data
# ============================================================

if city_df.empty:

    st.error(
        f"No historical data found for {city_display}."
    )

    st.stop()


# ============================================================
# Date Range
# ============================================================

valid_dates = (
    pd.to_datetime(
        city_df["week_start_date"],
        errors="coerce"
    )
    .dropna()
)


if valid_dates.empty:

    st.error(
        f"No valid dates found for {city_display}."
    )

    st.stop()


min_date = valid_dates.min()
max_date = valid_dates.max()


selected_dates = st.sidebar.date_input(
    "Select Time Period",

    value=(
        min_date.date(),
        max_date.date()
    ),

    min_value=min_date.date(),

    max_value=max_date.date()
)


# ============================================================
# Handle Date Selection
# ============================================================

if isinstance(
    selected_dates,
    tuple
):

    if len(selected_dates) == 2:

        start_date = selected_dates[0]
        end_date = selected_dates[1]

    else:

        start_date = selected_dates[0]
        end_date = selected_dates[0]

else:

    start_date = selected_dates
    end_date = selected_dates


# ============================================================
# Generate Predictions
# ============================================================

prediction_df = generate_predictions(
    city_df,
    model,
    imputer
)


# ============================================================
# Filter Date Range
# ============================================================

filtered_df = prediction_df[
    (
        prediction_df[
            "week_start_date"
        ].dt.date >= start_date
    )
    &
    (
        prediction_df[
            "week_start_date"
        ].dt.date <= end_date
    )
].copy()


# ============================================================
# Forecast Overview
# ============================================================

st.header("📊 Forecast Overview")


if not filtered_df.empty:

    total_actual = (
        filtered_df[
            "total_cases"
        ].sum()
    )

    total_predicted = (
        filtered_df[
            "predicted_cases"
        ].sum()
    )

    observations = len(
        filtered_df
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "City",
            city_display
        )

    with col2:

        st.metric(
            "Actual Cases",
            f"{total_actual:,.0f}"
        )

    with col3:

        st.metric(
            "Predicted Cases",
            f"{total_predicted:,.0f}"
        )

    with col4:

        st.metric(
            "Observations",
            f"{observations:,}"
        )


    # ========================================================
    # Difference
    # ========================================================

    difference = (
        total_predicted
        -
        total_actual
    )

    percentage_difference = (
        abs(difference)
        /
        total_actual
        *
        100
        if total_actual != 0
        else 0
    )

    st.caption(
        f"Prediction difference: "
        f"{difference:,.0f} cases "
        f"({percentage_difference:.2f}%)"
    )


    # ========================================================
    # Actual vs Predicted Chart
    # ========================================================

    st.header(
        "📈 Actual vs Predicted Dengue Cases"
    )

    fig = go.Figure()


    # --------------------------------------------------------
    # Actual
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=filtered_df[
                "week_start_date"
            ],

            y=filtered_df[
                "total_cases"
            ],

            mode="lines",

            name="Actual Cases",

            line=dict(
                width=3
            )
        )
    )


    # --------------------------------------------------------
    # Predicted
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=filtered_df[
                "week_start_date"
            ],

            y=filtered_df[
                "predicted_cases"
            ],

            mode="lines",

            name="Predicted Cases",

            line=dict(
                width=3,
                dash="dash"
            )
        )
    )


    fig.update_layout(
        template="plotly_dark",

        height=520,

        hovermode="x unified",

        xaxis_title="Date",

        yaxis_title="Dengue Cases",

        legend_title="Series",

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # Weekly Prediction Details
    # ========================================================

    st.header(
        "📋 Weekly Prediction Details"
    )


    display_df = filtered_df[
        [
            "week_start_date",
            "year",
            "weekofyear",
            "total_cases",
            "predicted_cases"
        ]
    ].copy()


    display_df.columns = [
        "Date",
        "Year",
        "Week",
        "Actual Cases",
        "Predicted Cases"
    ]


    display_df[
        "Predicted Cases"
    ] = (
        display_df[
            "Predicted Cases"
        ].round(2)
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


else:

    st.warning(
        "There are not enough historical observations "
        "with the required lag features for the selected period."
    )


# ============================================================
# Model Information
# ============================================================

st.header(
    "🤖 About the Model"
)

st.write(
    """
    DengAI uses separate **Random Forest Regression**
    models for San Juan and Iquitos.

    The final models use:

    - Temporal features
    - Environmental variables
    - Vegetation indicators
    - Temperature-related features
    - Historical dengue-case lag features

    Historical lag features include previous dengue cases
    from **1, 2, 4 and 12 weeks earlier**.

    The application uses the same saved Random Forest models
    and the same training-time imputers used during model
    development.
    """
)


# ============================================================
# Data Information
# ============================================================

st.header(
    "📁 Data Information"
)

st.info(
    f"""
    The application reads the project dataset directly
    from the `data/raw` directory.

    **Selected city:** {city_display}

    **Available historical period:**
    {min_date.date()} → {max_date.date()}

    The application does not generate artificial observations.
    """
)


# ============================================================
# Project Status
# ============================================================

st.header(
    "✅ Project Status"
)

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:

    st.success(
        "Dataset loaded"
    )

with status_col2:

    st.success(
        "Random Forest model loaded"
    )

with status_col3:

    st.success(
        "Imputer loaded"
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "DengAI • Dengue Forecasting Project • "
    "Random Forest Regression"
)