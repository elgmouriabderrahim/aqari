from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_engineering import add_features


MODEL_PATH = ROOT / "models" / "final_model.joblib"
LOGO_PATH = ROOT / "assets" / "logo.png"


NEIGHBORHOODS = {
    "Blmngtn": "Bloomington Heights",
    "Blueste": "Bluestem",
    "BrDale": "Briardale",
    "BrkSide": "Brookside",
    "ClearCr": "Clear Creek",
    "CollgCr": "College Creek",
    "Crawfor": "Crawford",
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT and Railroad",
    "MeadowV": "Meadow Village",
    "Mitchel": "Mitchell",
    "NAmes": "North Ames",
    "NPkVill": "Northpark Villa",
    "NWAmes": "Northwest Ames",
    "NoRidge": "Northridge",
    "NridgHt": "Northridge Heights",
    "OldTown": "Old Town",
    "SWISU": "South & West Iowa State",
    "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West",
    "Somerst": "Somerset",
    "StoneBr": "Stone Brook",
    "Timber": "Timberland",
    "Veenker": "Veenker",
}


PROPERTY_TYPES = {
    "1Fam": "Detached house",
    "2fmCon": "Two-family conversion",
    "Duplex": "Duplex",
    "Twnhs": "Townhouse",
    "TwnhsE": "End townhouse",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_model_data(model):
    preprocessor = model.named_steps["preprocessor"]

    num_pipeline = preprocessor.named_transformers_["num"]
    cat_pipeline = preprocessor.named_transformers_["cat"]

    num_cols = list(preprocessor.transformers_[0][2])
    cat_cols = list(preprocessor.transformers_[1][2])

    defaults = dict(
        zip(
            num_cols,
            num_pipeline.named_steps["imputer"].statistics_
        )
    )

    defaults.update(
        zip(
            cat_cols,
            cat_pipeline.named_steps["imputer"].statistics_
        )
    )

    categories = dict(
        zip(
            cat_cols,
            cat_pipeline.named_steps["encoder"].categories_
        )
    )

    return defaults, categories


def build_input(values, defaults, expected_columns):
    row = defaults.copy()
    row.update(values)

    df = pd.DataFrame([row])

    df["GrLivArea"] = (
        df["1stFlrSF"]
        + df["2ndFlrSF"]
        + df["LowQualFinSF"]
    )

    df = add_features(df)

    return df[expected_columns]


st.set_page_config(
    page_title="Aqari · House Price Estimate",
    page_icon="🏠",
    layout="centered"
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #f6f7f9;
        color: black;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #000000 !important;
    }

    .block-container {
        max-width: 880px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #102d4e;
    }

    [data-testid="stForm"] {
        background: white;
        padding: 1.6rem;
        border: 1px solid #dce3eb;
        border-radius: 18px;
    }

    [data-testid="stBaseButton-primary"] {
        background: #153d63;
        border-color: #153d63;
        min-height: 48px;
        border-radius: 9px;
    }

    [data-testid="stMetric"] {
        background: #edf3f8;
        padding: 1rem;
        border-radius: 12px;
    }

    .brand {
        color: #47627d;
        font-size: .8rem;
        letter-spacing: 3px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Header
logo, heading = st.columns([1, 4], vertical_alignment="center")

with logo:
    st.image(str(LOGO_PATH), width=125)

with heading:
    st.markdown(
        '<div class="brand">AQARI · HOUSE PRICE ESTIMATOR</div>',
        unsafe_allow_html=True
    )
    st.title("What could this home sell for?")


st.write(
    "Describe the home below and AQARI will estimate its sale price."
)

st.info(
    "For homes in Ames, Iowa · Prices in US dollars · "
    "Estimate uses a 2010 sale date."
)


# Load model
model = load_model()
defaults, categories = get_model_data(model)


# Form
with st.form("prediction_form"):

    st.subheader("1. Where is the home?")

    left, right = st.columns(2)

    with left:
        neighborhood = st.selectbox(
            "Neighborhood in Ames",
            categories["Neighborhood"],
            format_func=lambda x: NEIGHBORHOODS.get(x, x)
        )

    with right:
        property_type = st.selectbox(
            "Type of home",
            categories["BldgType"],
            format_func=lambda x: PROPERTY_TYPES.get(x, x)
        )

    st.divider()

    st.subheader("2. How big is it?")

    left, right = st.columns(2)

    with left:
        lot_area = st.number_input(
            "Total land area (sq ft)",
            min_value=1,
            value=8000
        )

        first_floor = st.number_input(
            "Ground-floor living area (sq ft)",
            min_value=1,
            value=1000
        )

        bedrooms = st.number_input(
            "Number of bedrooms",
            min_value=0,
            value=3
        )

        full_bath = st.number_input(
            "Bathrooms with bath/shower",
            min_value=0,
            value=2
        )

    with right:
        garage_cars = st.number_input(
            "Garage capacity",
            min_value=0,
            value=2
        )

        second_floor = st.number_input(
            "Upper-floor living area (sq ft)",
            min_value=0,
            value=500
        )

        half_bath = st.number_input(
            "Extra toilet rooms",
            min_value=0,
            value=1
        )

    st.divider()

    st.subheader("3. Age & quality")

    overall_quality = st.slider(
        "Quality of materials and finish",
        1,
        10,
        5
    )

    left, right = st.columns(2)

    with left:
        year_built = st.number_input(
            "Year built",
            min_value=1800,
            max_value=2010,
            value=2000
        )

    with right:
        year_remodeled = st.number_input(
            "Last renovation year",
            min_value=1800,
            max_value=2010,
            value=2000
        )

    submitted = st.form_submit_button(
        "Estimate sale price",
        type="primary",
        use_container_width=True
    )


# Prediction
if submitted:

    if year_remodeled < year_built:
        st.error(
            "Renovation year cannot be earlier than construction year."
        )
        st.stop()

    values = {
        "Neighborhood": neighborhood,
        "BldgType": property_type,
        "OverallQual": overall_quality,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remodeled,
        "YrSold": 2010,
        "LotArea": lot_area,
        "1stFlrSF": first_floor,
        "2ndFlrSF": second_floor,
        "BedroomAbvGr": bedrooms,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "GarageCars": garage_cars,
    }

    input_df = build_input(
        values,
        defaults,
        model.feature_names_in_
    )

    prediction = model.predict(input_df)[0]

    st.markdown(f"## ${prediction:,.0f}")
