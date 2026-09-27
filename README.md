# Aqari — House Price Prediction

Aqari estimates house sale prices using property characteristics. It covers data exploration, preprocessing, model comparison, evaluation, and a simple Streamlit app with Docker support.

## Business problem

Buyers, sellers, and real estate professionals need a starting point for estimating a property's value. Aqari uses size, quality, location, age, and other characteristics to produce an indicative price. The current implementation uses Ames, Iowa housing data; it is not a current Moroccan market valuation.

## Dataset

**House Prices - Advanced Regression Techniques** describes residential properties in Ames, Iowa. The target is **`SalePrice`**, measured in US dollars; area measurements are in square feet.

The local file, `data/raw/House_Prices.csv`, contains 2,919 rows and 81 columns. The training workflow uses the first 1,460 labeled rows, then removes two outliers, leaving 1,458 houses. `data/raw/data_description.txt` explains the variables and category codes.

## Project structure

```text
aqari/
├── assets/logo.png
├── dashboard/app.py                 # Streamlit interface
├── data/raw/                        # House_Prices.csv and field descriptions
├── models/final_model.joblib         # Saved pipeline; ignored by Git
├── notebooks/                       # Exploration through interpretation (01–11)
├── src/
│   ├── data_preparation.py           # Cleaning
│   ├── feature_engineering.py        # Derived features
│   └── preprocessing.py              # Numeric and categorical preprocessing
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Data preparation and EDA

`src/data_preparation.py` removes `Id`, treats `MSSubClass` as categorical, and drops rows 523 and 1298: unusually large homes with low sale prices. Selected missing amenity categories become `"None"`, selected numeric values become zero, and garage construction years later than the sale year become missing.

The fitted preprocessing pipeline applies median imputation with missing-value indicators and standard scaling to numeric columns. Categorical columns use most-frequent imputation and one-hot encoding, with unknown categories ignored. These steps are fitted within each training pipeline.

The [EDA notebook](notebooks/03_eda.ipynb) finds:

- Sale prices are right-skewed, with a smaller number of expensive homes.
- Larger living areas and higher overall quality are associated with higher prices.
- Prices vary by neighborhood, and newer homes generally sell for more.
- Garage capacity and basement area also correlate positively with price.

## Feature engineering

`src/feature_engineering.py` adds five features before the preprocessing pipeline:

| Feature | Calculation |
| --- | --- |
| `TotalArea` | Basement + first-floor + second-floor area |
| `TotalBathrooms` | Full bathrooms + half of half-bathroom counts, including basement bathrooms |
| `HouseAge` | Sale year − construction year |
| `YearsSinceRemodel` | Sale year − renovation year |
| `TotalPorchArea` | Sum of deck, open, enclosed, three-season, and screened porch areas |

Negative ages are replaced with missing values. The saved pipeline expects 84 input columns, including these derived features.

## Models and evaluation

The project compares **Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor** using the same preprocessing. Modeling notebooks also log metrics and models with MLflow.

- **MAE:** average absolute prediction error, in dollars. Lower is better.
- **RMSE:** square root of mean squared error, in dollars; it penalizes large errors more strongly. Lower is better.
- **R²:** improvement over predicting the target mean. Higher is better; 1 is perfect and negative values are possible. It is not an accuracy percentage.

### Holdout comparison

Recorded outputs in notebooks [05](notebooks/05_linear_regression.ipynb), [06](notebooks/06_random_forest.ipynb), and [07](notebooks/07_gradient_boosting_regression.ipynb), using an 80/20 split with `random_state=42`:

| Model | MAE ($) | RMSE ($) | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 21,406.71 | 66,603.95 | 0.1969 |
| Random Forest Regressor | 16,698.27 | 24,695.31 | 0.8896 |
| **Gradient Boosting Regressor** | **14,531.81** | **20,072.89** | **0.9271** |

### Cross-validation with KFold

[Notebook 08](notebooks/08_cross_validation.ipynb) evaluates the full cleaned dataset with `KFold(n_splits=5, shuffle=True, random_state=42)`. These are mean scores across folds, separate from the holdout results:

| Model | Mean MAE ($) | Mean RMSE ($) | Mean R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 18,613.28 | 36,355.34 | 0.7374 |
| Random Forest Regressor | 16,518.04 | 25,730.59 | 0.8936 |
| **Gradient Boosting Regressor** | **14,908.95** | **22,740.90** | **0.9173** |

Gradient Boosting's R² standard deviation is 0.0113, compared with 0.2706 for Linear Regression. This indicates more consistent performance across these folds.

### GridSearchCV and final selection

[Notebook 09](notebooks/09_gridsearchcv.ipynb) tunes Gradient Boosting on the training partition only, using five shuffled folds and negative RMSE scoring:

```python
param_grid = {
    "model__n_estimators": [100, 200, 300, 400, 500],
    "model__learning_rate": [0.02, 0.05, 0.07, 0.1, 0.2],
    "model__max_depth": [2, 3, 4, 5, 6],
}
```

The best combination uses 500 estimators, a learning rate of 0.1, and depth 2, with a training-fold CV RMSE of **$22,754.93**. Its holdout MAE is **$14,649.61**, RMSE **$20,788.84**, and R² **0.9218**.

The **original `GradientBoostingRegressor(random_state=42)`** is therefore selected: tuning did not improve the holdout results. The saved model has 100 estimators, learning rate 0.1, and depth 3. It remains fitted on the 80% training partition. Because the holdout informed model selection, it is not an untouched final validation set.

[Notebook 10](notebooks/10_final_evaluation.ipynb) also ranks the ten largest absolute prediction errors. The largest recorded error is approximately **$91,729**, showing that individual estimates can differ substantially from actual prices.

## Model interpretation

[Notebook 11](notebooks/11_model_interpretation.ipynb.ipynb) plots the ten leading features using the fitted model's `feature_importances_`:

| Feature | Importance |
| --- | ---: |
| Total area | 0.4371 |
| Overall quality | 0.3049 |
| Total bathrooms | 0.0344 |
| Garage capacity | 0.0255 |
| House age | 0.0233 |

These are relative tree-based importance scores, not causal effects or dollar contributions. Size and quality leading the ranking is consistent with the EDA.

## Saving and loading the final model

After training, notebook 10 saves the entire preprocessing and regression pipeline:

```python
# Run in notebooks/ after training gradient_boosting_pipeline.
import joblib
from pathlib import Path

Path("../models").mkdir(exist_ok=True)
joblib.dump(gradient_boosting_pipeline, "../models/final_model.joblib")
loaded_model = joblib.load("../models/final_model.joblib")
print(loaded_model.predict(X_test.iloc[[0]]))
# Recorded output: [213387.73737419]
```

The loaded pipeline can predict without retraining. Cleaning and feature engineering are separate functions in `src/`; the saved pipeline includes imputation, scaling, categorical encoding, and the trained regressor.

## Install and run locally

Use Python 3.14, matching the Docker environment. From the project root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt scikit-learn==1.9.1 joblib==1.6.0 numpy==2.5.3 pandas==3.0.6
pip install -e . --no-deps
streamlit run dashboard/app.py
```

Open **http://localhost:8501**. Stop the local app with **Ctrl+C**. The version pins match the saved model environment; the editable install makes `src` importable from notebooks.

Ensure `models/final_model.joblib` is present: Git ignores it, so a fresh clone needs a copy of the artifact or a run of the final evaluation notebook. The training notebooks currently contain a machine-specific MLflow SQLite path that must be adapted when rerunning elsewhere.

The app loads the model once, creates a one-row DataFrame, calculates engineered features, orders columns using `model.feature_names_in_`, and calls `model.predict(input_df)[0]`. It does not train a model. Unlisted inputs use fitted numeric medians and categorical modes, and the sale year is fixed at 2010.

### Run the notebooks

After installation, activate the environment and launch Jupyter from the project root:

```bash
source .venv/bin/activate
jupyter notebook
```

Open the local URL printed in the terminal, then open `notebooks/`. Select the
Python kernel from `.venv`. Run notebooks 01–11 in order to follow the full
workflow, or run `10_final_evaluation.ipynb` from top to bottom to train and save
the final pipeline. The notebooks use paths relative to `notebooks/` and require
`data/raw/House_Prices.csv`.

Before running a modeling notebook on another machine, replace its hard-coded
`mlflow.set_tracking_uri(...)` call with this portable version, executed from
the notebook directory:

```python
from pathlib import Path
mlflow.set_tracking_uri(f"sqlite:///{(Path.cwd().parent / 'mlflow.db').resolve()}")
```

This keeps experiment records in the project's root `mlflow.db`. Stop Jupyter
with **Ctrl+C** in its terminal and confirm shutdown if prompted.

### Open MLflow experiment tracking

In a separate terminal, from the project root:

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open **http://localhost:5000** and select **Aqari Model Comparison** to inspect
logged runs, parameters, and metrics. The notebooks and this UI must use the
same database. On a fresh clone, run the modeling notebooks first to create
experiment records: `mlflow.db` and local run artifacts are ignored by Git.

Stop MLflow with **Ctrl+C**. The notebooks log directly to SQLite, so the UI
does not need to be running during training. Streamlit also works independently
of MLflow; the Docker container runs only Streamlit.

### Example prediction

Keeping the app's initial selections and values gives this example:

| Input | Value |
| --- | --- |
| Neighborhood / type | Bloomington Heights / Detached house |
| Land area | 8,000 sq ft |
| Ground / upper-floor area | 1,000 / 500 sq ft |
| Bedrooms / full bathrooms / extra toilets | 3 / 2 / 1 |
| Garage capacity / quality | 2 cars / 5 out of 10 |
| Construction / renovation year | 2000 / 2000 |

With the current saved model and defaults for other fields, clicking **Estimate sale price** displays approximately **$163,628**. This is an illustrative historical estimate, not a current appraisal.

## Run with Docker

Ensure the saved model exists locally before building. From the project root:

```bash
# Build the image.
docker build -t aqari .

# Start the container.
docker run --rm -d --name aqari -p 8501:8501 aqari
```

Open **http://localhost:8501**. To stop and automatically remove the container:

```bash
docker stop aqari
```

The image uses Python 3.14 slim, installs `requirements.txt` with model-library version pins, and includes the saved model, `src/`, app, and logo. `.dockerignore` excludes local environments, training data, notebooks, and MLflow artifacts. No Docker Compose setup is needed.

## Screenshots and branding

`assets/` currently contains the project logo; no application screenshots have been added yet.

<img src="assets/logo.png" alt="Aqari house price predictor logo" width="220">
