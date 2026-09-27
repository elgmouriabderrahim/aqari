# aqari
Aqari is a Machine Learning project for the Moroccan real estate market that predicts house sale prices based on property characteristics. It includes data preprocessing, exploratory data analysis, feature engineering, regression model comparison, hyperparameter tuning, model evaluation, a Streamlit prediction interface, and Docker deployment.


The project covers the Machine Learning workflow, including:

* Data exploration and cleaning
* Data preprocessing
* Exploratory Data Analysis
* Feature Engineering
* Regression model training
* Model evaluation and comparison
* Cross-validation
* Hyperparameter tuning
* Model interpretation
* Streamlit application
* Docker deployment

## Project Goal

The goal is to build a regression model capable of predicting the `SalePrice` of a property from features such as surface area, number of rooms, construction year, quality, location, and other housing characteristics.

## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Joblib
* Docker

## Run the prediction app

From the project root, with your virtual environment activated:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The app loads `models/final_model.joblib` without retraining. This artifact is
ignored by Git, so it must be present locally. Keep `assets/` and
`data/raw/data_description.txt` available for the interface and field help.

Enter the main property details and review the prefilled additional fields, then
select **Estimate sale price**. The app calculates engineered features using
`src/feature_engineering.py` and matches the saved pipeline's input columns.
The current model uses Ames housing data, square feet, and US dollars; it does
not provide a current Moroccan market valuation.

## Run with Docker

Make sure `models/final_model.joblib` exists locally before building. It is
included in the Docker image even though Git ignores it. No model is retrained.

From the project root, build the image:

```bash
docker build -t aqari .
```

Start the container:

```bash
docker run --rm -d --name aqari -p 8501:8501 aqari
```

Open <http://localhost:8501> in your browser.

Stop the container:

```bash
docker stop aqari
```

The container is removed automatically when stopped; the image remains available.
The Dockerfile uses Python 3.14 and pins the model-related libraries to the
versions used by the saved pipeline. Other dependencies come from `requirements.txt`.

## Project Structure

```text
aqari/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── dashboard/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```
