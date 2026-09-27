import numpy as np


def add_features(df):
    df = df.copy()

    df["TotalArea"] = df[
        ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]
    ].sum(axis=1)

    df["TotalBathrooms"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    df["TotalPorchArea"] = (
        df["WoodDeckSF"]
        + df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
    )

    df.loc[df["HouseAge"] < 0, "HouseAge"] = np.nan
    df.loc[df["YearsSinceRemodel"] < 0, "YearsSinceRemodel"] = np.nan

    return df