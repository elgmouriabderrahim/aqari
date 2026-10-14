import numpy as np


def clean_data(df):
    df = df.iloc[:1460].copy()

    # remove outliers that has higher GrLivArea with less saleprice
    df = df.drop(index=[523, 1298], errors="ignore")

    df = df.drop(columns="Id")

    # MSSubClass is a category code
    df["MSSubClass"] = df["MSSubClass"].astype(str)

    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageQual", "GarageCond", "GarageFinish", "GarageType",
        "BsmtCond", "BsmtQual", "BsmtFinType1"
    ]

    df[none_cols] = df[none_cols].fillna("None")

    # fill missing values in num columns with 0
    zero_cols = [
        "BsmtFullBath", "BsmtHalfBath", "BsmtFinSF1",
        "GarageArea", "GarageCars"
    ]

    df[zero_cols] = df[zero_cols].fillna(0)

    # if the garage built year greater than the sold year mark the values as missing
    df.loc[df["GarageYrBlt"] > df["YrSold"], "GarageYrBlt"] = np.nan

    return df