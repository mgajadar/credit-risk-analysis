from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

@dataclass
class DataSplit:
    XTrain: pd.DataFrame
    XVal: pd.DataFrame
    XTest: pd.DataFrame
    yTrain: pd.Series
    yVal: pd.Series
    yTest: pd.Series
    preprocessor: ColumnTransformer

def buildPreprocessorAndSplit(df: pd.DataFrame, cfg) -> DataSplit:
    if "default" not in df.columns:
        raise ValueError("Expected 'default' target column not found.")

    X = df.drop(columns=["default"])
    y = df["default"].astype(int)

    # Split into train+temp and test
    XTrainFull, XTest, yTrainFull, yTest = train_test_split(
        X, y,
        test_size=cfg.testSize,
        random_state=cfg.randomSeed,
        stratify=y
    )

    # Split train into train and val
    XTrain, XVal, yTrain, yVal = train_test_split(
        XTrainFull, yTrainFull,
        test_size=cfg.valSize,
        random_state=cfg.randomSeed,
        stratify=yTrainFull
    )

    numericCols = XTrain.select_dtypes(include=["number"]).columns.tolist()
    categoricalCols = [c for c in XTrain.columns if c not in numericCols]

    numericPipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categoricalPipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numericPipe, numericCols),
            ("cat", categoricalPipe, categoricalCols)
        ],
        remainder="drop"
    )

    return DataSplit(
        XTrain=XTrain, XVal=XVal, XTest=XTest,
        yTrain=yTrain, yVal=yVal, yTest=yTest,
        preprocessor=preprocessor
    )
