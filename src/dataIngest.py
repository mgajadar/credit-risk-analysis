import pandas as pd
from sklearn.datasets import fetch_openml

def fetchCreditData(cfg) -> pd.DataFrame:
    """
    Fetch German Credit ("credit-g") from OpenML and return a dataframe with a binary target.
    OpenML returns target values 'good'/'bad' (or similar). We map to 0/1 where 1 = default/bad.
    """
    bunch = fetch_openml(name=cfg.openmlDatasetName, version=1, as_frame=True)
    df = bunch.frame.copy()

    # Target column name can vary; for credit-g it is typically "class"
    if "class" not in df.columns:
        raise ValueError(f"Expected target column 'class' not found. Columns: {df.columns.tolist()}")

    # Normalize target
    yRaw = df["class"].astype(str).str.lower().str.strip()
    # Common labels: "good"/"bad"
    df["default"] = (yRaw == "bad").astype(int)

    df = df.drop(columns=["class"])
    return df
