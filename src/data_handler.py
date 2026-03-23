import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Constants 
TEST_SIZE = 0.2
RANDOM_STATE = 42

SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
}

def get_sample_dataset():
    iris = load_iris(as_frame=True)
    df = iris.frame  # includes all features + target column
    df["target"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    return df


def load_file(file) -> pd.DataFrame:

    extension = _get_extension(file.name)

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            f"Supported types: {list(SUPPORTED_EXTENSIONS.keys())}"
        )
    
    try:
        if SUPPORTED_EXTENSIONS[extension] == "csv":
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as exc:
        raise ValueError(f"Could not read file: {exc}") from exc


def get_column_names(df: pd.DataFrame) -> list[str]:
    
    #Return all column names available as target candidates
    
    return df.columns.tolist()


def data_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    #Split a DataFrame into train/test feature and target sets
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# --- Private helpers ---

def _get_extension(filename: str) -> str:
    """Extract the lowercase file extension from a filename."""
    return "." + filename.rsplit(".", 1)[-1].lower()