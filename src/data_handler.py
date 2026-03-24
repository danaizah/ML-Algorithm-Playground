"""
Data loading and splitting utilities for tabular ML workflows.

Supports CSV and Excel files, and provides a sample Iris dataset
for quick experimentation.
"""
from pathlib import Path
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

def get_sample_dataset()-> pd.DataFrame:
    """Load the Iris dataset as a DataFrame with string target labels.

    Returns:
        A DataFrame with sepal/petal features and a 'target' column
        containing class names ('setosa', 'versicolor', 'virginica').
    """
    iris = load_iris(as_frame=True)
    df = iris.frame  # includes all features + target column
    df["target"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    return df


def load_file(file: object) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame.

    Args:
        file: A file containing the data to load.

    Returns:
        A DataFrame containing the file's data.

    Raises:
        ValueError: If the file extension is not supported or the file
                    cannot be parsed.
    """

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
    except (pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Could not read file '{file.name}': {exc}") from exc


def get_column_names(df: pd.DataFrame) -> list[str]:
    """Return all column names available as target candidates.

    Args:
        df: The input DataFrame.

    Returns:
        A list of column name strings.
    """
    
    return df.columns.tolist()


def data_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train/test sets.

    Args:
        df: The full dataset including the target column.
        target_col: Name of the column to use as the prediction target.
        test_size: Fraction of data to reserve for the test set.
        random_state: Seed for reproducibility.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).
    """
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# --- Private helpers ---

def _get_extension(filename: str) -> str:
    """Extract the lowercase file extension from a filename.

    Args:
        filename: The name of the file (e.g. 'data.csv').

    Returns:
        The extension, or an empty string if there is no extension.
    """
    return Path(filename).suffix.lower()