import pandas as pd
from sklearn.model_selection import train_test_split

# Constants 
MIN_ROWS = 20
MIN_COLS = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_csv(file) -> pd.DataFrame:
    
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}") from exc

    return df


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


