"""
Feature preprocessing utilities for building sklearn-compatible pipelines.

Handles null column removal, target encoding, and numeric/categorical
feature transformation.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

# Columns with more missing values than this threshold are dropped entirely.
NULL_DROP_THRESHOLD = 0.5


def drop_high_null_columns(
    df: pd.DataFrame,
    threshold: float = NULL_DROP_THRESHOLD,
) -> pd.DataFrame:
    """Drop columns whose fraction of missing values exceeds the threshold.

    Args:
        df: The input DataFrame.
        threshold: Maximum allowed fraction of null values.

    Returns:
        A DataFrame with high-null columns removed.
    """
  
    null_fraction = df.isnull().mean()
    cols_to_keep = null_fraction[null_fraction <= threshold].index.tolist()
    return df[cols_to_keep]


def encode_target(y: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    """Encode a categorical target series into integer labels.

    Args:
        y: The target column to encode.

    Returns:
        A tuple of (encoded Series, fitted LabelEncoder).
    """

    encoder = LabelEncoder()
    y_encoded = pd.Series(encoder.fit_transform(y), name=y.name)
    return y_encoded, encoder


def build_feature_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric and categorical features.

    Numeric columns are median-imputed and scaled. Categorical columns are
    most-frequent-imputed and ordinal-encoded. All other column types are dropped.

    Args:
        X: The feature DataFrame.

    Returns:
        An unfitted ColumnTransformer ready to be used in an sklearn Pipeline.
    """
  
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )