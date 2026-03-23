import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Columns with more missing values than this threshold are dropped entirely.
NULL_DROP_THRESHOLD = 0.5


def drop_high_null_columns(df: pd.DataFrame, threshold: float = NULL_DROP_THRESHOLD) -> pd.DataFrame:
  
    null_fraction = df.isnull().mean()
    cols_to_keep = null_fraction[null_fraction <= threshold].index.tolist()
    return df[cols_to_keep]


def encode_target(y: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    
    encoder = LabelEncoder()
    y_encoded = pd.Series(encoder.fit_transform(y), name=y.name)
    return y_encoded, encoder


def build_feature_pipeline(X: pd.DataFrame) -> ColumnTransformer:
  
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

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return transformer