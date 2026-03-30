import pytest
import pandas as pd
from src.preprocessing import drop_high_null_columns, encode_target, build_feature_pipeline

def test_drop_high_null_columns():
    df = pd.DataFrame({
        "a": [1, None, None, None],
        "b": [1, 2, 3, 4],
        "c": [1,None,2,None]
    })
    result = drop_high_null_columns(df, threshold=0.5)
    assert "a" not in result.columns
    assert "b" in result.columns
    assert "c" in result.columns

def test_encode_target_encoder():
    y = pd.Series(["cat","fish","cat","bird"])
    _, encoder = encode_target(y)
    assert set(encoder.classes_) == {"fish","bird","cat"}




