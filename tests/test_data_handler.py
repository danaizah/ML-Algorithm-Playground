import pytest
import pandas as pd
from src.data_handler import get_sample_dataset, load_file, get_column_names, data_split


def test_get_sample_dataset():
    df = get_sample_dataset()
    assert set(df["target"].unique()) == {"setosa", "versicolor", "virginica"}

def test_load_file_unsupported_extension_raises_error(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("testing unsupported file extensions")
    with open(f) as file:
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(file)

def test_get_column_names_returns_list():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert get_column_names(df) == ["a", "b"]

def test_data_split_shapes():
    df = get_sample_dataset()
    X_train, X_test, y_train, y_test = data_split(df, "target", test_size=0.2, random_state=42)
    assert len(X_train) + len(X_test) == len(df)

def test_data_split_drops_target_column():
    df = get_sample_dataset()
    X_train, _, _, _ = data_split(df, "target", test_size=0.2, random_state=42)
    assert "target" not in X_train.columns

def test_data_split_reproducible_with_same_seed():
    df = get_sample_dataset()
    split_1 = data_split(df, "target", random_state=42)
    split_2 = data_split(df, "target", random_state=42)
    pd.testing.assert_frame_equal(split_1[0], split_2[0])

