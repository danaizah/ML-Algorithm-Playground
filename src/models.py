import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator

MODELS: dict[str, type] = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "K-Nearest Neighbors": KNeighborsClassifier,
}

# Maps display name → list of hyperparameter specs consumed by the Streamlit UI.
# Each spec is a dict with keys: name, type, min, max, default, step.
HYPERPARAMS: dict[str, list[dict]] = {
    "Logistic Regression": [
        {"name": "C", "type": "float", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
        {"name": "max_iter", "type": "int", "min": 100, "max": 1000, "default": 100, "step": 50},
    ],
    "Decision Tree": [
        {"name": "max_depth", "type": "int", "min": 1, "max": 20, "default": 5, "step": 1},
        {"name": "min_samples_split", "type": "int", "min": 2, "max": 20, "default": 2, "step": 1},
    ],
    "Random Forest": [
        {"name": "n_estimators", "type": "int", "min": 10, "max": 300, "default": 100, "step": 10},
        {"name": "max_depth", "type": "int", "min": 1, "max": 20, "default": 5, "step": 1},
    ],
    "K-Nearest Neighbors": [
        {"name": "n_neighbors", "type": "int", "min": 1, "max": 20, "default": 5, "step": 1},
        {"name": "leaf_size", "type": "int", "min": 10, "max": 100, "default": 30, "step": 5},
    ],
}


def get_model_names() -> list[str]:
    """Return all available algorithm display names."""
    return list(MODELS.keys())


def get_hyperparams(model_name: str) -> list[dict]:
  
    return HYPERPARAMS.get(model_name, [])


def train(
    model_name: str,
    hyperparams: dict,
    X_train: np.ndarray,
    y_train: pd.Series,
) -> BaseEstimator:
 
    if model_name not in MODELS:
        raise KeyError(f"Unknown model: '{model_name}'. Choose from: {list(MODELS.keys())}")

    model_class = MODELS[model_name]
    model = model_class(**hyperparams, random_state=42) if _supports_random_state(model_class) else model_class(**hyperparams)
    model.fit(X_train, y_train)
    return model


def get_feature_importances(model: BaseEstimator, feature_names: list[str]) -> pd.DataFrame | None:
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        return None

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# --- Private helpers ---

def _supports_random_state(model_class: type) -> bool:
    """Check if a scikit-learn class accepts a random_state parameter."""
    import inspect
    sig = inspect.signature(model_class.__init__)
    return "random_state" in sig.parameters