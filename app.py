"""
Streamlit entry point for the ML Algorithm Playground.

"""

import streamlit as st
import pandas as pd

from src import data_handler, preprocessing, models, metrics

# --- Page config ---
st.set_page_config(
    page_title="ML Playground",
    page_icon="🛝",
    layout="wide",
)

st.title("🛝 ML Algorithm Playground")
st.markdown("Upload a classification dataset, pick an algorithm, tune it, and explore the results.")


# Section 1 — Data Upload

st.header("1. Load Data")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("No file uploaded yet. Using the built-in sample dataset.")
    df = data_handler.get_sample_dataset()
else:
    try:
        df = data_handler.load_file(uploaded_file)
        st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    except ValueError as e:
        st.error(str(e))
        st.stop()

with st.expander("Preview data"):
    st.dataframe(df.head(20), use_container_width=True)


# Section 2 — Configuration

st.header("2. Configure")

col_left, col_right = st.columns(2)

with col_left:
    target_col = st.selectbox(
        "Target column (what to predict)",
        options=data_handler.get_column_names(df),
        index=len(df.columns) - 1,
    )

with col_right:
    model_name = st.selectbox("Algorithm", options=models.get_model_names())

# Hyperparameter sliders — generated dynamically from models.HYPERPARAMS
st.subheader("Hyperparameters")
hyperparam_specs = models.get_hyperparams(model_name)
chosen_hyperparams = {}

if hyperparam_specs:
    hp_cols = st.columns(len(hyperparam_specs))
    for col, spec in zip(hp_cols, hyperparam_specs):
        with col:
            if spec["type"] == "int":
                chosen_hyperparams[spec["name"]] = st.slider(
                    spec["name"],
                    min_value=spec["min"],
                    max_value=spec["max"],
                    value=spec["default"],
                    step=spec["step"],
                )
            else:
                chosen_hyperparams[spec["name"]] = st.slider(
                    spec["name"],
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=float(spec["default"]),
                    step=float(spec["step"]),
                )


# Section 3 — Train

st.header("3. Train & Evaluate")

if st.button("Train Model", type="primary"):
    with st.spinner("Preprocessing and training..."):
        # --- Preprocessing ---
        df_clean = preprocessing.drop_high_null_columns(df)
        X_train, X_test, y_train, y_test = data_handler.data_split(df_clean, target_col)
        y_train_enc, label_enc = preprocessing.encode_target(y_train)
        y_test_enc = pd.Series(label_enc.transform(y_test), name=y_test.name)
    
        feature_pipeline = preprocessing.build_feature_pipeline(X_train)
        X_train_proc = feature_pipeline.fit_transform(X_train)
        X_test_proc = feature_pipeline.transform(X_test)

        # --- Training ---
        model = models.train(model_name, chosen_hyperparams, X_train_proc, y_train_enc)
        y_pred = model.predict(X_test_proc)

        # --- Metrics ---
        summary = metrics.compute_summary(y_test_enc, y_pred)
        class_names = label_enc.classes_.astype(str).tolist()

    
    # Section 4 — Results
    
    st.header("4. Results")

    m1, m2 = st.columns(2)
    m1.metric("Accuracy", f"{summary['accuracy']:.2%}")
    m2.metric("F1 Score (weighted)", f"{summary['f1_weighted']:.2%}")

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.subheader("Confusion Matrix")
        fig_cm = metrics.plot_confusion_matrix(y_test_enc, y_pred, class_names)
        st.pyplot(fig_cm)

    with plot_col2:
        roc_fig = metrics.plot_roc_curve(model, X_test_proc, y_test_enc, class_names)
        if roc_fig:
            st.subheader("ROC Curve")
            st.pyplot(roc_fig)
        else:
            st.info(f"{model_name} does not support probability estimates — ROC curve unavailable.")

    feature_names = (
        X_train.select_dtypes(include=["number"]).columns.tolist()
        + X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    importance_df = models.get_feature_importances(model, feature_names)
    if importance_df is not None:
        st.subheader("Feature Importances")
        fig_imp = metrics.plot_feature_importance(importance_df)
        st.pyplot(fig_imp)

    with st.expander("Full classification report"):
        st.text(summary["classification_report"])


