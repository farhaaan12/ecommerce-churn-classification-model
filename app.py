"""Streamlit app for churn scoring. Run: streamlit run app.py"""
import pandas as pd
import streamlit as st

from churn_service import load_model, predict_churn

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
st.title("E-Commerce Customer Churn Prediction")

try:
    bundle = load_model()
except Exception as e:  # missing model files, version mismatch, etc.
    st.error(f"Could not load the model: {e}")
    st.stop()

meta = bundle["metadata"]
with st.sidebar:
    st.header("Model info")
    st.write(f"**Model:** {meta['model_name']} ({meta['model_candidate']})")
    st.write(f"**Threshold:** {bundle['threshold']:.3f}")
    st.write(f"**Created:** {meta['created_at']}")
    if meta.get("unverified_features"):
        st.warning(f"{len(meta['unverified_features'])} feature(s) have unverified recording time "
                   "(temporal leakage risk). Confirm with the data team.")
    st.caption("Raw score = risk ranking. Use the flag/tier for decisions.")

ranges = bundle["training_ranges"]
cats = bundle["category_values"]


def num(label, col, default, step=1.0):
    r = ranges[col]
    return st.number_input(label, float(r["min"]), float(r["max"]), float(default), step=step)


def show(result):
    for w in result.attrs.get("warnings", []):
        st.warning(w)


tab1, tab2 = st.tabs(["Single customer", "Batch (CSV)"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure_unknown = st.checkbox("Tenure unknown")
        tenure = None if tenure_unknown else num("Tenure (months)", "Tenure", 10)
        wh_unknown = st.checkbox("WarehouseToHome unknown")
        wh = None if wh_unknown else num("Warehouse to home", "WarehouseToHome", 14)
        dso_unknown = st.checkbox("DaySinceLastOrder unknown")
        dso = None if dso_unknown else num("Days since last order", "DaySinceLastOrder", 3)
    with c2:
        devices = int(num("Devices registered", "NumberOfDeviceRegistered", 4))
        addresses = int(num("Number of addresses", "NumberOfAddress", 3))
        sat = int(num("Satisfaction score (1-5)", "SatisfactionScore", 3))
        cashback = num("Cashback amount", "CashbackAmount", 165, step=1.0)
    with c3:
        cat = st.selectbox("Preferred order category", cats["PreferedOrderCat"])
        marital = st.selectbox("Marital status", cats["MaritalStatus"])
        complain = int(st.radio("Has complained?", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True))

    if st.button("Predict", type="primary"):
        row = {"Tenure": tenure, "WarehouseToHome": wh, "NumberOfDeviceRegistered": devices,
               "PreferedOrderCat": cat, "SatisfactionScore": sat, "MaritalStatus": marital,
               "NumberOfAddress": addresses, "Complain": complain, "DaySinceLastOrder": dso,
               "CashbackAmount": cashback}
        df = pd.DataFrame([row]).astype({"Tenure": "float", "WarehouseToHome": "float", "DaySinceLastOrder": "float"})
        res = predict_churn(df, bundle).iloc[0]
        show(predict_churn(df, bundle))
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn score", f"{res.churn_score:.3f}")
        m2.metric("Risk tier", res.risk_tier)
        m3.metric("Retention action", "Intervene" if res.churn_flag else "No campaign")
        st.caption(f"Calibrated probability: {res.churn_probability_calibrated:.1%}")
        st.info(bundle["risk_actions"][res.risk_tier])

with tab2:
    st.write("Upload a CSV with exactly these columns: " + ", ".join(f"`{c}`" for c in bundle["feature_columns"]))
    up = st.file_uploader("CSV file", type="csv")
    if up:
        data = pd.read_csv(up)
        try:
            res = predict_churn(data, bundle)
        except ValueError as e:
            st.error(str(e))
        else:
            show(res)
            out = pd.concat([data.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
            st.write(f"{int(res.churn_flag.sum())} of {len(res)} customers flagged for intervention.")
            st.dataframe(out.sort_values("churn_score", ascending=False), use_container_width=True)
            st.download_button("Download results", out.to_csv(index=False).encode(), "churn_predictions.csv", "text/csv")
