import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Semantic Audience Studio", page_icon="🧠", layout="wide")
st.title("🧠 Semantic Audience Studio (Prototype)")
st.caption("Step 0: Campaign input (CSV upload or JSON paste) — read-only demo")

st.subheader("Campaign Input")

input_mode = st.radio("Choose input mode", ["Paste JSON", "Upload CSV"], horizontal=True)

campaigns_df = None

if input_mode == "Paste JSON":
    default_json = [
        {
            "campaign_id": "CAMP_VALENTINES",
            "campaign_name": "Valentine’s Day",
            "campaign_brief": "Romantic gifting + date-night bundles: fragrance, chocolates, candles, dinner-at-home, self-care sets."
        }
    ]
    json_text = st.text_area(
        "Paste campaign JSON (one campaign object or a list of objects)",
        value=json.dumps(default_json, indent=2),
        height=220
    )

    if st.button("Load JSON", type="primary"):
        try:
            obj = json.loads(json_text)

            # allow single dict or list
            if isinstance(obj, dict):
                obj = [obj]
            if not isinstance(obj, list) or len(obj) == 0:
                st.error("JSON must be a campaign object or a non-empty list of campaign objects.")
            else:
                campaigns_df = pd.DataFrame(obj)

        except Exception as e:
            st.error(f"Invalid JSON: {e}")

else:
    uploaded = st.file_uploader("Upload a campaigns CSV", type=["csv"])
    if uploaded is not None:
        try:
            campaigns_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# Validate + preview
required_cols = {"campaign_id", "campaign_name", "campaign_brief"}

if campaigns_df is not None:
    missing = required_cols - set(campaigns_df.columns)
    if missing:
        st.error(f"Missing columns: {sorted(list(missing))}. Required: {sorted(list(required_cols))}")
    else:
        st.success(f"✅ Loaded {len(campaigns_df)} campaign(s)")
        st.dataframe(campaigns_df, use_container_width=True)

        # store for next steps
        st.session_state["campaigns_df"] = campaigns_df

        # download back (optional)
        st.download_button(
            "Download campaigns.csv",
            data=campaigns_df.to_csv(index=False).encode("utf-8"),
            file_name="campaigns.csv",
            mime="text/csv"
        )
else:
    st.info("Provide campaign input to continue.")
