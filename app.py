import streamlit as st
import pandas as pd
import json

# ============================================================================
# PAGE CONFIGURATION + GLOBAL STYLE
# ============================================================================
st.set_page_config(
    page_title="Semantic Audience Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.35rem;
        line-height: 1.1;
    }
    .sub-header {
        color: #555;
        margin-top: -0.15rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .step-header {
        padding: 1rem 1rem 0.85rem 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        margin: 1rem 0 0.5rem 0;
    }
    .step-header h2 {
        margin: 0;
        font-size: 1.35rem;
    }
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        background: #f8f9ff;
        margin: 0.75rem 0 1rem 0;
        color: #444;
        font-weight: 600;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0;
        color: #155724;
        font-weight: 600;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0;
        color: #0c5460;
        font-weight: 600;
    }
    .warn-box {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0;
        color: #856404;
        font-weight: 600;
    }
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border: 1px solid #eaeaea;
        height: 100%;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #222;
        line-height: 1.1;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 700;
        transition: all 0.25s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    [data-testid="stExpander"] summary {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

def step_header(title: str, caption: str | None = None):
    st.markdown(f'<div class="step-header"><h2>{title}</h2></div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)

def upload_zone(text: str):
    st.markdown(f'<div class="upload-zone">{text}</div>', unsafe_allow_html=True)

def success_box(text: str):
    st.markdown(f'<div class="success-box">✅ {text}</div>', unsafe_allow_html=True)

def info_box(text: str):
    st.markdown(f'<div class="info-box">ℹ️ {text}</div>', unsafe_allow_html=True)

def warn_box(text: str):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)

def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================================
# APP TITLE
# ============================================================================
st.markdown('<div class="main-header">🧠 Semantic Audience Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Customer Segmentation & Campaign Targeting Platform</div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown("### 📋 Navigation")
    st.markdown("---")

    steps_status = {
        "Step 0: Campaign Input": "campaigns_df" in st.session_state,
        "Step 1: Data Upload": ("catalog_df" in st.session_state) and ("txn_df" in st.session_state),
        "Step 2: Ontology": "ontology" in st.session_state,
        "Step 3: Campaign Profiles": ("campaign_intent_profile_df" in st.session_state) and (st.session_state.get("campaign_intent_profile_df") is not None),
        "Step 4: Product Labels": ("product_intent_labels_df" in st.session_state) and (st.session_state.get("product_intent_labels_df") is not None),
        "Step 5: Customer Profiles": ("customer_intent_profile_df" in st.session_state) and (st.session_state.get("customer_intent_profile_df") is not None),
        "Step 6: Audience Builder": ("campaign_audience_ranked_df" in st.session_state) and (st.session_state.get("campaign_audience_ranked_df") is not None),
    }

    for step, completed in steps_status.items():
        icon = "✅" if completed else "⭕"
        st.markdown(f"{icon} {step}")

    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info("Upload existing files to skip LLM generation and save API quota.")

    if st.button("🗑️ Clear All Session Data", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================================================
# STEP 0: CAMPAIGN INPUT
# ============================================================================
step_header("Step 0: Campaign Input", "CSV upload or JSON paste")
input_mode = st.radio("Choose input mode", ["Paste JSON", "Upload CSV"], horizontal=True)
campaigns_df = None

if input_mode == "Paste JSON":
    upload_zone("Paste your campaign JSON below, then click **Load JSON**.")
    default_json = [{
        "campaign_id": "CAMP_VALENTINES",
        "campaign_name": "Valentine's Day",
        "campaign_brief": "Romantic gifting + date-night bundles: fragrance, chocolates, candles, dinner-at-home, self-care sets."
    }]
    json_text = st.text_area(
        "Paste campaign JSON (one campaign object or a list of objects)",
        value=json.dumps(default_json, indent=2),
        height=220
    )
    if st.button("Load JSON", type="primary", key="load_campaign_json"):
        try:
            obj = json.loads(json_text)
            if isinstance(obj, dict):
                obj = [obj]
            if not isinstance(obj, list) or len(obj) == 0:
                st.error("JSON must be a campaign object or a non-empty list of campaign objects.")
            else:
                campaigns_df = pd.DataFrame(obj)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
else:
    upload_zone("Upload your **campaigns.csv** here.")
    uploaded = st.file_uploader("Upload a campaigns CSV", type=["csv"], key="campaign_csv")
    if uploaded is not None:
        try:
            campaigns_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

required_cols = {"campaign_id", "campaign_name", "campaign_brief"}
if campaigns_df is not None:
    missing = required_cols - set(campaigns_df.columns)
    if missing:
        st.error(f"Missing columns: {sorted(list(missing))}. Required: {sorted(list(required_cols))}")
    else:
        success_box(f"Loaded {len(campaigns_df)} campaign(s)")
        st.dataframe(campaigns_df, use_container_width=True)
        st.session_state["campaigns_df"] = campaigns_df
        st.download_button(
            "Download campaigns.csv",
            data=campaigns_df.to_csv(index=False).encode("utf-8"),
            file_name="campaigns.csv",
            mime="text/csv"
        )
else:
    info_box("Provide campaign input to continue.")

st.divider()

# ============================================================================
# STEP 1: PRODUCT & TRANSACTION DATA UPLOAD
# ============================================================================
step_header("Step 1: Upload Product & Transaction Data", "Upload both CSVs to unlock summaries + later steps")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Product Table")
    st.caption("Required columns: product_id, product_title, product_description")
    upload_zone("Drop your **Product CSV** here.")
    product_file = st.file_uploader("Upload Product CSV", type=["csv"], key="product_csv")

    if product_file is not None:
        try:
            product_df = pd.read_csv(product_file)
            product_df.columns = product_df.columns.str.strip().str.lower()

            required_product_cols = ["product_id", "product_title", "product_description"]
            missing = set(required_product_cols) - set(product_df.columns)

            if missing:
                st.error(f"❌ Missing columns: {sorted(list(missing))}")
                st.info(f"Found columns: {product_df.columns.tolist()}")
            else:
                product_df["product_id"] = product_df["product_id"].astype(str)
                product_df["product_text"] = (
                    product_df["product_title"].fillna("").astype(str) + " | " +
                    product_df["product_description"].fillna("").astype(str)
                ).str.lower()

                catalog_df = product_df[["product_id", "product_title", "product_text"]].copy()
                catalog_df = catalog_df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
                catalog_df["product_name"] = catalog_df["product_title"]

                st.session_state["catalog_df"] = catalog_df
                success_box(f"Loaded {len(catalog_df)} products")

                with st.expander("Preview Product Data"):
                    st.dataframe(catalog_df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Error reading product CSV: {e}")

with col2:
    st.subheader("🛒 Transaction Table")
    st.caption("Required columns: tx_id, customer_id, product_id, tx_date, qty, price")
    upload_zone("Drop your **Transaction CSV** here.")
    txn_file = st.file_uploader("Upload Transaction CSV", type=["csv"], key="txn_csv")

    if txn_file is not None:
        try:
            txn_df = pd.read_csv(txn_file)
            txn_df.columns = txn_df.columns.str.strip().str.lower()

            required_txn_cols = ["tx_id", "customer_id", "product_id", "tx_date", "qty", "price"]
            missing = set(required_txn_cols) - set(txn_df.columns)

            if missing:
                st.error(f"❌ Missing columns: {sorted(list(missing))}")
                st.info(f"Found columns: {txn_df.columns.tolist()}")
            else:
                txn_df["customer_id"] = txn_df["customer_id"].astype(str)
                txn_df["product_id"] = txn_df["product_id"].astype(str)
                txn_df["tx_date"] = pd.to_datetime(txn_df["tx_date"], errors="coerce")
                txn_df["qty"] = pd.to_numeric(txn_df["qty"], errors="coerce").fillna(0.0)
                txn_df["price"] = pd.to_numeric(txn_df["price"], errors="coerce").fillna(0.0)
                txn_df["amt"] = txn_df["qty"] * txn_df["price"]

                st.session_state["txn_df"] = txn_df
                success_box(f"Loaded {len(txn_df)} transactions")

                with st.expander("Preview Transaction Data"):
                    st.dataframe(txn_df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Error reading transaction CSV: {e}")

st.divider()

# ============================================================================
# DATA SUMMARY & ANALYTICS
# ============================================================================
if "catalog_df" in st.session_state and "txn_df" in st.session_state:
    catalog_df = st.session_state["catalog_df"]
    txn_df = st.session_state["txn_df"]

    step_header("📊 Data Summary", "Quick sanity checks before the AI steps")

    tab1, tab2, tab3 = st.tabs(["📦 Products", "🛒 Transactions", "📈 Analytics"])

    with tab1:
        st.subheader("Product Catalog")
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Total Products", f"{len(catalog_df):,}")
        with c2: metric_card("Unique Product IDs", f"{catalog_df['product_id'].nunique():,}")
        with c3: metric_card("Columns", f"{len(catalog_df.columns):,}")

        st.dataframe(catalog_df, use_container_width=True, height=400)
        st.download_button(
            "📥 Download Processed Product Data",
            data=catalog_df.to_csv(index=False).encode("utf-8"),
            file_name="catalog_processed.csv",
            mime="text/csv"
        )

    with tab2:
        st.subheader("Transaction History")
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Total Transactions", f"{len(txn_df):,}")
        with c2: metric_card("Unique Customers", f"{txn_df['customer_id'].nunique():,}")
        with c3: metric_card("Total Revenue", f"${txn_df['amt'].sum():,.2f}")
        with c4: metric_card("Avg Transaction", f"${txn_df['amt'].mean():.2f}")

        st.dataframe(txn_df, use_container_width=True, height=400)
        st.download_button(
            "📥 Download Processed Transaction Data",
            data=txn_df.to_csv(index=False).encode("utf-8"),
            file_name="transactions_processed.csv",
            mime="text/csv"
        )

    with tab3:
        st.subheader("Quick Analytics")

        st.write("**Top 10 Customers by Revenue**")
        top_customers = (
            txn_df.groupby("customer_id")["amt"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_customers.columns = ["Customer ID", "Total Spent"]
        st.dataframe(top_customers, use_container_width=True)

        st.write("**Top 10 Products by Transaction Count**")
        top_products = (
            txn_df.groupby("product_id")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="transaction_count")
        )

        top_products = top_products.merge(
            catalog_df[["product_id", "product_name"]],
            on="product_id",
            how="left"
        ).rename(columns={
            "product_id": "Product ID",
            "product_name": "Product Name",
            "transaction_count": "Transaction Count"
        })

        st.dataframe(top_products[["Product ID", "Product Name", "Transaction Count"]], use_container_width=True)

else:
    info_box("Upload both Product and Transaction CSV files to see the data summary.")

st.divider()

# ---- NEXT: Part 2 continues with Step 2 (Ontology) ----
