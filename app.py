import streamlit as st
import pandas as pd
import json
from google.cloud import bigquery
from google.oauth2 import service_account

st.set_page_config(page_title="Semantic Audience Studio", page_icon="🧠", layout="wide")
st.title("🧠 Semantic Audience Studio (Prototype)")

# ===== STEP 0: Campaign Input =====
st.header("Step 0: Campaign Input")
st.caption("CSV upload or JSON paste — read-only demo")

input_mode = st.radio("Choose input mode", ["Paste JSON", "Upload CSV"], horizontal=True)
campaigns_df = None

if input_mode == "Paste JSON":
    default_json = [
        {
            "campaign_id": "CAMP_VALENTINES",
            "campaign_name": "Valentine's Day",
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

# Validate + preview campaigns
required_cols = {"campaign_id", "campaign_name", "campaign_brief"}
if campaigns_df is not None:
    missing = required_cols - set(campaigns_df.columns)
    if missing:
        st.error(f"Missing columns: {sorted(list(missing))}. Required: {sorted(list(required_cols))}")
    else:
        st.success(f"✅ Loaded {len(campaigns_df)} campaign(s)")
        st.dataframe(campaigns_df, use_container_width=True)
        st.session_state["campaigns_df"] = campaigns_df
        st.download_button(
            "Download campaigns.csv",
            data=campaigns_df.to_csv(index=False).encode("utf-8"),
            file_name="campaigns.csv",
            mime="text/csv"
        )
else:
    st.info("Provide campaign input to continue.")

st.divider()

# ===== STEP 1: BigQuery Connection =====
st.header("Step 1: Connect to BigQuery")

# BigQuery configuration in sidebar
with st.sidebar:
    st.subheader("⚙️ BigQuery Configuration")
    
    auth_method = st.radio(
        "Authentication method",
        ["Service Account JSON", "Default Credentials"],
        help="Choose how to authenticate with BigQuery"
    )
    
    if auth_method == "Service Account JSON":
        credentials_json = st.text_area(
            "Paste Service Account JSON",
            height=150,
            help="Paste your GCP service account JSON key here"
        )
    
    project_id = st.text_input("Project ID", value="your-project-id")
    product_table = st.text_input(
        "Product Table", 
        value="your-project.dataset.product_table",
        help="Format: project.dataset.table"
    )
    transactions_table = st.text_input(
        "Transactions Table", 
        value="your-project.dataset.transactions_table",
        help="Format: project.dataset.table"
    )
    
    load_data = st.button("🔄 Load Data from BigQuery", type="primary")

# Initialize session state for data
if "catalog_df" not in st.session_state:
    st.session_state["catalog_df"] = None
if "txn_df" not in st.session_state:
    st.session_state["txn_df"] = None

# Load data when button is clicked
if load_data:
    try:
        with st.spinner("Connecting to BigQuery..."):
            # Setup BigQuery client
            if auth_method == "Service Account JSON":
                if not credentials_json:
                    st.error("Please paste your service account JSON")
                    st.stop()
                credentials_info = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(credentials_info)
                client = bigquery.Client(credentials=credentials, project=project_id)
            else:
                client = bigquery.Client(project=project_id)
            
            st.info("Loading product table...")
            product_df = client.query(f"SELECT * FROM `{product_table}`").to_dataframe()
            
            st.info("Loading transactions table...")
            txn_df = client.query(f"SELECT * FROM `{transactions_table}`").to_dataframe()
            
            # Normalize column names
            product_df.columns = product_df.columns.str.strip().str.lower()
            txn_df.columns = txn_df.columns.str.strip().str.lower()
            
            # ---- Validate Product Schema ----
            required_product_cols = ["product_id", "product_title", "product_description"]
            for c in required_product_cols:
                if c not in product_df.columns:
                    raise ValueError(f"Missing {c} in product_table. Found: {product_df.columns.tolist()}")
            
            product_df["product_id"] = product_df["product_id"].astype(str)
            product_df["product_text"] = (
                product_df["product_title"].fillna("").astype(str) + " | " +
                product_df["product_description"].fillna("").astype(str)
            ).str.lower()
            
            catalog_df = (
                product_df[["product_id", "product_title", "product_text"]]
                .rename(columns={"product_title": "product_name"})
                .drop_duplicates(subset=["product_id"])
                .reset_index(drop=True)
            )
            
            # ---- Validate Transaction Schema ----
            required_txn_cols = ["tx_id", "customer_id", "product_id", "tx_date", "qty", "price"]
            for c in required_txn_cols:
                if c not in txn_df.columns:
                    raise ValueError(f"Missing {c} in transaction_table. Found: {txn_df.columns.tolist()}")
            
            txn_df["customer_id"] = txn_df["customer_id"].astype(str)
            txn_df["product_id"] = txn_df["product_id"].astype(str)
            txn_df["tx_date"] = pd.to_datetime(txn_df["tx_date"], errors="coerce")
            txn_df["qty"] = pd.to_numeric(txn_df["qty"], errors="coerce").fillna(0.0)
            txn_df["price"] = pd.to_numeric(txn_df["price"], errors="coerce").fillna(0.0)
            txn_df["amt"] = txn_df["qty"] * txn_df["price"]
            
            # Store in session state
            st.session_state["catalog_df"] = catalog_df
            st.session_state["txn_df"] = txn_df
            
            st.success(f"✅ Data loaded successfully!")
            st.success(f"📦 Products: {catalog_df.shape[0]} rows, {catalog_df.shape[1]} columns")
            st.success(f"🛒 Transactions: {txn_df.shape[0]} rows, {txn_df.shape[1]} columns")
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

# Display loaded data
if st.session_state["catalog_df"] is not None and st.session_state["txn_df"] is not None:
    catalog_df = st.session_state["catalog_df"]
    txn_df = st.session_state["txn_df"]
    
    tab1, tab2 = st.tabs(["📦 Product Catalog", "🛒 Transactions"])
    
    with tab1:
        st.subheader("Product Catalog Preview")
        st.dataframe(catalog_df.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(catalog_df)} products")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Products", len(catalog_df))
        with col2:
            st.metric("Columns", len(catalog_df.columns))
        with col3:
            unique_products = catalog_df["product_id"].nunique()
            st.metric("Unique Product IDs", unique_products)
    
    with tab2:
        st.subheader("Transactions Preview")
        st.dataframe(txn_df.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(txn_df)} transactions")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(txn_df):,}")
        with col2:
            unique_customers = txn_df["customer_id"].nunique()
            st.metric("Unique Customers", f"{unique_customers:,}")
        with col3:
            total_revenue = txn_df["amt"].sum()
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        with col4:
            avg_order = txn_df["amt"].mean()
            st.metric("Avg Transaction", f"${avg_order:.2f}")
else:
    st.info("👆 Configure BigQuery settings in the sidebar and click 'Load Data' to continue.")
