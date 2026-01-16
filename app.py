import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Semantic Audience Studio", page_icon="🧠", layout="wide")
st.title("🧠 Semantic Audience Studio (Prototype)")

# ===== STEP 0: Campaign Input =====
st.header("Step 0: Campaign Input")
st.caption("CSV upload or JSON paste")

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
    uploaded = st.file_uploader("Upload a campaigns CSV", type=["csv"], key="campaign_csv")
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

# ===== STEP 1: Upload Product & Transaction Tables =====
st.header("Step 1: Upload Product & Transaction Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Product Table")
    st.caption("Required columns: product_id, product_title, product_description")
    
    product_file = st.file_uploader("Upload Product CSV", type=["csv"], key="product_csv")
    
    if product_file is not None:
        try:
            product_df = pd.read_csv(product_file)
            
            # Normalize column names
            product_df.columns = product_df.columns.str.strip().str.lower()
            
            # Validate schema
            required_product_cols = ["product_id", "product_title", "product_description"]
            missing = set(required_product_cols) - set(product_df.columns)
            
            if missing:
                st.error(f"❌ Missing columns: {sorted(list(missing))}")
                st.info(f"Found columns: {product_df.columns.tolist()}")
            else:
                # Process product data
                product_df["product_id"] = product_df["product_id"].astype(str)
                product_df["product_text"] = (
                    product_df["product_title"].fillna("").astype(str) + " | " +
                    product_df["product_description"].fillna("").astype(str)
                ).str.lower()
                
                catalog_df = (
                    product_df[["product_id", "product_title", "product_text"]]
                    .drop_duplicates(subset=["product_id"])
                    .reset_index(drop=True)
                )
                catalog_df["product_name"] = catalog_df["product_title"]
                
                st.session_state["catalog_df"] = catalog_df
                st.success(f"✅ Loaded {len(catalog_df)} products")
                
                with st.expander("Preview Product Data"):
                    st.dataframe(catalog_df.head(10), use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error reading product CSV: {e}")

with col2:
    st.subheader("🛒 Transaction Table")
    st.caption("Required columns: tx_id, customer_id, product_id, tx_date, qty, price")
    
    txn_file = st.file_uploader("Upload Transaction CSV", type=["csv"], key="txn_csv")
    
    if txn_file is not None:
        try:
            txn_df = pd.read_csv(txn_file)
            
            # Normalize column names
            txn_df.columns = txn_df.columns.str.strip().str.lower()
            
            # Validate schema
            required_txn_cols = ["tx_id", "customer_id", "product_id", "tx_date", "qty", "price"]
            missing = set(required_txn_cols) - set(txn_df.columns)
            
            if missing:
                st.error(f"❌ Missing columns: {sorted(list(missing))}")
                st.info(f"Found columns: {txn_df.columns.tolist()}")
            else:
                # Process transaction data
                txn_df["customer_id"] = txn_df["customer_id"].astype(str)
                txn_df["product_id"] = txn_df["product_id"].astype(str)
                txn_df["tx_date"] = pd.to_datetime(txn_df["tx_date"], errors="coerce")
                txn_df["qty"] = pd.to_numeric(txn_df["qty"], errors="coerce").fillna(0.0)
                txn_df["price"] = pd.to_numeric(txn_df["price"], errors="coerce").fillna(0.0)
                txn_df["amt"] = txn_df["qty"] * txn_df["price"]
                
                st.session_state["txn_df"] = txn_df
                st.success(f"✅ Loaded {len(txn_df)} transactions")
                
                with st.expander("Preview Transaction Data"):
                    st.dataframe(txn_df.head(10), use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error reading transaction CSV: {e}")

st.divider()

# ===== Data Summary =====
if "catalog_df" in st.session_state and "txn_df" in st.session_state:
    catalog_df = st.session_state["catalog_df"]
    txn_df = st.session_state["txn_df"]
    
    st.header("📊 Data Summary")
    
    tab1, tab2, tab3 = st.tabs(["📦 Products", "🛒 Transactions", "📈 Analytics"])
    
    with tab1:
        st.subheader("Product Catalog")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Products", f"{len(catalog_df):,}")
        with col2:
            st.metric("Unique Product IDs", f"{catalog_df['product_id'].nunique():,}")
        with col3:
            st.metric("Columns", len(catalog_df.columns))
        
        st.dataframe(catalog_df, use_container_width=True, height=400)
        
        # Download button
        st.download_button(
            "📥 Download Processed Product Data",
            data=catalog_df.to_csv(index=False).encode("utf-8"),
            file_name="catalog_processed.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Transaction History")
        
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
        
        st.dataframe(txn_df, use_container_width=True, height=400)
        
        # Download button
        st.download_button(
            "📥 Download Processed Transaction Data",
            data=txn_df.to_csv(index=False).encode("utf-8"),
            file_name="transactions_processed.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("Quick Analytics")
        
        # Top customers
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
        
        # Top products
        st.write("**Top 10 Products by Transaction Count**")
        top_products = (
            txn_df.groupby("product_id")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_products.columns = ["Product ID", "Transaction Count"]
        
        # Merge with product names
        top_products = top_products.merge(
            catalog_df[["product_id", "product_name"]], 
            on="product_id", 
            how="left"
        )
        st.dataframe(top_products[["Product ID", "product_name", "Transaction Count"]], use_container_width=True)
        
else:
    st.info("👆 Upload both Product and Transaction CSV files to see the data summary.")
