import streamlit as st
import pandas as pd
import json

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Semantic Audience Studio", page_icon="🧠", layout="wide")
st.title("🧠 Semantic Audience Studio (Prototype)")


# ============================================================================
# STEP 0: CAMPAIGN INPUT
# ============================================================================
st.header("Step 0: Campaign Input")
st.caption("CSV upload or JSON paste")

input_mode = st.radio("Choose input mode", ["Paste JSON", "Upload CSV"], horizontal=True)
campaigns_df = None

if input_mode == "Paste JSON":
    default_json = [
  {
    "campaign_id": "CAMP_VALENTINES",
    "campaign_name": "Valentine’s Day",
    "campaign_brief": "Romantic gifting + date-night bundles: fragrance, chocolates, candles, dinner-at-home, self-care sets."
  },
  {
    "campaign_id": "CAMP_MOTHERSDAY",
    "campaign_name": "Mother’s Day",
    "campaign_brief": "Gifts for moms/parents + care & comfort: home items, personal care, gratitude-themed premium gifts."
  },
  {
    "campaign_id": "CAMP_FATHERSDAY",
    "campaign_name": "Father’s Day",
    "campaign_brief": "Gifts for dads/men + practical utility: grooming kits, tech accessories, hobby-related items."
  },
  {
    "campaign_id": "CAMP_GRAD",
    "campaign_name": "Graduation Season",
    "campaign_brief": "Congrats gifting + career transition: productivity tech, professional bags, style essentials, keepsakes."
  },
  {
    "campaign_id": "CAMP_WEDDING",
    "campaign_name": "Wedding Season",
    "campaign_brief": "Wedding gifting + home setup: premium bundles, couple-focused gifts, celebration essentials."
  },
  {
    "campaign_id": "CAMP_LUNAR_NEWYEAR",
    "campaign_name": "Lunar New Year",
    "campaign_brief": "Family reunion + home hosting: auspicious gifting, home refresh items, festive celebration supplies."
  },
  {
    "campaign_id": "CAMP_RAMADAN_EID",
    "campaign_name": "Ramadan & Eid",
    "campaign_brief": "Family gatherings + culturally sensitive hosting: food prep, personal care, Eid gifting traditions."
  },
  {
    "campaign_id": "CAMP_DIWALI",
    "campaign_name": "Diwali",
    "campaign_brief": "Festive home decoration + family celebration: lights, new outfits, gifting, traditional home items."
  },
  {
    "campaign_id": "CAMP_EASTER",
    "campaign_name": "Easter",
    "campaign_brief": "Family gathering + spring refresh: home decor, seasonal sweets, small gift bundles."
  },
  {
    "campaign_id": "CAMP_HALLOWEEN",
    "campaign_name": "Halloween",
    "campaign_brief": "Costumes + party hosting: festive decorations, themed items, treats and confectionery."
  },
  {
    "campaign_id": "CAMP_THANKSGIVING",
    "campaign_name": "Thanksgiving",
    "campaign_brief": "Hosting meals + kitchen prep: home warmth and comfort, family gathering essentials."
  },
  {
    "campaign_id": "CAMP_BACKTOSCHOOL",
    "campaign_name": "Back to School",
    "campaign_brief": "Kids school needs + study productivity: stationery, bags, learning gadgets, organizational tools."
  },
  {
    "campaign_id": "CAMP_SUMMERTRAVEL",
    "campaign_name": "Summer Travel / Vacation",
    "campaign_brief": "Travel prep + outdoor essentials: sun protection, beach gear, lightweight packing sets."
  },
  {
    "campaign_id": "CAMP_WINTERWARM",
    "campaign_name": "Winter Warm-Up",
    "campaign_brief": "Cold-season wellness + cozy home: warm clothing, protective gear, wellness and comfort items."
  },
  {
    "campaign_id": "CAMP_SPRINGCLEAN",
    "campaign_name": "Spring Cleaning / Home Refresh",
    "campaign_brief": "Home improvement + organization: cleaning supplies, storage solutions, decluttering tools."
  },
  {
    "campaign_id": "CAMP_BFCM",
    "campaign_name": "Black Friday / Cyber Monday",
    "campaign_brief": "Deep deals + category upgrades: online shopping peak, high-value tech, stock-up bundles."
  },
  {
    "campaign_id": "CAMP_PRIMEDAY",
    "campaign_name": "Prime Day (Deal Event)",
    "campaign_brief": "Mid-year festival + home essentials: tech upgrades, impulse buys, high-discount favorites."
  },
  {
    "campaign_id": "CAMP_1111",
    "campaign_name": "Singles’ Day (11.11)",
    "campaign_brief": "Self-reward + gift stock-up: big online deals, personal upgrades, bundled promotions."
  },
  {
    "campaign_id": "CAMP_618",
    "campaign_name": "618 Mid-Year Festival",
    "campaign_brief": "Mid-year platform sale + stock-up: category promotions, hardware upgrades, volume deals."
  },
  {
    "campaign_id": "CAMP_CHRISTMAS",
    "campaign_name": "Christmas / Year-End Holidays",
    "campaign_brief": "Gift-giving + festive hosting: home decor, family celebration sets, premium style gifts."
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


# ============================================================================
# STEP 1: PRODUCT & TRANSACTION DATA UPLOAD
# ============================================================================
st.header("Step 1: Upload Product & Transaction Data")

col1, col2 = st.columns(2)

# --- PRODUCT TABLE UPLOAD ---
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

# --- TRANSACTION TABLE UPLOAD ---
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


# ============================================================================
# DATA SUMMARY & ANALYTICS
# ============================================================================
if "catalog_df" in st.session_state and "txn_df" in st.session_state:
    catalog_df = st.session_state["catalog_df"]
    txn_df = st.session_state["txn_df"]
    
    st.header("📊 Data Summary")
    
    tab1, tab2, tab3 = st.tabs(["📦 Products", "🛒 Transactions", "📈 Analytics"])
    
    # --- PRODUCTS TAB ---
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
        
        st.download_button(
            "📥 Download Processed Product Data",
            data=catalog_df.to_csv(index=False).encode("utf-8"),
            file_name="catalog_processed.csv",
            mime="text/csv"
        )
    
    # --- TRANSACTIONS TAB ---
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
        
        st.download_button(
            "📥 Download Processed Transaction Data",
            data=txn_df.to_csv(index=False).encode("utf-8"),
            file_name="transactions_processed.csv",
            mime="text/csv"
        )
    
    # --- ANALYTICS TAB ---
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

st.divider()


# ============================================================================
# STEP 2: ONTOLOGY GENERATION (OPTIONAL)
# ============================================================================
st.header("Step 2: Generate Ontology & Dimensions (Optional)")

if "catalog_df" in st.session_state:
    catalog_df = st.session_state["catalog_df"]
    
    col1, col2 = st.columns([2, 1])
    
    # --- ONTOLOGY CONFIGURATION ---
    with col1:
        st.subheader("Configuration")
        
        st.write("**Ontology Settings**")
        ontology_name = st.text_input("Ontology Name", value="Product Ontology v1")
        ontology_version = st.text_input("Version", value="1.0")
        
        st.write("**Lifestyle Dimensions**")
        lifestyle_input = st.text_area(
            "Enter lifestyle categories (one per line)",
            value="Health & Wellness\nHome & Living\nFashion & Beauty\nFood & Beverage\nTechnology\nSports & Fitness",
            height=150
        )
        
        st.write("**Intent Dimensions**")
        intent_input = st.text_area(
            "Enter intent categories (one per line)",
            value="Gift Giving\nSelf Care\nHome Improvement\nDaily Essentials\nSpecial Occasions\nExploration",
            height=150
        )
    
    # --- GENERATION ACTIONS ---
    with col2:
        st.subheader("Actions")
        
        generate_btn = st.button("🔨 Generate Ontology", type="primary", use_container_width=True)
        
        st.info("This will create:\n- Ontology JSON\n- Lifestyle CSV\n- Intent CSV")
    
    # --- GENERATE ONTOLOGY ---
    if generate_btn:
        lifestyle_categories = [line.strip() for line in lifestyle_input.split("\n") if line.strip()]
        intent_categories = [line.strip() for line in intent_input.split("\n") if line.strip()]
        
        with st.spinner("Generating ontology and dimensions..."):
            
            # Create ontology structure
            ontology = {
                "name": ontology_name,
                "version": ontology_version,
                "created_at": pd.Timestamp.now().isoformat(),
                "total_products": len(catalog_df),
                "lifestyle_dimensions": lifestyle_categories,
                "intent_dimensions": intent_categories,
                "metadata": {
                    "description": "Auto-generated product ontology",
                    "source": "product_catalog"
                }
            }
            
            # Create lifestyle dimension table
            lifestyle_data = []
            for idx, category in enumerate(lifestyle_categories, 1):
                lifestyle_data.append({
                    "lifestyle_id": f"LIFE_{idx:03d}",
                    "lifestyle_name": category,
                    "lifestyle_description": f"Products related to {category.lower()}",
                    "product_count": 0
                })
            
            dim_lifestyle_df = pd.DataFrame(lifestyle_data)
            
            # Create intent dimension table
            intent_data = []
            for idx, category in enumerate(intent_categories, 1):
                intent_data.append({
                    "intent_id": f"INT_{idx:03d}",
                    "intent_name": category,
                    "intent_description": f"Products for {category.lower()} purposes",
                    "product_count": 0
                })
            
            dim_intent_df = pd.DataFrame(intent_data)
            
            # Store in session state
            st.session_state["ontology"] = ontology
            st.session_state["dim_lifestyle_df"] = dim_lifestyle_df
            st.session_state["dim_intent_df"] = dim_intent_df
            
            st.success("✅ Ontology generated successfully!")
    
    # --- DISPLAY & DOWNLOAD ONTOLOGY ---
    if "ontology" in st.session_state:
        st.divider()
        st.subheader("📥 Download Ontology Files")
        
        ontology = st.session_state["ontology"]
        dim_lifestyle_df = st.session_state["dim_lifestyle_df"]
        dim_intent_df = st.session_state["dim_intent_df"]
        
        tab1, tab2, tab3 = st.tabs(["📋 Ontology JSON", "🎨 Lifestyle Dimensions", "🎯 Intent Dimensions"])
        
        with tab1:
            st.json(ontology)
            st.download_button(
                label="📥 Download ontology_v1.json",
                data=json.dumps(ontology, ensure_ascii=False, indent=2),
                file_name="ontology_v1.json",
                mime="application/json",
                use_container_width=True
            )
        
        with tab2:
            st.dataframe(dim_lifestyle_df, use_container_width=True)
            st.download_button(
                label="📥 Download dim_lifestyle_v1.csv",
                data=dim_lifestyle_df.to_csv(index=False).encode("utf-8"),
                file_name="dim_lifestyle_v1.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with tab3:
            st.dataframe(dim_intent_df, use_container_width=True)
            st.download_button(
                label="📥 Download dim_intent_v1.csv",
                data=dim_intent_df.to_csv(index=False).encode("utf-8"),
                file_name="dim_intent_v1.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.info("👆 Upload Product CSV in Step 1 to enable ontology generation.")


# ============================================================================
# END OF APP
# ============================================================================
