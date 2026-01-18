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

# ============================================================================
# STEP 2: AI-POWERED ONTOLOGY GENERATION (OR REUSE VIA UPLOAD)
# ============================================================================
step_header("Step 2: AI-Powered Ontology Generation",
            "Option A: Generate with Gemini • Option B: Upload ontology_v1.json to reuse (recommended for demos / save quota)")

if "catalog_df" in st.session_state:
    catalog_df = st.session_state["catalog_df"]

    st.subheader("♻️ Reuse Existing Ontology (Upload)")
    with st.expander("Upload ontology_v1.json to reuse (recommended for demos / save quota)", expanded=True):
        upload_zone("Upload your **ontology_v1.json** here (no API key needed).")
        uploaded_ontology_json = st.file_uploader(
            "Upload ontology JSON (e.g., ontology_v1.json)",
            type=["json"],
            key="upload_ontology_json"
        )

        col_u1, col_u2, col_u3 = st.columns([1, 1, 2])
        with col_u1:
            load_uploaded_ontology_btn = st.button("Load Uploaded Ontology", type="primary", key="load_uploaded_ontology_btn")
        with col_u2:
            clear_ontology_btn = st.button("Clear Ontology (session)", key="clear_ontology_btn")
        with col_u3:
            st.caption("Loads ontology + rebuilds dim_lifestyle and dim_intent in session_state. No API key needed.")

        if clear_ontology_btn:
            for k in ["ontology", "dim_lifestyle_df", "dim_intent_df"]:
                if k in st.session_state:
                    del st.session_state[k]
            success_box("Cleared ontology from session_state.")

        if load_uploaded_ontology_btn:
            if uploaded_ontology_json is None:
                st.error("Please upload an ontology JSON file first.")
            else:
                try:
                    ontology = json.loads(uploaded_ontology_json.read().decode("utf-8"))

                    if not isinstance(ontology, dict) or "lifestyles" not in ontology:
                        raise ValueError("Invalid ontology JSON. Expected a dict with key: 'lifestyles'.")

                    lifestyles = ontology.get("lifestyles", [])
                    if not isinstance(lifestyles, list) or len(lifestyles) == 0:
                        raise ValueError("Ontology JSON has no lifestyles (empty or invalid).")

                    dim_lifestyle_rows, dim_intent_rows = [], []
                    version = str(ontology.get("version", "v1"))

                    for ls in lifestyles:
                        ls_id = ls.get("lifestyle_id", "")
                        ls_name = ls.get("lifestyle_name", "")
                        ls_def = ls.get("definition", "")

                        dim_lifestyle_rows.append({
                            "lifestyle_id": ls_id,
                            "lifestyle_name": ls_name,
                            "definition": ls_def,
                            "version": version
                        })

                        for it in ls.get("intents", []) or []:
                            dim_intent_rows.append({
                                "intent_id": it.get("intent_id", ""),
                                "intent_name": it.get("intent_name", ""),
                                "definition": it.get("definition", ""),
                                "lifestyle_id": ls_id,
                                "include_examples": json.dumps(it.get("include_examples", []), ensure_ascii=False),
                                "exclude_examples": json.dumps(it.get("exclude_examples", []), ensure_ascii=False),
                                "version": version
                            })

                    dim_lifestyle_df = pd.DataFrame(dim_lifestyle_rows).drop_duplicates()
                    dim_intent_df = pd.DataFrame(dim_intent_rows).drop_duplicates()

                    st.session_state["ontology"] = ontology
                    st.session_state["dim_lifestyle_df"] = dim_lifestyle_df
                    st.session_state["dim_intent_df"] = dim_intent_df

                    success_box(f"Loaded ontology: {len(dim_lifestyle_df)} lifestyles, {len(dim_intent_df)} intents")

                except Exception as e:
                    st.error(f"❌ Failed to load uploaded ontology: {e}")

    st.divider()

    st.subheader("🤖 Generate Ontology with AI (Gemini)")
    st.caption("Only needed if you do not upload an existing ontology above.")

    gemini_api_key = None
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        success_box("Gemini API key loaded from secrets")
    else:
        gemini_api_key = st.text_input(
            "Enter your Gemini API Key (only needed if you generate)",
            type="password",
            help="Get your API key from https://aistudio.google.com/apikey",
            key="gemini_key_step2"
        )
        if gemini_api_key:
            info_box("Tip: Add your API key to Streamlit secrets for persistence")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("**Ontology Settings**")
        n_lifestyles = st.number_input("Number of Lifestyle Categories", min_value=3, max_value=15, value=6, key="step2_n_lifestyles")
        max_intents_per_lifestyle = st.number_input("Max Intents per Lifestyle", min_value=2, max_value=10, value=5, key="step2_max_intents")
        chunk_size = st.number_input("Chunk Size (products per API call)", min_value=20, max_value=100, value=40, key="step2_chunk_size")
        language = st.selectbox("Output Language", ["en", "th", "zh", "ja", "es", "fr"], key="step2_lang")

    with col2:
        st.write("**Actions**")
        generate_btn = st.button("🤖 Generate with AI", type="primary", use_container_width=True, key="step2_generate_btn")
        st.info(f"Will analyze {len(catalog_df)} products using Gemini 2.5 Flash")

    if generate_btn:
        if not gemini_api_key:
            st.error("⚠️ Please provide a Gemini API key to generate ontology (or upload an existing ontology above).")
        else:
            try:
                import google.generativeai as genai
                import re

                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")

                def extract_json_from_text(text: str) -> dict:
                    text = text.strip()
                    if text.startswith("```"):
                        lines = text.split("\n")
                        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
                    text = re.sub(r"^```json\s*", "", text)
                    text = re.sub(r"```\s*$", "", text)
                    return json.loads(text.strip())

                all_product_texts = catalog_df["product_text"].tolist()

                with st.spinner(f"🤖 Analyzing products in chunks of {chunk_size}..."):
                    chunk_outputs = []
                    progress_bar = st.progress(0)
                    total_chunks = (len(all_product_texts) + int(chunk_size) - 1) // int(chunk_size)

                    for idx, start in enumerate(range(0, len(all_product_texts), int(chunk_size))):
                        chunk = all_product_texts[start:start + int(chunk_size)]
                        examples = "\n".join([f"- {t[:240]}" for t in chunk])

                        prompt = f"""
You are proposing a Lifestyle→Intent ontology for retail marketing.

Input: Product catalog examples (titles + descriptions):
{examples}

Task:
- Propose Lifestyle parents and Intents under each.
- Each Intent must include:
  intent_name (2–5 words), definition (1 sentence),
  include_examples (2–3), exclude_examples (1–2).
- Output language: {language}

Return STRICT minified JSON:
{{
  "lifestyles":[
    {{
      "lifestyle_name":"...",
      "definition":"...",
      "intents":[
        {{
          "intent_name":"...",
          "definition":"...",
          "include_examples":["..."],
          "exclude_examples":["..."]
        }}
      ]
    }}
  ]
}}
""".strip()

                        resp = model.generate_content(prompt)
                        chunk_outputs.append(extract_json_from_text(resp.text))
                        progress_bar.progress((idx + 1) / max(total_chunks, 1))

                    success_box(f"Analyzed {total_chunks} chunks")

                with st.spinner("🔄 Consolidating ontology..."):
                    pool = {}
                    for obj in chunk_outputs:
                        for ls in obj.get("lifestyles", []):
                            ls_name = ls.get("lifestyle_name", "").strip()
                            if not ls_name:
                                continue
                            pool.setdefault(ls_name, {"definition": ls.get("definition", ""), "intents": []})
                            pool[ls_name]["intents"].extend(ls.get("intents", []))

                    pool_text = json.dumps(pool, ensure_ascii=False)[:20000]

                    prompt2 = f"""
You are consolidating multiple ontology proposals into ONE final ontology.

Input pool (may contain duplicates/overlaps):
{pool_text}

Task:
1) Produce EXACTLY {n_lifestyles} Lifestyle parents.
2) Under each, produce up to {max_intents_per_lifestyle} Intents.
3) Remove duplicates, merge similar intents, ensure MECE as much as possible.
4) Create unique IDs:
   - lifestyle_id: LS_...
   - intent_id: IN_...
5) Keep include/exclude examples concise.

Return STRICT minified JSON:
{{
  "lifestyles":[
    {{
      "lifestyle_id":"LS_...",
      "lifestyle_name":"...",
      "definition":"...",
      "intents":[
        {{
          "intent_id":"IN_...",
          "intent_name":"...",
          "definition":"...",
          "include_examples":["..."],
          "exclude_examples":["..."]
        }}
      ]
    }}
  ]
}}
""".strip()

                    final = model.generate_content(prompt2)
                    ontology_data = extract_json_from_text(final.text)

                    dim_lifestyle_rows, dim_intent_rows = [], []
                    for ls in ontology_data.get("lifestyles", []):
                        dim_lifestyle_rows.append({
                            "lifestyle_id": ls.get("lifestyle_id"),
                            "lifestyle_name": ls.get("lifestyle_name"),
                            "definition": ls.get("definition", ""),
                            "version": "v1"
                        })
                        for it in ls.get("intents", []):
                            dim_intent_rows.append({
                                "intent_id": it.get("intent_id", ""),
                                "intent_name": it.get("intent_name", ""),
                                "definition": it.get("definition", ""),
                                "lifestyle_id": ls.get("lifestyle_id"),
                                "include_examples": json.dumps(it.get("include_examples", []), ensure_ascii=False),
                                "exclude_examples": json.dumps(it.get("exclude_examples", []), ensure_ascii=False),
                                "version": "v1"
                            })

                    dim_lifestyle_df = pd.DataFrame(dim_lifestyle_rows).drop_duplicates()
                    dim_intent_df = pd.DataFrame(dim_intent_rows).drop_duplicates()

                    ontology = {
                        "name": "AI-Generated Product Ontology",
                        "version": "v1",
                        "created_at": pd.Timestamp.now().isoformat(),
                        "total_products": len(catalog_df),
                        "model": "gemini-2.5-flash",
                        "language": language,
                        "lifestyles": ontology_data.get("lifestyles", []),
                        "metadata": {
                            "description": "AI-generated ontology from product catalog",
                            "n_lifestyles": len(dim_lifestyle_df),
                            "n_intents": len(dim_intent_df)
                        }
                    }

                    st.session_state["ontology"] = ontology
                    st.session_state["dim_lifestyle_df"] = dim_lifestyle_df
                    st.session_state["dim_intent_df"] = dim_intent_df

                    success_box(f"Generated {len(dim_lifestyle_df)} lifestyles and {len(dim_intent_df)} intents!")

            except ImportError:
                st.error("❌ Missing library: google-generativeai. Please add it to requirements.txt")
            except Exception as e:
                st.error(f"❌ Error generating ontology: {str(e)}")
                st.exception(e)

    # Display & Downloads (works for both generated or uploaded ontology)
    if "ontology" in st.session_state:
        st.divider()
        st.subheader("📥 Ontology Files")

        ontology = st.session_state["ontology"]
        dim_lifestyle_df = st.session_state.get("dim_lifestyle_df", pd.DataFrame())
        dim_intent_df = st.session_state.get("dim_intent_df", pd.DataFrame())

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
    info_box("Upload Product CSV in Step 1 to enable ontology generation or reuse.")

# ---- NEXT: Part 3 continues with Step 3 (Campaign Intent Profiles) ----
# ============================================================================
# STEP 3: CAMPAIGN → WEIGHTED INTENT PROFILE (LLM) + (UPLOAD TO REUSE)
# ============================================================================
step_header("Step 3: Campaign → Weighted Intent Profile (LLM)",
            "Option A: Generate with Gemini • Option B: Upload campaign_intent_profile.csv for reuse (no LLM).")

if "campaigns_df" not in st.session_state:
    info_box("Please load campaign input in Step 0 first.")
    st.stop()

campaigns_df = st.session_state["campaigns_df"].copy()

# -------------------------
# Option B: Upload to reuse
# -------------------------
st.subheader("♻️ Reuse Existing Campaign Intent Profile (Upload)")
with st.expander("Upload campaign_intent_profile.csv to reuse (recommended for demos / save quota)", expanded=True):
    upload_zone("Upload your **campaign_intent_profile.csv** here (no API key needed).")
    uploaded_campaign_profile = st.file_uploader(
        "Upload campaign intent profile CSV (e.g., campaign_intent_profile.csv)",
        type=["csv"],
        key="upload_campaign_intent_profile_csv"
    )

    col_u1, col_u2, col_u3 = st.columns([1, 1, 2])
    with col_u1:
        load_uploaded_campaign_profile_btn = st.button("Load Uploaded Campaign Profile", type="primary", key="load_uploaded_campaign_profile_btn")
    with col_u2:
        clear_campaign_profile_btn = st.button("Clear Campaign Profile (session)", key="clear_campaign_profile_btn")
    with col_u3:
        st.caption("Loads the profile into session_state so Step 4 can proceed without any campaign-intent matching.")

    if clear_campaign_profile_btn:
        for k in ["campaign_intent_profile_df", "campaign_intent_profiles_json"]:
            if k in st.session_state:
                del st.session_state[k]
        success_box("Cleared campaign intent profile from session_state.")

    if load_uploaded_campaign_profile_btn:
        if uploaded_campaign_profile is None:
            st.error("Please upload a campaign_intent_profile.csv first.")
        else:
            try:
                df_up = pd.read_csv(uploaded_campaign_profile)
                df_up.columns = df_up.columns.str.strip().str.lower()

                required_cols = {"campaign_id", "intent_id", "weight"}
                missing = required_cols - set(df_up.columns)
                if missing:
                    st.error(f"❌ Missing columns in uploaded CSV: {sorted(list(missing))}")
                    st.info(f"Found columns: {df_up.columns.tolist()}")
                else:
                    df_up["campaign_id"] = df_up["campaign_id"].astype(str)
                    df_up["intent_id"] = df_up["intent_id"].astype(str)
                    df_up["weight"] = pd.to_numeric(df_up["weight"], errors="coerce").fillna(0.0)

                    if "campaign_name" not in df_up.columns:
                        df_up = df_up.merge(
                            campaigns_df[["campaign_id", "campaign_name"]],
                            on="campaign_id",
                            how="left"
                        )

                    if "rank" not in df_up.columns:
                        df_up = df_up.sort_values(["campaign_id", "weight"], ascending=[True, False]).copy()
                        df_up["rank"] = df_up.groupby("campaign_id").cumcount() + 1

                    def _norm_group(g):
                        s = g["weight"].sum()
                        if s <= 0:
                            g["weight"] = 1.0 / max(len(g), 1)
                        else:
                            g["weight"] = g["weight"] / s
                        return g

                    df_up = df_up.groupby("campaign_id", group_keys=False).apply(_norm_group)

                    st.session_state["campaign_intent_profile_df"] = df_up

                    profiles_json = []
                    for cid, g in df_up.sort_values(["campaign_id", "rank"]).groupby("campaign_id"):
                        cname = ""
                        if "campaign_name" in g.columns and g["campaign_name"].notna().any():
                            cname = str(g["campaign_name"].dropna().iloc[0])
                        top_intents = []
                        for _, r in g.iterrows():
                            top_intents.append({
                                "intent_id": str(r.get("intent_id", "")),
                                "intent_name": str(r.get("intent_name", "")) if "intent_name" in g.columns else "",
                                "weight": float(r.get("weight", 0.0)),
                                "rationale": str(r.get("rationale", "")) if "rationale" in g.columns else ""
                            })
                        profiles_json.append({"campaign_id": str(cid), "campaign_name": cname, "top_intents": top_intents})

                    st.session_state["campaign_intent_profiles_json"] = profiles_json
                    success_box(f"Loaded uploaded campaign intent profile: {len(df_up):,} rows")

                    with st.expander("Preview uploaded profile (first 200 rows)"):
                        st.dataframe(df_up.head(200), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Failed to load uploaded campaign intent profile: {e}")

st.divider()

# -------------------------
# Option A: Generate with LLM (only if ontology exists)
# -------------------------
st.subheader("🤖 Generate Campaign Intent Profiles with AI (Gemini)")

if "ontology" not in st.session_state or "dim_intent_df" not in st.session_state:
    info_box("To generate profiles with AI, generate/upload ontology in Step 2 first. (Upload reuse above still works.)")
else:
    dim_intent_df = st.session_state["dim_intent_df"].copy()

    st.write("**🔑 API Configuration**")
    gemini_api_key = None
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        success_box("Gemini API key loaded from secrets")
    else:
        gemini_api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            help="Get your API key from https://aistudio.google.com/apikey",
            key="gemini_key_step3"
        )
    if not gemini_api_key:
        warn_box("Provide API key only if you want to generate. (Or upload reuse above.)")

    left, right = st.columns([2, 1])
    with left:
        st.write("**Configuration**")
        top_n = st.number_input("Top intents per campaign", min_value=3, max_value=12, value=6, key="step3_top_n")
        output_language = st.selectbox("Output Language", ["en", "th"], index=0, key="step3_output_language")
        include_lifestyle_context = st.checkbox("Include lifestyle context in prompt", value=True, key="step3_include_lifestyle")
    with right:
        st.write("**Rate-limit settings (avoid 429)**")
        model_name = st.text_input("Model name", value="gemini-2.5-flash", key="step3_model_name")
        rpm_limit = st.number_input("Max requests per minute (RPM)", min_value=1, max_value=60, value=8, key="step3_rpm_limit")
        batch_size = st.number_input("Campaigns per request (batch size)", min_value=1, max_value=10, value=3, key="step3_batch_size")
        max_retries = st.number_input("Max retries on 429", min_value=0, max_value=10, value=4, key="step3_max_retries")
        gen_campaign_profile_btn = st.button("🤖 Generate Campaign Intent Profiles", type="primary", use_container_width=True, key="step3_generate_btn")

    intent_candidates = []
    for _, r in dim_intent_df.iterrows():
        intent_candidates.append({
            "intent_id": str(r.get("intent_id", "")).strip(),
            "intent_name": str(r.get("intent_name", "")).strip(),
            "definition": str(r.get("definition", "")).strip(),
            "lifestyle_id": str(r.get("lifestyle_id", "")).strip(),
        })

    def _safe_json_snippet(obj, max_chars=16000):
        txt = json.dumps(obj, ensure_ascii=False)
        return txt[:max_chars]

    intent_snippet = _safe_json_snippet(intent_candidates)

    def normalize_weights(items):
        w = [float(x.get("weight", 0)) for x in items]
        s = sum(w)
        if s <= 0:
            n = len(items)
            for x in items:
                x["weight"] = round(1.0 / max(n, 1), 4)
            return items
        for x in items:
            x["weight"] = round(float(x.get("weight", 0)) / s, 4)
        drift = round(1.0 - sum([x["weight"] for x in items]), 4)
        if items:
            items[0]["weight"] = round(items[0]["weight"] + drift, 4)
        return items

    def extract_json_from_text(text: str) -> dict:
        import re
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        return json.loads(text.strip())

    def parse_retry_delay_seconds(err_text: str) -> int:
        import re
        m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}", err_text)
        return int(m.group(1)) if m else 0

    def chunk_df(df, size: int):
        for start in range(0, len(df), size):
            yield df.iloc[start:start+size]

    if "campaign_intent_profile_df" not in st.session_state:
        st.session_state["campaign_intent_profile_df"] = None
    if "campaign_intent_profiles_json" not in st.session_state:
        st.session_state["campaign_intent_profiles_json"] = None

    if gen_campaign_profile_btn:
        if not gemini_api_key:
            st.error("Please provide a Gemini API key to generate (or use upload reuse above).")
        else:
            try:
                import time, random
                import google.generativeai as genai

                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name)
                min_interval = 60.0 / float(rpm_limit)

                existing_rows = []
                existing_profiles = []
                if st.session_state["campaign_intent_profile_df"] is not None:
                    existing_rows = st.session_state["campaign_intent_profile_df"].to_dict("records")
                if st.session_state["campaign_intent_profiles_json"] is not None:
                    existing_profiles = list(st.session_state["campaign_intent_profiles_json"])

                already_done = set([p["campaign_id"] for p in existing_profiles if "campaign_id" in p])

                def call_llm_with_retry(prompt: str):
                    last_call = st.session_state.get("_last_llm_call_ts_step3", 0.0)
                    now = time.time()
                    wait = (last_call + min_interval) - now
                    if wait > 0:
                        time.sleep(wait)

                    for attempt in range(int(max_retries) + 1):
                        try:
                            resp = model.generate_content(prompt)
                            st.session_state["_last_llm_call_ts_step3"] = time.time()
                            return resp.text
                        except Exception as e:
                            msg = str(e)
                            if "429" in msg or "ResourceExhausted" in msg or "quota" in msg.lower():
                                retry_s = parse_retry_delay_seconds(msg)
                                if retry_s <= 0:
                                    retry_s = int(min(60, (2 ** attempt) * 5))
                                retry_s = retry_s + random.randint(1, 3)
                                warn_box(f"Rate limit hit (429). Sleeping {retry_s}s then retrying... (attempt {attempt+1}/{int(max_retries)+1})")
                                time.sleep(retry_s)
                                continue
                            raise
                    raise RuntimeError("Exceeded retry attempts due to rate limiting.")

                lifestyle_note = ""
                if include_lifestyle_context:
                    ontology = st.session_state.get("ontology", {})
                    ls_list = ontology.get("lifestyles", [])
                    ls_names = [x.get("lifestyle_name", "") for x in ls_list][:12]
                    lifestyle_note = f"\nLifestyle context (high-level): {', '.join([x for x in ls_names if x])}\n"

                results_rows = list(existing_rows)
                profiles_json = list(existing_profiles)

                todo_df = campaigns_df[~campaigns_df["campaign_id"].astype(str).isin(already_done)].reset_index(drop=True)

                if len(todo_df) == 0:
                    info_box("All campaigns already have intent profiles in session_state.")
                else:
                    with st.spinner("🤖 Generating intent profiles (batched + throttled)..."):
                        progress = st.progress(0)
                        total_batches = (len(todo_df) + int(batch_size) - 1) // int(batch_size)

                        for b_idx, batch in enumerate(chunk_df(todo_df, int(batch_size))):
                            batch_campaigns = []
                            for _, r in batch.iterrows():
                                batch_campaigns.append({
                                    "campaign_id": str(r["campaign_id"]),
                                    "campaign_name": str(r["campaign_name"]),
                                    "campaign_brief": str(r["campaign_brief"]),
                                })

                            prompt = f"""
You are a marketing analyst. Map EACH campaign brief into a weighted intent profile.

You MUST choose intents ONLY from this intent list (do not invent new intents):
{intent_snippet}

{lifestyle_note}

Input campaigns (JSON):
{json.dumps(batch_campaigns, ensure_ascii=False)}

Task for EACH campaign:
1) Select the TOP {top_n} intents most relevant to the campaign brief.
2) For each selected intent, provide:
   - intent_id
   - intent_name
   - rationale (1–2 sentences, grounded in the brief)
   - weight (positive number; does NOT need to sum to 1 yet)
3) Output language for rationales: {output_language}

Return STRICT minified JSON only:
{{
  "campaign_profiles":[
    {{
      "campaign_id":"...",
      "campaign_name":"...",
      "top_intents":[
        {{
          "intent_id":"IN_...",
          "intent_name":"...",
          "weight":0.0,
          "rationale":"..."
        }}
      ]
    }}
  ]
}}
""".strip()

                            raw = call_llm_with_retry(prompt)
                            data = extract_json_from_text(raw)
                            campaign_profiles = data.get("campaign_profiles", [])

                            for cp in campaign_profiles:
                                cid = str(cp.get("campaign_id", "")).strip()
                                cname = str(cp.get("campaign_name", "")).strip()
                                top_intents_list = cp.get("top_intents", []) or []
                                top_intents_list = normalize_weights(top_intents_list)

                                for rank, it in enumerate(top_intents_list, start=1):
                                    results_rows.append({
                                        "campaign_id": cid,
                                        "campaign_name": cname,
                                        "rank": rank,
                                        "intent_id": it.get("intent_id", ""),
                                        "intent_name": it.get("intent_name", ""),
                                        "weight": it.get("weight", 0.0),
                                        "rationale": it.get("rationale", ""),
                                    })

                                profiles_json.append({
                                    "campaign_id": cid,
                                    "campaign_name": cname,
                                    "top_intents": top_intents_list
                                })

                            progress.progress((b_idx + 1) / max(total_batches, 1))

                            st.session_state["campaign_intent_profile_df"] = pd.DataFrame(results_rows)
                            st.session_state["campaign_intent_profiles_json"] = profiles_json

                    success_box("Campaign intent profiles generated (weights normalized to sum = 1).")

                st.session_state["campaign_intent_profile_df"] = pd.DataFrame(results_rows)
                st.session_state["campaign_intent_profiles_json"] = profiles_json

            except ImportError:
                st.error("❌ Missing library: google-generativeai. Please add it to requirements.txt")
            except Exception as e:
                st.error(f"❌ Error generating campaign intent profiles: {str(e)}")
                st.exception(e)

# Output preview/download (works for uploaded OR generated)
if st.session_state.get("campaign_intent_profile_df") is not None:
    st.subheader("📌 Campaign Intent Profiles (Preview)")
    df = st.session_state["campaign_intent_profile_df"]
    st.dataframe(df, use_container_width=True, height=420)
    st.download_button(
        "📥 Download campaign_intent_profile.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="campaign_intent_profile.csv",
        mime="text/csv",
        use_container_width=True
    )

if st.session_state.get("campaign_intent_profiles_json") is not None:
    st.subheader("📦 Campaign Intent Profiles (JSON)")
    st.json(st.session_state["campaign_intent_profiles_json"][:50])
    st.caption("Showing first 50 campaigns in JSON preview.")
    st.download_button(
        "📥 Download campaign_intent_profiles.json",
        data=json.dumps(st.session_state["campaign_intent_profiles_json"], ensure_ascii=False, indent=2),
        file_name="campaign_intent_profiles.json",
        mime="application/json",
        use_container_width=True
    )

# ---- NEXT: Part 4 continues with Steps 4–6 ----
# ============================================================================
# STEP 4: PRODUCT → INTENT LABELING (MERGED SINGLE STEP)
# ============================================================================
step_header("Step 4: Product → Intent Labeling (Merged)",
            "Either upload existing product→intent labels CSV, or run LLM labeling. Outputs stored in session_state only.")

if "catalog_df" not in st.session_state:
    info_box("Upload Product CSV in Step 1 to enable Step 4.")
    st.stop()

catalog_df = st.session_state["catalog_df"].copy()

if "product_intent_labels_df" not in st.session_state:
    st.session_state["product_intent_labels_df"] = None
if "product_intent_labels_json" not in st.session_state:
    st.session_state["product_intent_labels_json"] = None

mode = st.radio(
    "Choose Step 4 mode",
    ["Upload product→intent labels CSV", "Run LLM labeling"],
    horizontal=True,
    key="step4_mode"
)

# ----------------------------------------------------------------------------
# MODE A: UPLOAD CSV
# ----------------------------------------------------------------------------
if mode == "Upload product→intent labels CSV":
    st.subheader("📤 Upload product_intent_labels.csv")
    st.caption("Required columns: product_id, intent_id, score (or weight). Recommended: product_name, intent_name, rank.")

    upload_zone("Upload your **product_intent_labels.csv** here (skip LLM).")
    uploaded_labels = st.file_uploader("Upload product_intent_labels CSV", type=["csv"], key="step4_upload_labels")

    if uploaded_labels is not None:
        try:
            labels_df = pd.read_csv(uploaded_labels)
            labels_df.columns = labels_df.columns.str.strip().str.lower()

            required_base = {"product_id", "intent_id"}
            missing_base = required_base - set(labels_df.columns)
            if missing_base:
                st.error(f"❌ Missing required columns: {sorted(list(missing_base))}")
                st.info(f"Found columns: {labels_df.columns.tolist()}")
                st.stop()

            if "score" not in labels_df.columns and "weight" not in labels_df.columns:
                st.error("❌ CSV must contain either 'score' or 'weight' column.")
                st.stop()

            if "score" not in labels_df.columns and "weight" in labels_df.columns:
                labels_df["score"] = pd.to_numeric(labels_df["weight"], errors="coerce").fillna(0.0)

            labels_df["score"] = pd.to_numeric(labels_df["score"], errors="coerce").fillna(0.0)

            if "product_name" not in labels_df.columns:
                if "product_name" in catalog_df.columns:
                    labels_df = labels_df.merge(catalog_df[["product_id", "product_name"]], on="product_id", how="left")
                else:
                    labels_df["product_name"] = ""

            if "intent_name" not in labels_df.columns:
                labels_df["intent_name"] = ""

            if "rank" not in labels_df.columns:
                labels_df = labels_df.sort_values(["product_id", "score"], ascending=[True, False])
                labels_df["rank"] = labels_df.groupby("product_id").cumcount() + 1

            labels_df["product_id"] = labels_df["product_id"].astype(str)
            labels_df["intent_id"] = labels_df["intent_id"].astype(str)
            labels_df["rank"] = pd.to_numeric(labels_df["rank"], errors="coerce").fillna(0).astype(int)

            labels_json = []
            for pid, g in labels_df.sort_values(["product_id", "rank"]).groupby("product_id"):
                pname = str(g["product_name"].iloc[0]) if "product_name" in g.columns else ""
                top_intents = []
                for _, row in g.iterrows():
                    top_intents.append({
                        "intent_id": str(row.get("intent_id", "")),
                        "intent_name": str(row.get("intent_name", "")),
                        "score": float(row.get("score", 0.0)),
                        "evidence": str(row.get("evidence", "")) if "evidence" in labels_df.columns else "",
                        "reason": str(row.get("reason", "")) if "reason" in labels_df.columns else ""
                    })
                labels_json.append({"product_id": str(pid), "product_name": pname, "top_intents": top_intents, "source": "uploaded_csv"})

            st.session_state["product_intent_labels_df"] = labels_df
            st.session_state["product_intent_labels_json"] = labels_json

            success_box(f"Loaded product→intent labels: {len(labels_df):,} rows, {labels_df['product_id'].nunique():,} products")
            with st.expander("Preview uploaded labels"):
                st.dataframe(labels_df.head(50), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error reading labels CSV: {e}")

# ----------------------------------------------------------------------------
# MODE B: RUN LLM LABELING
# ----------------------------------------------------------------------------
else:
    if "dim_intent_df" not in st.session_state or "ontology" not in st.session_state:
        info_box("Generate/upload ontology in Step 2 first (needed for LLM labeling).")
        st.stop()

    dim_intent_df = st.session_state["dim_intent_df"].copy()
    ontology = st.session_state["ontology"]
    ontology_version = str(ontology.get("version", "v1"))

    st.subheader("🔑 API Configuration")
    gemini_api_key = None
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        success_box("Gemini API key loaded from secrets")
    else:
        gemini_api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            help="Get your API key from https://aistudio.google.com/apikey",
            key="gemini_key_step4"
        )

    if not gemini_api_key:
        warn_box("Please provide a Gemini API key to run product labeling.")
        st.stop()

    st.divider()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Labeling settings")
        top_k_intents = st.number_input("Top-K intents per product", min_value=1, max_value=10, value=3, key="step4_topk")
        min_score = st.slider("Minimum score threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05, key="step4_minscore")
        product_text_chars = st.number_input("Max characters of product_text sent to LLM", min_value=120, max_value=2000, value=600, step=60, key="step4_textchars")
    with c2:
        st.subheader("Rate-limit settings (avoid 429)")
        model_name = st.text_input("Model name", value="gemini-2.5-flash", key="step4_model_name")
        rpm_limit = st.number_input("Max requests per minute (RPM)", min_value=1, max_value=60, value=8, key="step4_rpm_limit")
        products_per_request = st.number_input("Products per request (batch size)", min_value=1, max_value=12, value=4, key="step4_products_per_request")
        max_retries = st.number_input("Max retries on 429", min_value=0, max_value=10, value=4, key="step4_max_retries")

    label_btn = st.button("🏷️ Run Product → Intent Labeling", type="primary", use_container_width=True, key="step4_run_btn")

    intent_candidates = []
    for _, r in dim_intent_df.iterrows():
        intent_candidates.append({
            "intent_id": str(r.get("intent_id", "")).strip(),
            "intent_name": str(r.get("intent_name", "")).strip(),
            "definition": str(r.get("definition", "")).strip(),
        })

    def _safe_json_snippet(obj, max_chars=16000):
        txt = json.dumps(obj, ensure_ascii=False)
        return txt[:max_chars]

    intent_snippet = _safe_json_snippet(intent_candidates)

    def extract_json_from_text(text: str) -> dict:
        import re
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        return json.loads(text.strip())

    def parse_retry_delay_seconds(err_text: str) -> int:
        import re
        m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}", err_text)
        return int(m.group(1)) if m else 0

    def chunk_df(df, size: int):
        for start in range(0, len(df), size):
            yield df.iloc[start:start+size]

    def clamp01(x):
        try:
            v = float(x)
        except:
            return 0.0
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    if label_btn:
        try:
            import time, random
            import google.generativeai as genai

            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel(model_name)
            min_interval = 60.0 / float(rpm_limit)

            existing_rows = []
            existing_json = []
            if st.session_state["product_intent_labels_df"] is not None:
                existing_rows = st.session_state["product_intent_labels_df"].to_dict("records")
            if st.session_state["product_intent_labels_json"] is not None:
                existing_json = list(st.session_state["product_intent_labels_json"])

            already_done = set([x.get("product_id") for x in existing_json if x.get("product_id")])
            todo_df = catalog_df[~catalog_df["product_id"].astype(str).isin(already_done)].reset_index(drop=True)

            def call_llm_with_retry(prompt: str):
                last_call = st.session_state.get("_last_llm_call_ts_step4", 0.0)
                now = time.time()
                wait = (last_call + min_interval) - now
                if wait > 0:
                    time.sleep(wait)

                for attempt in range(int(max_retries) + 1):
                    try:
                        resp = model.generate_content(prompt)
                        st.session_state["_last_llm_call_ts_step4"] = time.time()
                        return resp.text
                    except Exception as e:
                        msg = str(e)
                        if "429" in msg or "ResourceExhausted" in msg or "quota" in msg.lower():
                            retry_s = parse_retry_delay_seconds(msg)
                            if retry_s <= 0:
                                retry_s = int(min(60, (2 ** attempt) * 5))
                            retry_s = retry_s + random.randint(1, 3)
                            warn_box(f"Rate limit hit (429). Sleeping {retry_s}s then retrying... (attempt {attempt+1}/{int(max_retries)+1})")
                            time.sleep(retry_s)
                            continue
                        raise
                raise RuntimeError("Exceeded retry attempts due to rate limiting.")

            created_at = pd.Timestamp.now().isoformat()
            labeling_run_id = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

            results_rows = list(existing_rows)
            labels_json = list(existing_json)

            if len(todo_df) == 0:
                info_box("All products already labeled in session_state.")
            else:
                with st.spinner("🏷️ Labeling products (batched + throttled)..."):
                    progress = st.progress(0)
                    total_batches = (len(todo_df) + int(products_per_request) - 1) // int(products_per_request)

                    for b_idx, batch in enumerate(chunk_df(todo_df, int(products_per_request))):
                        batch_products = []
                        for _, r in batch.iterrows():
                            pid = str(r["product_id"])
                            pname = str(r.get("product_name", "") or r.get("product_title", ""))
                            ptext = str(r.get("product_text", ""))[:int(product_text_chars)]
                            batch_products.append({"product_id": pid, "product_name": pname, "product_text": ptext})

                        prompt = f"""
You are labeling retail products into a fixed Intent Ontology for marketing.

You MUST choose intents ONLY from this list (do not invent new intents):
{intent_snippet}

Input products (JSON):
{json.dumps(batch_products, ensure_ascii=False)}

Task for EACH product:
1) Select TOP {int(top_k_intents)} intents most relevant to the product.
2) For each selected intent, output:
   - intent_id
   - intent_name
   - score (0.0 to 1.0, higher = more relevant)
   - evidence (a short phrase copied or paraphrased from the product text)
   - reason (1 short sentence)
3) Return valid JSON only.

Return STRICT minified JSON:
{{
  "labels":[
    {{
      "product_id":"...",
      "product_name":"...",
      "top_intents":[
        {{
          "intent_id":"IN_...",
          "intent_name":"...",
          "score":0.0,
          "evidence":"...",
          "reason":"..."
        }}
      ]
    }}
  ]
}}
""".strip()

                        raw = call_llm_with_retry(prompt)
                        data = extract_json_from_text(raw)
                        batch_labels = data.get("labels", []) or []

                        for item in batch_labels:
                            pid = str(item.get("product_id", "")).strip()
                            pname = str(item.get("product_name", "")).strip()
                            intents = item.get("top_intents", []) or []

                            cleaned = []
                            for it in intents:
                                iid = str(it.get("intent_id", "")).strip()
                                iname = str(it.get("intent_name", "")).strip()
                                score = clamp01(it.get("score", 0.0))
                                evidence = str(it.get("evidence", "")).strip()
                                reason = str(it.get("reason", "")).strip()
                                if iid and iname:
                                    cleaned.append({
                                        "intent_id": iid,
                                        "intent_name": iname,
                                        "score": score,
                                        "evidence": evidence,
                                        "reason": reason
                                    })

                            cleaned = sorted(cleaned, key=lambda x: x["score"], reverse=True)
                            cleaned = [x for x in cleaned if x["score"] >= float(min_score)]
                            cleaned = cleaned[:int(top_k_intents)]

                            labels_json.append({
                                "product_id": pid,
                                "product_name": pname,
                                "top_intents": cleaned,
                                "ontology_version": ontology_version,
                                "labeling_run_id": labeling_run_id,
                                "model": model_name,
                                "created_at": created_at
                            })

                            for rank, it in enumerate(cleaned, start=1):
                                results_rows.append({
                                    "product_id": pid,
                                    "product_name": pname,
                                    "rank": rank,
                                    "intent_id": it["intent_id"],
                                    "intent_name": it["intent_name"],
                                    "score": it["score"],
                                    "evidence": it["evidence"],
                                    "reason": it["reason"],
                                    "ontology_version": ontology_version,
                                    "labeling_run_id": labeling_run_id,
                                    "model": model_name,
                                    "created_at": created_at
                                })

                        st.session_state["product_intent_labels_df"] = pd.DataFrame(results_rows)
                        st.session_state["product_intent_labels_json"] = labels_json
                        progress.progress((b_idx + 1) / max(total_batches, 1))

                success_box("Product → Intent labeling completed (Top-K + scores + evidence).")

            st.session_state["product_intent_labels_df"] = pd.DataFrame(results_rows)
            st.session_state["product_intent_labels_json"] = labels_json

        except ImportError:
            st.error("❌ Missing library: google-generativeai. Please add it to requirements.txt")
        except Exception as e:
            st.error(f"❌ Error labeling products: {str(e)}")
            st.exception(e)

# Display outputs (both modes)
if st.session_state.get("product_intent_labels_df") is not None:
    st.subheader("🏷️ Product Intent Labels (Preview)")
    df = st.session_state["product_intent_labels_df"]
    st.dataframe(df.head(500), use_container_width=True, height=420)
    st.caption(f"Showing up to 500 rows. Total rows: {len(df):,}")
    st.download_button(
        "📥 Download product_intent_labels.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="product_intent_labels.csv",
        mime="text/csv",
        use_container_width=True
    )

if st.session_state.get("product_intent_labels_json") is not None:
    st.subheader("📦 Product Intent Labels (JSON)")
    st.json(st.session_state["product_intent_labels_json"][:50])
    st.caption("Showing first 50 products in JSON preview.")
    st.download_button(
        "📥 Download product_intent_labels.json",
        data=json.dumps(st.session_state["product_intent_labels_json"], ensure_ascii=False, indent=2),
        file_name="product_intent_labels.json",
        mime="application/json",
        use_container_width=True
    )

# ============================================================================
# STEP 5: CUSTOMER INTENT PROFILE BUILDER
# ============================================================================
step_header("Step 5: Customer Intent Profile Builder",
            "Aggregate transactions into customer-level intent weights using Product → Intent labels (CSV upload or Step 4 outputs).")

if "txn_df" not in st.session_state:
    info_box("Upload Transaction CSV in Step 1 to enable Step 5.")
    st.stop()

txn_df = st.session_state["txn_df"].copy()

st.subheader("5.0 Product → Intent Labels Source")
labels_source = st.radio(
    "Choose labels source",
    ["Use Step 4 output (session)", "Upload product_intent_labels.csv"],
    horizontal=True,
    key="step5_labels_source"
)

labels_df = None
if labels_source == "Use Step 4 output (session)":
    if "product_intent_labels_df" in st.session_state and st.session_state["product_intent_labels_df"] is not None:
        labels_df = st.session_state["product_intent_labels_df"].copy()
        success_box(f"Loaded labels from session_state: {len(labels_df):,} rows")
        with st.expander("Preview labels (session)"):
            st.dataframe(labels_df.head(20), use_container_width=True)
    else:
        warn_box("No Step 4 labels found in session_state. Please run Step 4 or upload a CSV.")
else:
    upload_zone("Upload your **product_intent_labels.csv** here for Step 5.")
    uploaded_labels = st.file_uploader("Upload product_intent_labels.csv", type=["csv"], key="step5_labels_csv")
    if uploaded_labels is not None:
        try:
            labels_df = pd.read_csv(uploaded_labels)
            labels_df.columns = labels_df.columns.str.strip().str.lower()
            success_box(f"Uploaded labels CSV: {len(labels_df):,} rows")
            with st.expander("Preview uploaded labels"):
                st.dataframe(labels_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading labels CSV: {e}")

if labels_df is None:
    info_box("Provide Product → Intent labels (Step 4 output or CSV upload) to continue.")
    st.stop()

st.subheader("5.1 Validation & Configuration")

txn_df.columns = txn_df.columns.str.strip().str.lower()
required_txn = {"tx_id", "customer_id", "product_id", "tx_date", "qty", "price"}
missing_txn = required_txn - set(txn_df.columns)
if missing_txn:
    st.error(f"❌ Transactions missing columns: {sorted(list(missing_txn))}")
    st.stop()

txn_df["customer_id"] = txn_df["customer_id"].astype(str)
txn_df["product_id"] = txn_df["product_id"].astype(str)
txn_df["tx_date"] = pd.to_datetime(txn_df["tx_date"], errors="coerce")
txn_df["qty"] = pd.to_numeric(txn_df["qty"], errors="coerce").fillna(0.0)
txn_df["price"] = pd.to_numeric(txn_df["price"], errors="coerce").fillna(0.0)
txn_df["amt"] = txn_df["qty"] * txn_df["price"]

labels_df.columns = labels_df.columns.str.strip().str.lower()
required_labels = {"product_id", "intent_id", "intent_name"}
missing_labels = required_labels - set(labels_df.columns)
if missing_labels:
    st.error(f"❌ Labels missing columns: {sorted(list(missing_labels))}. Required: {sorted(list(required_labels))}")
    st.info(f"Found columns: {labels_df.columns.tolist()}")
    st.stop()

if "score" not in labels_df.columns:
    labels_df["score"] = 1.0

labels_df["product_id"] = labels_df["product_id"].astype(str)
labels_df["intent_id"] = labels_df["intent_id"].astype(str)
labels_df["intent_name"] = labels_df["intent_name"].astype(str)
labels_df["score"] = pd.to_numeric(labels_df["score"], errors="coerce").fillna(0.0)

labels_df = (
    labels_df.sort_values("score", ascending=False)
    .drop_duplicates(subset=["product_id", "intent_id"])
    .reset_index(drop=True)
)

c1, c2, c3 = st.columns(3)
with c1:
    weight_method = st.selectbox("Weighting method for transactions", ["amt (qty*price)", "qty", "count"], index=0, key="step5_weight_method")
with c2:
    min_customer_txns = st.number_input("Min transactions per customer (filter)", min_value=0, max_value=10_000, value=0, key="step5_min_txns")
with c3:
    min_customer_spend = st.number_input("Min spend per customer (filter)", min_value=0.0, max_value=1e12, value=0.0, step=10.0, key="step5_min_spend")

apply_score = st.checkbox("Multiply by intent score (recommended)", value=True, key="step5_apply_score")
normalize_mode = st.selectbox("Normalize customer intent weights", ["sum_to_1", "none"], index=0, key="step5_normalize_mode")
build_btn = st.button("🧩 Build Customer Intent Profiles", type="primary", use_container_width=True)

if build_btn:
    try:
        t = txn_df[["tx_id", "customer_id", "product_id", "tx_date", "qty", "price", "amt"]].copy()
        l = labels_df[["product_id", "intent_id", "intent_name", "score"]].copy()
        joined = t.merge(l, on="product_id", how="inner")
        if joined.empty:
            st.error("❌ No matches between transactions.product_id and labels.product_id. Check your IDs.")
            st.stop()

        if weight_method.startswith("amt"):
            base = joined["amt"]
        elif weight_method == "qty":
            base = joined["qty"]
        else:
            base = 1.0

        joined["base_weight"] = base
        joined["contribution"] = joined["base_weight"] * (joined["score"] if apply_score else 1.0)

        agg = (
            joined.groupby(["customer_id", "intent_id", "intent_name"], as_index=False)
            .agg(
                intent_value=("contribution", "sum"),
                txn_count=("tx_id", "nunique"),
                last_tx_date=("tx_date", "max")
            )
        )

        cust_totals = (
            joined.groupby("customer_id", as_index=False)
            .agg(
                customer_total_value=("contribution", "sum"),
                customer_txn_count=("tx_id", "nunique"),
                customer_last_tx_date=("tx_date", "max")
            )
        )
        agg = agg.merge(cust_totals, on="customer_id", how="left")

        if min_customer_txns > 0:
            agg = agg[agg["customer_txn_count"] >= int(min_customer_txns)]
        if min_customer_spend > 0:
            agg = agg[agg["customer_total_value"] >= float(min_customer_spend)]

        if agg.empty:
            warn_box("After filters, no customers remain. Reduce thresholds.")
            st.stop()

        if normalize_mode == "sum_to_1":
            agg["intent_weight"] = agg["intent_value"] / agg["customer_total_value"].replace({0: pd.NA})
            agg["intent_weight"] = agg["intent_weight"].fillna(0.0)
        else:
            agg["intent_weight"] = agg["intent_value"]

        agg["intent_rank"] = (
            agg.sort_values(["customer_id", "intent_weight"], ascending=[True, False])
            .groupby("customer_id")
            .cumcount() + 1
        )

        customer_intent_profile_df = agg.sort_values(["customer_id", "intent_rank"]).reset_index(drop=True)
        st.session_state["customer_intent_profile_df"] = customer_intent_profile_df
        success_box(f"Built customer_intent_profile_df: {len(customer_intent_profile_df):,} rows")

        customer_lifestyle_profile_df = None
        if "dim_intent_df" in st.session_state and st.session_state["dim_intent_df"] is not None:
            dim_intent_map = st.session_state["dim_intent_df"].copy()
            dim_intent_map.columns = dim_intent_map.columns.str.strip().str.lower()
            if "intent_id" in dim_intent_map.columns and "lifestyle_id" in dim_intent_map.columns:
                map_df = dim_intent_map[["intent_id", "lifestyle_id"]].drop_duplicates()
                tmp = customer_intent_profile_df.merge(map_df, on="intent_id", how="left")

                ls = (
                    tmp.groupby(["customer_id", "lifestyle_id"], as_index=False)
                    .agg(
                        lifestyle_value=("intent_value", "sum"),
                        lifestyle_weight=("intent_weight", "sum"),
                        last_tx_date=("last_tx_date", "max"),
                        txn_count=("txn_count", "sum")
                    )
                )

                ls["lifestyle_rank"] = (
                    ls.sort_values(["customer_id", "lifestyle_weight"], ascending=[True, False])
                    .groupby("customer_id")
                    .cumcount() + 1
                )

                customer_lifestyle_profile_df = ls.sort_values(["customer_id", "lifestyle_rank"]).reset_index(drop=True)
                st.session_state["customer_lifestyle_profile_df"] = customer_lifestyle_profile_df
                success_box(f"Built customer_lifestyle_profile_df: {len(customer_lifestyle_profile_df):,} rows")
            else:
                info_box("dim_intent_df missing lifestyle_id mapping. Skipping customer_lifestyle_profile_df.")
        else:
            info_box("No dim_intent_df found. Skipping customer_lifestyle_profile_df.")

    except Exception as e:
        st.error(f"❌ Error building customer intent profiles: {e}")
        st.exception(e)

if st.session_state.get("customer_intent_profile_df") is not None:
    st.subheader("🧾 Customer Intent Profile (Preview)")
    df = st.session_state["customer_intent_profile_df"]
    st.dataframe(df.head(500), use_container_width=True, height=420)
    st.caption(f"Showing up to 500 rows. Total rows: {len(df):,}")
    st.download_button(
        "📥 Download customer_intent_profile.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="custome_

