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
                
                catalog_df = product_df[["product_id", "product_title", "product_text"]].copy()
                catalog_df = catalog_df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
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
# STEP 2: AI-POWERED ONTOLOGY GENERATION
# ============================================================================
st.header("Step 2: AI-Powered Ontology Generation")

if "catalog_df" in st.session_state:
    catalog_df = st.session_state["catalog_df"]
    
    # --- API KEY CONFIGURATION ---
    st.subheader("🔑 API Configuration")
    
    # Check for API key in secrets or input
    gemini_api_key = None
    if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
        gemini_api_key = st.secrets['GEMINI_API_KEY']
        st.success("✅ Gemini API key loaded from secrets")
    else:
        gemini_api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            help="Get your API key from https://aistudio.google.com/apikey"
        )
        if gemini_api_key:
            st.info("💡 Tip: Add your API key to Streamlit secrets for persistence")
    
    if not gemini_api_key:
        st.warning("⚠️ Please provide a Gemini API key to generate ontology")
        st.info("Get your free API key at: https://aistudio.google.com/apikey")
        st.stop()
    
    st.divider()
    
    # --- ONTOLOGY CONFIGURATION ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Configuration")
        
        st.write("**Ontology Settings**")
        n_lifestyles = st.number_input("Number of Lifestyle Categories", min_value=3, max_value=15, value=6)
        max_intents_per_lifestyle = st.number_input("Max Intents per Lifestyle", min_value=2, max_value=10, value=5)
        chunk_size = st.number_input("Chunk Size (products per API call)", min_value=20, max_value=100, value=40)
        language = st.selectbox("Output Language", ["en", "th", "zh", "ja", "es", "fr"])
    
    with col2:
        st.subheader("Actions")
        
        generate_btn = st.button("🤖 Generate with AI", type="primary", use_container_width=True)
        
        st.info(f"Will analyze {len(catalog_df)} products using Gemini 2.5 Flash")
    
    # --- GENERATE ONTOLOGY WITH AI ---
    if generate_btn:
        try:
            import google.generativeai as genai
            import re
            
            # Configure Gemini
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            
            # Helper function to extract JSON
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
                
                # Step 1: Create chunk-level proposals
                chunk_outputs = []
                progress_bar = st.progress(0)
                total_chunks = (len(all_product_texts) + chunk_size - 1) // chunk_size
                
                for idx, start in enumerate(range(0, len(all_product_texts), chunk_size)):
                    chunk = all_product_texts[start:start+chunk_size]
                    examples = "\n".join([f"- {t[:240]}" for t in chunk])
                    
                    prompt = f"""
You are proposing a Lifestyle→Intent ontology for retail marketing.

Input: Product catalog examples (titles + descriptions):
{examples}

Task:
- Propose Lifestyle parents and Intents under each.
- Each Intent must include:
  intent_id (snake_case), intent_name (2–5 words), definition (1 sentence),
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
                    progress_bar.progress((idx + 1) / total_chunks)
                
                st.success(f"✅ Analyzed {total_chunks} chunks")
            
            with st.spinner("🔄 Consolidating ontology..."):
                
                # Step 2: Merge chunk proposals
                pool = {}
                for obj in chunk_outputs:
                    for ls in obj.get("lifestyles", []):
                        ls_name = ls.get("lifestyle_name", "").strip()
                        if not ls_name:
                            continue
                        pool.setdefault(ls_name, {"definition": ls.get("definition", ""), "intents": []})
                        pool[ls_name]["intents"].extend(ls.get("intents", []))
                
                # Step 3: Ask Gemini to consolidate
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
                
                # Flatten into DataFrames
                dim_lifestyle_rows, dim_intent_rows = [], []
                for ls in ontology_data.get("lifestyles", []):
                    dim_lifestyle_rows.append({
                        "lifestyle_id": ls["lifestyle_id"],
                        "lifestyle_name": ls["lifestyle_name"],
                        "definition": ls.get("definition", ""),
                        "version": "v1"
                    })
                    for it in ls.get("intents", []):
                        dim_intent_rows.append({
                            "intent_id": it["intent_id"],
                            "intent_name": it["intent_name"],
                            "definition": it.get("definition", ""),
                            "lifestyle_id": ls["lifestyle_id"],
                            "include_examples": json.dumps(it.get("include_examples", []), ensure_ascii=False),
                            "exclude_examples": json.dumps(it.get("exclude_examples", []), ensure_ascii=False),
                            "version": "v1"
                        })
                
                dim_lifestyle_df = pd.DataFrame(dim_lifestyle_rows).drop_duplicates()
                dim_intent_df = pd.DataFrame(dim_intent_rows).drop_duplicates()
                
                # Create full ontology object
                ontology = {
                    "name": "AI-Generated Product Ontology",
                    "version": "v1",
                    "created_at": pd.Timestamp.now().isoformat(),
                    "total_products": len(catalog_df),
                    "model": "gemini-2.0-flash-exp",
                    "language": language,
                    "lifestyles": ontology_data.get("lifestyles", []),
                    "metadata": {
                        "description": "AI-generated ontology from product catalog",
                        "n_lifestyles": len(dim_lifestyle_df),
                        "n_intents": len(dim_intent_df)
                    }
                }
                
                # Store in session state
                st.session_state["ontology"] = ontology
                st.session_state["dim_lifestyle_df"] = dim_lifestyle_df
                st.session_state["dim_intent_df"] = dim_intent_df
                
                st.success(f"✅ Generated {len(dim_lifestyle_df)} lifestyles and {len(dim_intent_df)} intents!")
        
        except ImportError:
            st.error("❌ Missing library: google-generativeai. Please add it to requirements.txt")
        except Exception as e:
            st.error(f"❌ Error generating ontology: {str(e)}")
            st.exception(e)
    
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
# STEP 3: CAMPAIGN BRIEF → WEIGHTED INTENT PROFILE (LLM)
# - Select Top intents from the ontology
# - Provide rationale
# - Normalize weights to sum = 1
# - Store in session_state (read-only demo: no DWH writes)
# ============================================================================

st.divider()
st.header("Step 3: Campaign → Weighted Intent Profile (LLM)")
st.caption("Use LLM to map each campaign brief to Top intents from the ontology, with normalized weights + rationale.")

# Preconditions
if "campaigns_df" not in st.session_state:
    st.info("👆 Please load campaign input in Step 0 first.")
    st.stop()

if "ontology" not in st.session_state or "dim_intent_df" not in st.session_state:
    st.info("👆 Please generate the ontology in Step 2 first.")
    st.stop()

campaigns_df = st.session_state["campaigns_df"].copy()
dim_intent_df = st.session_state["dim_intent_df"].copy()

# API key: reuse the key from Step 2 if available
st.subheader("🔑 API Configuration")
gemini_api_key = None
if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
    gemini_api_key = st.secrets['GEMINI_API_KEY']
    st.success("✅ Gemini API key loaded from secrets")
else:
    gemini_api_key = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        help="Get your API key from https://aistudio.google.com/apikey",
        key="gemini_key_step3"
    )

if not gemini_api_key:
    st.warning("⚠️ Please provide a Gemini API key to generate campaign intent profiles.")
    st.stop()

st.divider()

# Controls
left, right = st.columns([2, 1])

with left:
    st.subheader("Configuration")
    top_n = st.number_input("Top intents per campaign", min_value=3, max_value=12, value=6)
    output_language = st.selectbox("Output Language", ["en", "th"], index=0, key="campaign_intent_lang")
    include_lifestyle_context = st.checkbox("Include lifestyle context in prompt", value=True)

with right:
    st.subheader("Actions")
    gen_campaign_profile_btn = st.button("🤖 Generate Campaign Intent Profiles", type="primary", use_container_width=True)
    st.info(f"Will process {len(campaigns_df)} campaign(s)")

# Build intent candidates from ontology
intent_candidates = []
for _, r in dim_intent_df.iterrows():
    intent_candidates.append({
        "intent_id": str(r.get("intent_id", "")).strip(),
        "intent_name": str(r.get("intent_name", "")).strip(),
        "definition": str(r.get("definition", "")).strip(),
        "lifestyle_id": str(r.get("lifestyle_id", "")).strip(),
        "include_examples": r.get("include_examples", "[]"),
        "exclude_examples": r.get("exclude_examples", "[]"),
    })

# Limit prompt size defensively (keep concise but useful)
def _safe_intent_snippet(intent_list, max_chars=18000):
    txt = json.dumps(intent_list, ensure_ascii=False)
    return txt[:max_chars]

intent_snippet = _safe_intent_snippet(intent_candidates)

# Helpers
def normalize_weights(items):
    # items: list of dicts with "weight"
    w = [float(x.get("weight", 0)) for x in items]
    s = sum(w)
    if s <= 0:
        # fallback: equal weights
        n = len(items)
        for x in items:
            x["weight"] = round(1.0 / max(n, 1), 4)
        return items
    for x in items:
        x["weight"] = round(float(x.get("weight", 0)) / s, 4)
    # fix rounding drift to make sum exactly 1.0
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

# Generate profiles
if gen_campaign_profile_btn:
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        results_rows = []
        profiles_json = []

        with st.spinner("🤖 Generating intent profiles for campaigns..."):
            progress = st.progress(0)
            total = len(campaigns_df)

            for i, row in campaigns_df.iterrows():
                campaign_id = str(row["campaign_id"])
                campaign_name = str(row["campaign_name"])
                campaign_brief = str(row["campaign_brief"])

                lifestyle_note = ""
                if include_lifestyle_context:
                    # lightweight lifestyle context from ontology JSON if present
                    ontology = st.session_state.get("ontology", {})
                    ls_list = ontology.get("lifestyles", [])
                    ls_names = [x.get("lifestyle_name", "") for x in ls_list][:15]
                    lifestyle_note = f"\nLifestyle context (high-level): {', '.join([x for x in ls_names if x])}\n"

                prompt = f"""
You are a marketing analyst. Your job is to map ONE campaign brief into a weighted intent profile.

Campaign:
- campaign_id: {campaign_id}
- campaign_name: {campaign_name}
- campaign_brief: {campaign_brief}
{lifestyle_note}

You MUST choose intents ONLY from this intent list (do not invent new intents):
{intent_snippet}

Task:
1) Select the TOP {top_n} intents most relevant to the campaign brief.
2) For each selected intent, provide:
   - intent_id
   - intent_name
   - rationale (1–2 sentences, grounded in the brief)
   - weight (a positive number; does NOT need to sum to 1 yet)
3) Ensure variety: do not select near-duplicates if there are broader alternatives.
4) Output language for rationales: {output_language}

Return STRICT minified JSON only:
{{
  "campaign_id":"{campaign_id}",
  "campaign_name":"{campaign_name}",
  "top_intents":[
    {{
      "intent_id":"IN_...",
      "intent_name":"...",
      "weight":0.0,
      "rationale":"..."
    }}
  ]
}}
""".strip()

                resp = model.generate_content(prompt)
                data = extract_json_from_text(resp.text)

                top_intents = data.get("top_intents", [])
                # Normalize weights to sum=1
                top_intents = normalize_weights(top_intents)

                # Build rows for table
                for rank, it in enumerate(top_intents, start=1):
                    results_rows.append({
                        "campaign_id": campaign_id,
                        "campaign_name": campaign_name,
                        "rank": rank,
                        "intent_id": it.get("intent_id", ""),
                        "intent_name": it.get("intent_name", ""),
                        "weight": it.get("weight", 0.0),
                        "rationale": it.get("rationale", ""),
                    })

                profiles_json.append({
                    "campaign_id": campaign_id,
                    "campaign_name": campaign_name,
                    "top_intents": top_intents
                })

                progress.progress((i + 1) / max(total, 1))

        campaign_intent_profile_df = pd.DataFrame(results_rows)
        st.session_state["campaign_intent_profile_df"] = campaign_intent_profile_df
        st.session_state["campaign_intent_profiles_json"] = profiles_json

        st.success("✅ Campaign intent profiles generated (weights normalized to sum = 1).")

    except ImportError:
        st.error("❌ Missing library: google-generativeai. Please add it to requirements.txt")
    except Exception as e:
        st.error(f"❌ Error generating campaign intent profiles: {str(e)}")
        st.exception(e)

# Display outputs
if "campaign_intent_profile_df" in st.session_state:
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

if "campaign_intent_profiles_json" in st.session_state:
    st.subheader("📦 Campaign Intent Profiles (JSON)")
    st.json(st.session_state["campaign_intent_profiles_json"])
    st.download_button(
        "📥 Download campaign_intent_profiles.json",
        data=json.dumps(st.session_state["campaign_intent_profiles_json"], ensure_ascii=False, indent=2),
        file_name="campaign_intent_profiles.json",
        mime="application/json",
        use_container_width=True
    )



# ============================================================================
# END OF APP
# ============================================================================
