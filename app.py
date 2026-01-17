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

with col1:
    st.subheader("📦 Product Table")
    st.caption("Required columns: product_id, product_title, product_description")

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

    with tab1:
        st.subheader("Product Catalog")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Products", f"{len(catalog_df):,}")
        with c2:
            st.metric("Unique Product IDs", f"{catalog_df['product_id'].nunique():,}")
        with c3:
            st.metric("Columns", len(catalog_df.columns))

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
        with c1:
            st.metric("Total Transactions", f"{len(txn_df):,}")
        with c2:
            st.metric("Unique Customers", f"{txn_df['customer_id'].nunique():,}")
        with c3:
            st.metric("Total Revenue", f"${txn_df['amt'].sum():,.2f}")
        with c4:
            st.metric("Avg Transaction", f"${txn_df['amt'].mean():.2f}")

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
            .reset_index()
        )
        top_products.columns = ["Product ID", "Transaction Count"]
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

    st.subheader("🔑 API Configuration")

    gemini_api_key = None
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini API key loaded from secrets")
    else:
        gemini_api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            help="Get your API key from https://aistudio.google.com/apikey",
            key="gemini_key_step2"
        )
        if gemini_api_key:
            st.info("💡 Tip: Add your API key to Streamlit secrets for persistence")

    if not gemini_api_key:
        st.warning("⚠️ Please provide a Gemini API key to generate ontology")
        st.stop()

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Configuration")
        n_lifestyles = st.number_input("Number of Lifestyle Categories", min_value=3, max_value=15, value=6, key="step2_n_lifestyles")
        max_intents_per_lifestyle = st.number_input("Max Intents per Lifestyle", min_value=2, max_value=10, value=5, key="step2_max_intents")
        chunk_size = st.number_input("Chunk Size (products per API call)", min_value=20, max_value=100, value=40, key="step2_chunk_size")
        language = st.selectbox("Output Language", ["en", "th", "zh", "ja", "es", "fr"], key="step2_lang")

    with col2:
        st.subheader("Actions")
        generate_btn = st.button("🤖 Generate with AI", type="primary", use_container_width=True, key="step2_generate_btn")
        st.info(f"Will analyze {len(catalog_df)} products using Gemini 2.5 Flash")

    if generate_btn:
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

                st.success(f"✅ Analyzed {total_chunks} chunks")

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
                            "intent_id": it.get("intent_id"),
                            "intent_name": it.get("intent_name"),
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

                st.success(f"✅ Generated {len(dim_lifestyle_df)} lifestyles and {len(dim_intent_df)} intents!")

        except ImportError:
            st.error("❌ Missing library: google-generativeai. Please add it to requirements.txt")
        except Exception as e:
            st.error(f"❌ Error generating ontology: {str(e)}")
            st.exception(e)

    if "ontology" in st.session_state:
        st.divider()
        st.subheader("📥 Download Ontology Files")

        ontology = st.session_state["ontology"]
        dim_lifestyle_df = st.session_state["dim_lifestyle_df"]
        dim_intent_df = st.session_state["dim_intent_df"]

        t1, t2, t3 = st.tabs(["📋 Ontology JSON", "🎨 Lifestyle Dimensions", "🎯 Intent Dimensions"])

        with t1:
            st.json(ontology)
            st.download_button(
                label="📥 Download ontology_v1.json",
                data=json.dumps(ontology, ensure_ascii=False, indent=2),
                file_name="ontology_v1.json",
                mime="application/json",
                use_container_width=True
            )

        with t2:
            st.dataframe(dim_lifestyle_df, use_container_width=True)
            st.download_button(
                label="📥 Download dim_lifestyle_v1.csv",
                data=dim_lifestyle_df.to_csv(index=False).encode("utf-8"),
                file_name="dim_lifestyle_v1.csv",
                mime="text/csv",
                use_container_width=True
            )

        with t3:
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
# STEP 3: CAMPAIGN BRIEF → WEIGHTED INTENT PROFILE (LLM) [RATE-LIMIT SAFE]
# ============================================================================
st.divider()
st.header("Step 3: Campaign → Weighted Intent Profile (LLM)")
st.caption("Rate-limit safe: batch + throttle + retry/backoff. No DWH writes (session_state only).")

if "campaigns_df" not in st.session_state:
    st.info("👆 Please load campaign input in Step 0 first.")
    st.stop()
if "ontology" not in st.session_state or "dim_intent_df" not in st.session_state:
    st.info("👆 Please generate the ontology in Step 2 first.")
    st.stop()

campaigns_df = st.session_state["campaigns_df"].copy()
dim_intent_df = st.session_state["dim_intent_df"].copy()

st.subheader("🔑 API Configuration")
gemini_api_key = None
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
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

left, right = st.columns([2, 1])

with left:
    st.subheader("Configuration")
    top_n = st.number_input("Top intents per campaign", min_value=3, max_value=12, value=6, key="step3_top_n")
    output_language = st.selectbox("Output Language", ["en", "th"], index=0, key="step3_lang")
    include_lifestyle_context = st.checkbox("Include lifestyle context in prompt", value=True, key="step3_lifestyle_ctx")

with right:
    st.subheader("Rate-limit settings (to avoid 429)")
    model_name = st.text_input("Model name", value="gemini-2.5-flash", key="step3_model_name")
    rpm_limit = st.number_input("Max requests per minute (RPM)", min_value=1, max_value=60, value=8, key="step3_rpm_limit")
    batch_size = st.number_input("Campaigns per request (batch size)", min_value=1, max_value=10, value=3, key="step3_batch_size")
    max_retries = st.number_input("Max retries on 429", min_value=0, max_value=10, value=4, key="step3_max_retries")
    gen_campaign_profile_btn = st.button("🤖 Generate Campaign Intent Profiles", type="primary", use_container_width=True, key="step3_gen_btn")
    st.info(f"Will process {len(campaigns_df)} campaign(s)")

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
        yield df.iloc[start:start + size]

if "campaign_intent_profile_df" not in st.session_state:
    st.session_state["campaign_intent_profile_df"] = None
if "campaign_intent_profiles_json" not in st.session_state:
    st.session_state["campaign_intent_profiles_json"] = None

if gen_campaign_profile_btn:
    try:
        import time
        import random
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
                        st.warning(f"⚠️ Rate limit hit (429). Sleeping {retry_s}s then retrying... (attempt {attempt+1}/{int(max_retries)+1})")
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
            st.info("✅ All campaigns already have intent profiles in session_state.")
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

            st.success("✅ Campaign intent profiles generated (weights normalized to sum = 1).")

        st.session_state["campaign_intent_profile_df"] = pd.DataFrame(results_rows)
        st.session_state["campaign_intent_profiles_json"] = profiles_json

    except ImportError:
        st.error("❌ Missing library: google-generativeai. Please add it to requirements.txt")
    except Exception as e:
        st.error(f"❌ Error generating campaign intent profiles: {str(e)}")
        st.exception(e)

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
    st.json(st.session_state["campaign_intent_profiles_json"])
    st.download_button(
        "📥 Download campaign_intent_profiles.json",
        data=json.dumps(st.session_state["campaign_intent_profiles_json"], ensure_ascii=False, indent=2),
        file_name="campaign_intent_profiles.json",
        mime="application/json",
        use_container_width=True
    )

# ============================================================================
# STEP 4: PRODUCT → INTENT LABELING (MERGED SINGLE STEP)
# ============================================================================
st.divider()
st.header("Step 4: Product → Intent Labeling (Merged)")
st.caption("Assign Top-K intents + scores + evidence per product using the fixed ontology (rate-limit safe).")

if "catalog_df" not in st.session_state:
    st.info("👆 Upload Product CSV in Step 1 to enable Product → Intent labeling.")
    st.stop()

if "dim_intent_df" not in st.session_state or "ontology" not in st.session_state:
    st.info("👆 Generate the ontology in Step 2 first.")
    st.stop()

catalog_df = st.session_state["catalog_df"].copy()
dim_intent_df = st.session_state["dim_intent_df"].copy()
ontology = st.session_state["ontology"]
ontology_version = str(ontology.get("version", "v1"))

st.subheader("🔑 API Configuration")
gemini_api_key = None
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    st.success("✅ Gemini API key loaded from secrets")
else:
    gemini_api_key = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        help="Get your API key from https://aistudio.google.com/apikey",
        key="gemini_key_step4"
    )
if not gemini_api_key:
    st.warning("⚠️ Please provide a Gemini API key to run product labeling.")
    st.stop()

st.divider()

c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Labeling settings")
    top_k_intents = st.number_input("Top-K intents per product", min_value=1, max_value=10, value=3, key="step4_topk")
    min_score = st.slider("Minimum score threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05, key="step4_min_score")
    product_text_chars = st.number_input("Max characters of product_text sent to LLM", min_value=120, max_value=2000, value=600, step=60, key="step4_text_chars")

with c2:
    st.subheader("Rate-limit settings (avoid 429)")
    model_name = st.text_input("Model name", value="gemini-2.5-flash", key="step4_model_name")
    rpm_limit = st.number_input("Max requests per minute (RPM)", min_value=1, max_value=60, value=8, key="step4_rpm_limit")
    products_per_request = st.number_input("Products per request (batch size)", min_value=1, max_value=12, value=4, key="step4_products_per_request")
    max_retries = st.number_input("Max retries on 429", min_value=0, max_value=10, value=4, key="step4_max_retries")

label_btn = st.button("🏷️ Run Product → Intent Labeling", type="primary", use_container_width=True, key="step4_label_btn")

intent_candidates = []
for _, r in dim_intent_df.iterrows():
    intent_candidates.append({
        "intent_id": str(r.get("intent_id", "")).strip(),
        "intent_name": str(r.get("intent_name", "")).strip(),
        "definition": str(r.get("definition", "")).strip(),
    })

intent_snippet = _safe_json_snippet(intent_candidates)

def clamp01(x):
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v

if "product_intent_labels_df" not in st.session_state:
    st.session_state["product_intent_labels_df"] = None
if "product_intent_labels_json" not in st.session_state:
    st.session_state["product_intent_labels_json"] = None

if label_btn:
    try:
        import time
        import random
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
                        st.warning(f"⚠️ Rate limit hit (429). Sleeping {retry_s}s then retrying... (attempt {attempt+1}/{int(max_retries)+1})")
                        time.sleep(retry_s)
                        continue
                    raise
            raise RuntimeError("Exceeded retry attempts due to rate limiting.")

        created_at = pd.Timestamp.now().isoformat()
        labeling_run_id = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        results_rows = list(existing_rows)
        labels_json = list(existing_json)

        if len(todo_df) == 0:
            st.info("✅ All products already labeled in session_state.")
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
                        batch_products.append({
                            "product_id": pid,
                            "product_name": pname,
                            "product_text": ptext
                        })

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
3) Ensure intents are not near-duplicates if broader intents exist.
4) Return valid JSON only.

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

            st.success("✅ Product → Intent labeling completed (Top-K + scores + evidence).")

        st.session_state["product_intent_labels_df"] = pd.DataFrame(results_rows)
        st.session_state["product_intent_labels_json"] = labels_json

    except ImportError:
        st.error("❌ Missing library: google-generativeai. Please add it to requirements.txt")
    except Exception as e:
        st.error(f"❌ Error labeling products: {str(e)}")
        st.exception(e)

if st.session_state.get("product_intent_labels_df") is not None:
    st.subheader("🏷️ Product Intent Labels (Preview)")
    df = st.session_state["product_intent_labels_df"]
    st.dataframe(df.head(500), use_container_width=True, height=420)
    st.caption(f"Showing up to 500 rows (long format). Total rows: {len(df):,}")
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
# END OF APP
# ============================================================================
