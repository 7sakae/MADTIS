import streamlit as st
import pandas as pd
import json

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="Semantic Audience Studio", page_icon="🧠", layout="wide")

# =============================================================================
# SESSION INIT
# =============================================================================
def init_state():
    defaults = {
        "campaigns_df": None,
        "catalog_df": None,
        "txn_df": None,

        "ontology": None,
        "dim_lifestyle_df": None,
        "dim_intent_df": None,

        "campaign_intent_profile_df": None,
        "campaign_intent_profiles_json": None,

        "product_intent_labels_df": None,
        "product_intent_labels_json": None,

        "customer_intent_profile_df": None,
        "customer_lifestyle_profile_df": None,

        "campaign_audience_ranked_df": None,

        # internal rate-limit markers
        "_last_llm_call_ts_step3": 0.0,
        "_last_llm_call_ts_step4": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_all_state():
    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]
    st.rerun()


init_state()

# =============================================================================
# HELPERS
# =============================================================================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def df_download_button(label: str, df: pd.DataFrame, filename: str, help_txt: str = ""):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        help=help_txt,
        use_container_width=True,
    )


def json_download_button(label: str, obj: dict | list, filename: str):
    st.download_button(
        label=label,
        data=json.dumps(obj, ensure_ascii=False, indent=2),
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )


def badge(ok: bool, text_ok="✅ Ready", text_no="⬜ Not ready"):
    return text_ok if ok else text_no


def step_ready_flags():
    return {
        "step0": st.session_state["campaigns_df"] is not None,
        "step1": (st.session_state["catalog_df"] is not None) and (st.session_state["txn_df"] is not None),
        "step2": st.session_state["ontology"] is not None and st.session_state["dim_intent_df"] is not None,
        "step3": st.session_state["campaign_intent_profile_df"] is not None,
        "step4": st.session_state["product_intent_labels_df"] is not None,
        "step5": st.session_state["customer_intent_profile_df"] is not None,
        "step6": st.session_state["campaign_audience_ranked_df"] is not None,
    }


def top_bar_status():
    flags = step_ready_flags()
    cols = st.columns(7)
    items = [
        ("0 Campaign", flags["step0"]),
        ("1 Data", flags["step1"]),
        ("2 Ontology", flags["step2"]),
        ("3 Camp→Intent", flags["step3"]),
        ("4 Prod→Intent", flags["step4"]),
        ("5 Cust Profile", flags["step5"]),
        ("6 Audience", flags["step6"]),
    ]
    for c, (name, ok) in zip(cols, items):
        with c:
            st.caption(name)
            st.write("✅" if ok else "—")


def safe_money(x):
    try:
        return f"${float(x):,.2f}"
    except:
        return "—"


# =============================================================================
# SIDEBAR NAV
# =============================================================================
with st.sidebar:
    st.title("🧠 Audience Studio")
    st.caption("Prototype wizard • upload → label → profile → rank")

    flags = step_ready_flags()
    st.markdown("### Progress")
    st.write(f"Step 0: {badge(flags['step0'])}")
    st.write(f"Step 1: {badge(flags['step1'])}")
    st.write(f"Step 2: {badge(flags['step2'])}")
    st.write(f"Step 3: {badge(flags['step3'])}")
    st.write(f"Step 4: {badge(flags['step4'])}")
    st.write(f"Step 5: {badge(flags['step5'])}")
    st.write(f"Step 6: {badge(flags['step6'])}")

    st.divider()

    step = st.radio(
        "Go to step",
        [
            "Step 0 — Campaign Input",
            "Step 1 — Upload Data",
            "Data Summary",
            "Step 2 — Ontology",
            "Step 3 — Campaign → Intent",
            "Step 4 — Product → Intent",
            "Step 5 — Customer Profile",
            "Step 6 — Audience Builder",
        ],
        index=0,
        key="sidebar_step_nav"
    )

    st.divider()
    if st.button("🧹 Reset session (clear everything)", use_container_width=True):
        reset_all_state()


# =============================================================================
# HEADER
# =============================================================================
st.title("🧠 Semantic Audience Studio (Prototype)")
st.caption("A guided pipeline: Campaign → Ontology → Labels → Profiles → Ranked Audience")
top_bar_status()
st.divider()
# =============================================================================
# STEP 0: CAMPAIGN INPUT
# =============================================================================
def render_step0():
    st.header("Step 0: Campaign Input")
    st.caption("Upload campaigns CSV or paste JSON. Required: campaign_id, campaign_name, campaign_brief")

    left, right = st.columns([2, 1])

    with right:
        st.markdown("#### Templates")
        st.download_button(
            "Download campaigns template CSV",
            data=("campaign_id,campaign_name,campaign_brief\n"
                  "CAMP_VALENTINES,Valentine's Day,\"Romantic gifting + date-night bundles: fragrance, chocolates, candles, dinner-at-home, self-care sets.\"\n").encode("utf-8"),
            file_name="campaigns_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.info("Tip: JSON for quick demos. CSV for bulk campaigns.")

    with left:
        input_mode = st.radio("Choose input mode", ["Paste JSON", "Upload CSV"], horizontal=True, key="step0_mode")
        campaigns_df = None

        if input_mode == "Paste JSON":
            default_json = [
                {
                    "campaign_id": "CAMP_VALENTINES",
                    "campaign_name": "Valentine's Day",
                    "campaign_brief": "Romantic gifting + date-night bundles: fragrance, chocolates, candles, dinner-at-home, self-care sets."
                }
            ]
            with st.form("step0_json_form", clear_on_submit=False):
                json_text = st.text_area(
                    "Paste campaign JSON (one object or list of objects)",
                    value=json.dumps(default_json, indent=2),
                    height=220
                )
                submit = st.form_submit_button("Load JSON", type="primary", use_container_width=True)

            if submit:
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
            uploaded = st.file_uploader("Upload campaigns CSV", type=["csv"], key="step0_campaign_csv")
            if uploaded is not None:
                try:
                    campaigns_df = pd.read_csv(uploaded)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        required_cols = {"campaign_id", "campaign_name", "campaign_brief"}
        if campaigns_df is not None:
            campaigns_df = normalize_cols(campaigns_df)
            missing = required_cols - set(campaigns_df.columns)

            if missing:
                st.error(f"Missing columns: {sorted(list(missing))}. Required: {sorted(list(required_cols))}")
            else:
                campaigns_df["campaign_id"] = campaigns_df["campaign_id"].astype(str)
                campaigns_df["campaign_name"] = campaigns_df["campaign_name"].astype(str)
                campaigns_df["campaign_brief"] = campaigns_df["campaign_brief"].astype(str)

                st.session_state["campaigns_df"] = campaigns_df
                st.success(f"✅ Loaded {len(campaigns_df)} campaign(s)")
                st.dataframe(campaigns_df, use_container_width=True, height=280)
                df_download_button("📥 Download campaigns.csv", campaigns_df, "campaigns.csv")


# =============================================================================
# STEP 1: UPLOAD PRODUCT + TRANSACTIONS
# =============================================================================
def render_step1():
    st.header("Step 1: Upload Product & Transaction Data")

    left, right = st.columns([2, 1])
    with right:
        st.markdown("#### Templates")
        st.download_button(
            "Download product template CSV",
            data=("product_id,product_title,product_description\n"
                  "P001,Romantic Candle Set,Scented candles perfect for date nights and gifting\n").encode("utf-8"),
            file_name="products_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download transactions template CSV",
            data=("tx_id,customer_id,product_id,tx_date,qty,price\n"
                  "T001,C001,P001,2025-01-10,1,299\n").encode("utf-8"),
            file_name="transactions_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.info("IDs must match: transactions.product_id ↔ products.product_id")

    col1, col2 = left.columns(2)

    with col1:
        st.subheader("📦 Product Table")
        st.caption("Required: product_id, product_title, product_description")

        product_file = st.file_uploader("Upload Product CSV", type=["csv"], key="product_csv")
        if product_file is not None:
            try:
                product_df = pd.read_csv(product_file)
                product_df = normalize_cols(product_df)

                required_product_cols = {"product_id", "product_title", "product_description"}
                missing = required_product_cols - set(product_df.columns)

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

                    with st.expander("Preview products"):
                        st.dataframe(catalog_df.head(30), use_container_width=True)

                    df_download_button("📥 Download catalog_processed.csv", catalog_df, "catalog_processed.csv")

            except Exception as e:
                st.error(f"Error reading product CSV: {e}")

    with col2:
        st.subheader("🛒 Transaction Table")
        st.caption("Required: tx_id, customer_id, product_id, tx_date, qty, price")

        txn_file = st.file_uploader("Upload Transaction CSV", type=["csv"], key="txn_csv")
        if txn_file is not None:
            try:
                txn_df = pd.read_csv(txn_file)
                txn_df = normalize_cols(txn_df)

                required_txn_cols = {"tx_id", "customer_id", "product_id", "tx_date", "qty", "price"}
                missing = required_txn_cols - set(txn_df.columns)

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

                    with st.expander("Preview transactions"):
                        st.dataframe(txn_df.head(30), use_container_width=True)

                    df_download_button("📥 Download transactions_processed.csv", txn_df, "transactions_processed.csv")

            except Exception as e:
                st.error(f"Error reading transaction CSV: {e}")
# =============================================================================
# DATA SUMMARY
# =============================================================================
def render_summary():
    st.header("📊 Data Summary")

    if st.session_state["catalog_df"] is None or st.session_state["txn_df"] is None:
        st.info("Upload both Product + Transaction CSVs in Step 1 first.")
        return

    catalog_df = st.session_state["catalog_df"].copy()
    txn_df = st.session_state["txn_df"].copy()

    tab1, tab2, tab3 = st.tabs(["📦 Products", "🛒 Transactions", "📈 Quick Analytics"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Products", f"{len(catalog_df):,}")
        c2.metric("Unique Product IDs", f"{catalog_df['product_id'].nunique():,}")
        c3.metric("Columns", f"{len(catalog_df.columns)}")
        st.dataframe(catalog_df, use_container_width=True, height=420)

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{len(txn_df):,}")
        c2.metric("Unique Customers", f"{txn_df['customer_id'].nunique():,}")
        c3.metric("Total Revenue", safe_money(txn_df["amt"].sum()))
        c4.metric("Avg Transaction", safe_money(txn_df["amt"].mean()))
        st.dataframe(txn_df, use_container_width=True, height=420)

    with tab3:
        left, right = st.columns(2)

        with left:
            st.write("**Top 10 Customers by Revenue**")
            top_customers = (
                txn_df.groupby("customer_id")["amt"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
                .rename(columns={"customer_id": "Customer ID", "amt": "Total Spent"})
            )
            st.dataframe(top_customers, use_container_width=True)

        with right:
            st.write("**Top 10 Products by Transaction Count**")
            top_products = (
                txn_df.groupby("product_id")
                .size()
                .sort_values(ascending=False)
                .head(10)
                .reset_index(name="Transaction Count")
            )
            top_products = top_products.merge(
                catalog_df[["product_id", "product_name"]],
                on="product_id",
                how="left"
            ).rename(columns={"product_id": "Product ID", "product_name": "Product Name"})
            st.dataframe(top_products[["Product ID", "Product Name", "Transaction Count"]], use_container_width=True)


# =============================================================================
# STEP 2: ONTOLOGY (UPLOAD OR GENERATE)
# =============================================================================
def render_step2():
    st.header("Step 2: Ontology")
    st.caption("Option A: Upload existing ontology JSON • Option B: Generate with Gemini")

    if st.session_state["catalog_df"] is None:
        st.info("Upload Product CSV in Step 1 first.")
        return

    catalog_df = st.session_state["catalog_df"].copy()

    # ---------------------------
    # Upload ontology
    # ---------------------------
    st.subheader("♻️ Reuse Existing Ontology (Upload)")
    with st.expander("Upload ontology_v1.json (recommended for demos)", expanded=True):
        uploaded_ontology_json = st.file_uploader("Upload ontology JSON", type=["json"], key="upload_ontology_json")

        c1, c2, c3 = st.columns([1, 1, 2])
        load_btn = c1.button("Load Uploaded Ontology", type="primary", use_container_width=True)
        clear_btn = c2.button("Clear Ontology (session)", use_container_width=True)
        c3.caption("Loads ontology + rebuilds dim tables. No API key needed.")

        if clear_btn:
            for k in ["ontology", "dim_lifestyle_df", "dim_intent_df"]:
                st.session_state[k] = None
            st.success("✅ Cleared ontology from session.")

        if load_btn:
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

                    st.success(f"✅ Loaded: {len(dim_lifestyle_df)} lifestyles, {len(dim_intent_df)} intents")

                except Exception as e:
                    st.error(f"❌ Failed to load ontology: {e}")

    st.divider()

    # ---------------------------
    # Generate ontology
    # ---------------------------
    st.subheader("🤖 Generate Ontology with AI (Gemini)")

    gemini_api_key = None
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini API key loaded from secrets")
    else:
        gemini_api_key = st.text_input("Gemini API Key (only if generating)", type="password", key="gemini_key_step2")

    left, right = st.columns([2, 1])
    with left:
        n_lifestyles = st.number_input("Number of Lifestyle Categories", 3, 15, 6, key="step2_n_lifestyles")
        max_intents_per_lifestyle = st.number_input("Max Intents per Lifestyle", 2, 10, 5, key="step2_max_intents")
        chunk_size = st.number_input("Chunk size (products per API call)", 20, 100, 40, key="step2_chunk_size")
        language = st.selectbox("Output Language", ["en", "th", "zh", "ja", "es", "fr"], key="step2_lang")
    with right:
        generate_btn = st.button("🤖 Generate with AI", type="primary", use_container_width=True, key="step2_generate_btn")
        st.info(f"Will analyze {len(catalog_df)} products using Gemini 2.5 Flash")

    if generate_btn:
        if not gemini_api_key:
            st.error("Provide Gemini API key, or upload an ontology above.")
            return

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
            chunk_outputs = []
            total_chunks = (len(all_product_texts) + int(chunk_size) - 1) // int(chunk_size)

            with st.status(f"Analyzing products in {total_chunks} chunk(s)...", expanded=True) as status:
                progress = st.progress(0)
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
                    progress.progress((idx + 1) / max(total_chunks, 1))

                status.update(label="Consolidating ontology...", state="running")

                pool = {}
                for obj in chunk_outputs:
                    for ls in obj.get("lifestyles", []):
                        ls_name = (ls.get("lifestyle_name", "") or "").strip()
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

                status.update(label="✅ Ontology ready!", state="complete", expanded=False)

        except ImportError:
            st.error("Missing library: google-generativeai. Add it to requirements.txt")
        except Exception as e:
            st.error(f"❌ Error generating ontology: {e}")
            st.exception(e)

    # Display outputs if available
    if st.session_state["ontology"] is not None:
        st.divider()
        st.subheader("📥 Ontology Outputs")
        ontology = st.session_state["ontology"]
        dim_lifestyle_df = st.session_state.get("dim_lifestyle_df") or pd.DataFrame()
        dim_intent_df = st.session_state.get("dim_intent_df") or pd.DataFrame()

        t1, t2, t3 = st.tabs(["📋 Ontology JSON", "🎨 Lifestyle Dim", "🎯 Intent Dim"])
        with t1:
            st.json(ontology)
            json_download_button("📥 Download ontology_v1.json", ontology, "ontology_v1.json")
        with t2:
            st.dataframe(dim_lifestyle_df, use_container_width=True, height=420)
            df_download_button("📥 Download dim_lifestyle_v1.csv", dim_lifestyle_df, "dim_lifestyle_v1.csv")
        with t3:
            st.dataframe(dim_intent_df, use_container_width=True, height=420)
            df_download_button("📥 Download dim_intent_v1.csv", dim_intent_df, "dim_intent_v1.csv")
# =============================================================================
# STEP 3: CAMPAIGN → WEIGHTED INTENT PROFILE (UPLOAD OR GENERATE)
# =============================================================================
def render_step3():
    st.header("Step 3: Campaign → Weighted Intent Profile")
    st.caption("Option A: Upload campaign_intent_profile.csv • Option B: Generate with Gemini")

    if st.session_state["campaigns_df"] is None:
        st.info("Load campaigns in Step 0 first.")
        return

    campaigns_df = st.session_state["campaigns_df"].copy()
    campaigns_df = normalize_cols(campaigns_df)

    # -------------------------
    # Option B: Upload to reuse
    # -------------------------
    st.subheader("♻️ Reuse Existing Campaign Intent Profile (Upload)")

    with st.expander("Upload campaign_intent_profile.csv to reuse (recommended for demos / save quota)", expanded=True):
        uploaded_campaign_profile = st.file_uploader(
            "Upload campaign_intent_profile.csv",
            type=["csv"],
            key="upload_campaign_intent_profile_csv"
        )

        c1, c2, c3 = st.columns([1, 1, 2])
        load_btn = c1.button("Load Uploaded Campaign Profile", type="primary", use_container_width=True)
        clear_btn = c2.button("Clear Campaign Profile (session)", use_container_width=True)
        c3.caption("Loads profile into session_state so Step 6 can run without LLM.")

        if clear_btn:
            st.session_state["campaign_intent_profile_df"] = None
            st.session_state["campaign_intent_profiles_json"] = None
            st.success("✅ Cleared campaign intent profile from session.")

        if load_btn:
            if uploaded_campaign_profile is None:
                st.error("Please upload a campaign_intent_profile.csv first.")
            else:
                try:
                    df_up = pd.read_csv(uploaded_campaign_profile)
                    df_up = normalize_cols(df_up)

                    required_cols = {"campaign_id", "intent_id", "weight"}
                    missing = required_cols - set(df_up.columns)
                    if missing:
                        st.error(f"❌ Missing columns in uploaded CSV: {sorted(list(missing))}")
                        st.info(f"Found columns: {df_up.columns.tolist()}")
                    else:
                        # Normalize types
                        df_up["campaign_id"] = df_up["campaign_id"].astype(str)
                        df_up["intent_id"] = df_up["intent_id"].astype(str)
                        df_up["weight"] = pd.to_numeric(df_up["weight"], errors="coerce").fillna(0.0)

                        # Optional enrich
                        if "campaign_name" not in df_up.columns:
                            df_up = df_up.merge(
                                campaigns_df[["campaign_id", "campaign_name"]],
                                on="campaign_id",
                                how="left"
                            )

                        if "rank" not in df_up.columns:
                            df_up = df_up.sort_values(["campaign_id", "weight"], ascending=[True, False]).copy()
                            df_up["rank"] = df_up.groupby("campaign_id").cumcount() + 1

                        # Normalize weights per campaign
                        def _norm_group(g):
                            s = g["weight"].sum()
                            if s <= 0:
                                g["weight"] = 1.0 / max(len(g), 1)
                            else:
                                g["weight"] = g["weight"] / s
                            return g

                        df_up = df_up.groupby("campaign_id", group_keys=False).apply(_norm_group)

                        # Store
                        st.session_state["campaign_intent_profile_df"] = df_up

                        # Build JSON view
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
                            profiles_json.append({
                                "campaign_id": str(cid),
                                "campaign_name": cname,
                                "top_intents": top_intents
                            })

                        st.session_state["campaign_intent_profiles_json"] = profiles_json

                        st.success(f"✅ Loaded uploaded campaign intent profile: {len(df_up):,} rows")

                        with st.expander("Preview uploaded profile"):
                            st.dataframe(df_up.head(200), use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Failed to load uploaded campaign intent profile: {e}")

    st.divider()

    # -------------------------
    # Option A: Generate with Gemini
    # -------------------------
    st.subheader("🤖 Generate Campaign Intent Profiles with AI (Gemini)")

    if st.session_state["dim_intent_df"] is None or st.session_state["ontology"] is None:
        st.info("To generate with AI, upload/generate ontology in Step 2 first. (Upload reuse above still works.)")
        return

    dim_intent_df = st.session_state["dim_intent_df"].copy()
    dim_intent_df = normalize_cols(dim_intent_df)

    # API key
    st.write("**🔑 API Configuration**")
    gemini_api_key = None
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini API key loaded from secrets")
    else:
        gemini_api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            key="gemini_key_step3"
        )

    if not gemini_api_key:
        st.warning("Provide API key only if you want to generate. Otherwise upload CSV above.")
        return

    # Controls
    left, right = st.columns([2, 1])

    with left:
        st.write("**Configuration**")
        top_n = st.number_input("Top intents per campaign", 3, 12, 6, key="step3_top_n")
        output_language = st.selectbox("Rationale language", ["en", "th"], index=0, key="step3_output_language")
        include_lifestyle_context = st.checkbox("Include lifestyle context", value=True, key="step3_include_lifestyle")

    with right:
        st.write("**Rate-limit (avoid 429)**")
        model_name = st.text_input("Model name", value="gemini-2.5-flash", key="step3_model_name")
        rpm_limit = st.number_input("Max requests/min", 1, 60, 8, key="step3_rpm_limit")
        batch_size = st.number_input("Campaigns per request", 1, 10, 3, key="step3_batch_size")
        max_retries = st.number_input("Max retries on 429", 0, 10, 4, key="step3_max_retries")
        gen_btn = st.button("🤖 Generate Campaign Intent Profiles", type="primary", use_container_width=True)

    # Build concise intent candidate list
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

    if gen_btn:
        try:
            import time, random
            import google.generativeai as genai

            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel(model_name)

            min_interval = 60.0 / float(rpm_limit)

            # Continue-runs support
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
                            st.warning(f"⚠️ Rate limit hit. Sleeping {retry_s}s then retry... (attempt {attempt+1}/{int(max_retries)+1})")
                            time.sleep(retry_s)
                            continue
                        raise
                raise RuntimeError("Exceeded retry attempts due to rate limiting.")

            lifestyle_note = ""
            if include_lifestyle_context:
                ontology = st.session_state.get("ontology") or {}
                ls_list = ontology.get("lifestyles", [])
                ls_names = [x.get("lifestyle_name", "") for x in ls_list][:12]
                lifestyle_note = f"\nLifestyle context: {', '.join([x for x in ls_names if x])}\n"

            results_rows = list(existing_rows)
            profiles_json = list(existing_profiles)

            todo_df = campaigns_df[~campaigns_df["campaign_id"].astype(str).isin(already_done)].reset_index(drop=True)

            if len(todo_df) == 0:
                st.info("✅ All campaigns already have profiles in session.")
            else:
                with st.status("Generating intent profiles (batched + throttled)...", expanded=True) as status:
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

You MUST choose intents ONLY from this list (do not invent new intents):
{intent_snippet}

{lifestyle_note}

Input campaigns (JSON):
{json.dumps(batch_campaigns, ensure_ascii=False)}

Task for EACH campaign:
1) Select the TOP {int(top_n)} intents most relevant to the campaign brief.
2) For each selected intent, provide:
   - intent_id
   - intent_name
   - rationale (1–2 sentences)
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
                        cps = data.get("campaign_profiles", []) or []

                        for cp in cps:
                            cid = str(cp.get("campaign_id", "")).strip()
                            cname = str(cp.get("campaign_name", "")).strip()
                            top_intents_list = normalize_weights(cp.get("top_intents", []) or [])

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
                    status.update(label="✅ Campaign profiles ready!", state="complete", expanded=False)

        except ImportError:
            st.error("Missing library: google-generativeai. Add it to requirements.txt")
        except Exception as e:
            st.error(f"❌ Error generating campaign intent profiles: {e}")
            st.exception(e)

    # -------------------------
    # OUTPUT PREVIEW/DOWNLOAD
    # -------------------------
    if st.session_state["campaign_intent_profile_df"] is not None:
        st.subheader("📌 Campaign Intent Profiles (Preview)")
        df = st.session_state["campaign_intent_profile_df"]
        st.dataframe(df, use_container_width=True, height=420)
        df_download_button("📥 Download campaign_intent_profile.csv", df, "campaign_intent_profile.csv")

    if st.session_state["campaign_intent_profiles_json"] is not None:
        st.subheader("📦 Campaign Intent Profiles (JSON)")
        st.json(st.session_state["campaign_intent_profiles_json"][:50])
        st.caption("Showing first 50 campaigns.")
        json_download_button(
            "📥 Download campaign_intent_profiles.json",
            st.session_state["campaign_intent_profiles_json"],
            "campaign_intent_profiles.json"
        )
# =============================================================================
# STEP 4: PRODUCT → INTENT LABELING (UPLOAD OR LLM)
# =============================================================================
def render_step4():
    st.header("Step 4: Product → Intent Labeling")
    st.caption("Upload product_intent_labels.csv or run LLM labeling. Outputs are stored in session_state only.")

    if st.session_state["catalog_df"] is None:
        st.info("Upload Product CSV in Step 1 first.")
        return

    catalog_df = st.session_state["catalog_df"].copy()
    catalog_df = normalize_cols(catalog_df)

    mode = st.radio(
        "Choose mode",
        ["Upload product→intent labels CSV", "Run LLM labeling"],
        horizontal=True,
        key="step4_mode"
    )

    # -------------------------------------------------------------
    # MODE A: Upload CSV
    # -------------------------------------------------------------
    if mode == "Upload product→intent labels CSV":
        st.subheader("📤 Upload product_intent_labels.csv")
        st.caption("Required: product_id, intent_id, and score (or weight). Recommended: product_name, intent_name, rank.")

        uploaded_labels = st.file_uploader("Upload labels CSV", type=["csv"], key="step4_upload_labels")

        if uploaded_labels is not None:
            try:
                labels_df = pd.read_csv(uploaded_labels)
                labels_df = normalize_cols(labels_df)

                required_base = {"product_id", "intent_id"}
                missing_base = required_base - set(labels_df.columns)
                if missing_base:
                    st.error(f"❌ Missing required columns: {sorted(list(missing_base))}")
                    st.info(f"Found columns: {labels_df.columns.tolist()}")
                    return

                # score/weight handling
                if "score" not in labels_df.columns and "weight" not in labels_df.columns:
                    st.error("❌ CSV must contain either 'score' or 'weight' column.")
                    st.info("Tip: If your file uses 'weight', we will map it to 'score'.")
                    return

                if "score" not in labels_df.columns and "weight" in labels_df.columns:
                    labels_df["score"] = pd.to_numeric(labels_df["weight"], errors="coerce").fillna(0.0)

                labels_df["score"] = pd.to_numeric(labels_df["score"], errors="coerce").fillna(0.0)

                # Optional enrich
                labels_df["product_id"] = labels_df["product_id"].astype(str)
                labels_df["intent_id"] = labels_df["intent_id"].astype(str)

                if "product_name" not in labels_df.columns:
                    if "product_name" in catalog_df.columns:
                        labels_df = labels_df.merge(
                            catalog_df[["product_id", "product_name"]],
                            on="product_id",
                            how="left"
                        )
                    else:
                        labels_df["product_name"] = ""

                if "intent_name" not in labels_df.columns:
                    labels_df["intent_name"] = ""

                if "rank" not in labels_df.columns:
                    labels_df = labels_df.sort_values(["product_id", "score"], ascending=[True, False])
                    labels_df["rank"] = labels_df.groupby("product_id").cumcount() + 1

                labels_df["rank"] = pd.to_numeric(labels_df["rank"], errors="coerce").fillna(0).astype(int)

                # Build JSON view
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
                    labels_json.append({
                        "product_id": str(pid),
                        "product_name": pname,
                        "top_intents": top_intents,
                        "source": "uploaded_csv"
                    })

                st.session_state["product_intent_labels_df"] = labels_df
                st.session_state["product_intent_labels_json"] = labels_json

                st.success(f"✅ Loaded labels: {len(labels_df):,} rows, {labels_df['product_id'].nunique():,} products")
                with st.expander("Preview labels"):
                    st.dataframe(labels_df.head(80), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error reading labels CSV: {e}")

    # -------------------------------------------------------------
    # MODE B: Run LLM labeling
    # -------------------------------------------------------------
    else:
        if st.session_state["dim_intent_df"] is None or st.session_state["ontology"] is None:
            st.info("Generate/upload ontology in Step 2 first (needed for LLM labeling).")
            return

        dim_intent_df = normalize_cols(st.session_state["dim_intent_df"].copy())
        ontology = st.session_state["ontology"]
        ontology_version = str(ontology.get("version", "v1"))

        st.subheader("🔑 API Configuration")
        gemini_api_key = None
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            gemini_api_key = st.secrets["GEMINI_API_KEY"]
            st.success("✅ Gemini API key loaded from secrets")
        else:
            gemini_api_key = st.text_input("Enter your Gemini API Key", type="password", key="gemini_key_step4")

        if not gemini_api_key:
            st.warning("Please provide Gemini API key to run labeling (or upload CSV).")
            return

        st.divider()

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Labeling settings")
            top_k_intents = st.number_input("Top-K intents per product", 1, 10, 3, key="step4_topk")
            min_score = st.slider("Minimum score threshold", 0.0, 1.0, 0.25, 0.05, key="step4_minscore")
            product_text_chars = st.number_input("Max product_text chars sent to LLM", 120, 2000, 600, 60, key="step4_textchars")
        with c2:
            st.subheader("Rate-limit (avoid 429)")
            model_name = st.text_input("Model name", value="gemini-2.5-flash", key="step4_model_name")
            rpm_limit = st.number_input("Max requests/min", 1, 60, 8, key="step4_rpm_limit")
            products_per_request = st.number_input("Products/request", 1, 12, 4, key="step4_products_per_request")
            max_retries = st.number_input("Max retries on 429", 0, 10, 4, key="step4_max_retries")

        label_btn = st.button("🏷️ Run Product → Intent Labeling", type="primary", use_container_width=True, key="step4_run_btn")

        # Prepare concise intent list
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
                yield df.iloc[start:start + size]

        def clamp01(x):
            try:
                v = float(x)
            except:
                return 0.0
            return 0.0 if v < 0 else (1.0 if v > 1 else v)

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
                                st.warning(f"⚠️ Rate limit hit. Sleeping {retry_s}s then retry... (attempt {attempt+1}/{int(max_retries)+1})")
                                time.sleep(retry_s)
                                continue
                            raise
                    raise RuntimeError("Exceeded retry attempts due to rate limiting.")

                created_at = pd.Timestamp.now().isoformat()
                labeling_run_id = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

                results_rows = list(existing_rows)
                labels_json = list(existing_json)

                if len(todo_df) == 0:
                    st.info("✅ All products already labeled in session.")
                else:
                    with st.status("Labeling products (batched + throttled)...", expanded=True) as status:
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
   - score (0.0 to 1.0)
   - evidence (short phrase from product text)
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

                        status.update(label="✅ Product labels ready!", state="complete", expanded=False)

            except ImportError:
                st.error("Missing library: google-generativeai. Add it to requirements.txt")
            except Exception as e:
                st.error(f"❌ Error labeling products: {e}")
                st.exception(e)

    # Outputs preview
    if st.session_state["product_intent_labels_df"] is not None:
        st.subheader("🏷️ Product Intent Labels (Preview)")
        df = st.session_state["product_intent_labels_df"]
        st.dataframe(df.head(500), use_container_width=True, height=420)
        st.caption(f"Showing up to 500 rows. Total rows: {len(df):,}")
        df_download_button("📥 Download product_intent_labels.csv", df, "product_intent_labels.csv")

    if st.session_state["product_intent_labels_json"] is not None:
        st.subheader("📦 Product Intent Labels (JSON)")
        st.json(st.session_state["product_intent_labels_json"][:30])
        st.caption("Showing first 30 products.")
        json_download_button("📥 Download product_intent_labels.json", st.session_state["product_intent_labels_json"], "product_intent_labels.json")


# =============================================================================
# STEP 5: CUSTOMER INTENT PROFILE BUILDER
# =============================================================================
def render_step5():
    st.header("Step 5: Customer Intent Profile Builder")
    st.caption("Aggregate transactions into customer-level intent weights using product→intent labels.")

    if st.session_state["txn_df"] is None:
        st.info("Upload Transaction CSV in Step 1 first.")
        return

    txn_df = normalize_cols(st.session_state["txn_df"].copy())

    st.subheader("5.0 Product → Intent Labels Source")
    labels_source = st.radio(
        "Choose labels source",
        ["Use Step 4 output (session)", "Upload product_intent_labels.csv"],
        horizontal=True,
        key="step5_labels_source"
    )

    labels_df = None

    if labels_source == "Use Step 4 output (session)":
        if st.session_state["product_intent_labels_df"] is not None:
            labels_df = normalize_cols(st.session_state["product_intent_labels_df"].copy())
            st.success(f"✅ Loaded labels from session: {len(labels_df):,} rows")
            with st.expander("Preview labels"):
                st.dataframe(labels_df.head(25), use_container_width=True)
        else:
            st.warning("No Step 4 labels found. Run Step 4 or upload CSV.")
            return
    else:
        uploaded_labels = st.file_uploader("Upload product_intent_labels.csv", type=["csv"], key="step5_labels_csv")
        if uploaded_labels is not None:
            try:
                labels_df = pd.read_csv(uploaded_labels)
                labels_df = normalize_cols(labels_df)
                st.success(f"✅ Uploaded labels CSV: {len(labels_df):,} rows")
                with st.expander("Preview uploaded labels"):
                    st.dataframe(labels_df.head(25), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading labels CSV: {e}")
                return
        else:
            st.info("Upload labels CSV to continue.")
            return

    # Validate txn schema
    required_txn = {"tx_id", "customer_id", "product_id", "tx_date", "qty", "price"}
    missing_txn = required_txn - set(txn_df.columns)
    if missing_txn:
        st.error(f"❌ Transactions missing columns: {sorted(list(missing_txn))}")
        return

    txn_df["customer_id"] = txn_df["customer_id"].astype(str)
    txn_df["product_id"] = txn_df["product_id"].astype(str)
    txn_df["tx_date"] = pd.to_datetime(txn_df["tx_date"], errors="coerce")
    txn_df["qty"] = pd.to_numeric(txn_df["qty"], errors="coerce").fillna(0.0)
    txn_df["price"] = pd.to_numeric(txn_df["price"], errors="coerce").fillna(0.0)
    txn_df["amt"] = txn_df["qty"] * txn_df["price"]

    # Validate labels schema
    required_labels = {"product_id", "intent_id", "intent_name"}
    missing_labels = required_labels - set(labels_df.columns)
    if missing_labels:
        st.error(f"❌ Labels missing columns: {sorted(list(missing_labels))}. Required: {sorted(list(required_labels))}")
        st.info(f"Found columns: {labels_df.columns.tolist()}")
        return

    if "score" not in labels_df.columns:
        labels_df["score"] = 1.0

    labels_df["product_id"] = labels_df["product_id"].astype(str)
    labels_df["intent_id"] = labels_df["intent_id"].astype(str)
    labels_df["intent_name"] = labels_df["intent_name"].astype(str)
    labels_df["score"] = pd.to_numeric(labels_df["score"], errors="coerce").fillna(0.0)

    # Keep best score per (product_id, intent_id)
    labels_df = (
        labels_df.sort_values("score", ascending=False)
        .drop_duplicates(subset=["product_id", "intent_id"])
        .reset_index(drop=True)
    )

    st.subheader("5.1 Configuration")

    c1, c2, c3 = st.columns(3)
    with c1:
        weight_method = st.selectbox("Weighting method", ["amt (qty*price)", "qty", "count"], index=0, key="step5_weight_method")
    with c2:
        min_customer_txns = st.number_input("Min transactions per customer", 0, 10_000, 0, key="step5_min_txns")
    with c3:
        min_customer_spend = st.number_input("Min spend per customer", 0.0, 1e12, 0.0, 10.0, key="step5_min_spend")

    apply_score = st.checkbox("Multiply by intent score", value=True, key="step5_apply_score")
    normalize_mode = st.selectbox("Normalize weights", ["sum_to_1", "none"], index=0, key="step5_normalize_mode")

    build_btn = st.button("🧩 Build Customer Intent Profiles", type="primary", use_container_width=True)

    if build_btn:
        try:
            t = txn_df[["tx_id", "customer_id", "product_id", "tx_date", "qty", "price", "amt"]].copy()
            l = labels_df[["product_id", "intent_id", "intent_name", "score"]].copy()

            joined = t.merge(l, on="product_id", how="inner")
            if joined.empty:
                st.error("❌ No matches between transactions.product_id and labels.product_id. Check IDs.")
                return

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

            # Filters
            if min_customer_txns > 0:
                agg = agg[agg["customer_txn_count"] >= int(min_customer_txns)]
            if min_customer_spend > 0:
                agg = agg[agg["customer_total_value"] >= float(min_customer_spend)]

            if agg.empty:
                st.warning("⚠️ After filters, no customers remain. Reduce thresholds.")
                return

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

            st.success(f"✅ Built customer_intent_profile_df: {len(customer_intent_profile_df):,} rows")

            # Optional: lifestyle aggregation if mapping exists
            customer_lifestyle_profile_df = None
            if st.session_state["dim_intent_df"] is not None:
                map_df = normalize_cols(st.session_state["dim_intent_df"].copy())
                if "intent_id" in map_df.columns and "lifestyle_id" in map_df.columns:
                    tmp = customer_intent_profile_df.merge(
                        map_df[["intent_id", "lifestyle_id"]].drop_duplicates(),
                        on="intent_id",
                        how="left"
                    )

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
                    st.success(f"✅ Built customer_lifestyle_profile_df: {len(customer_lifestyle_profile_df):,} rows")
                else:
                    st.info("dim_intent_df missing lifestyle_id mapping. Skipping lifestyle profile.")

        except Exception as e:
            st.error(f"❌ Error building profiles: {e}")
            st.exception(e)

    # Output previews
    if st.session_state["customer_intent_profile_df"] is not None:
        st.subheader("🧾 Customer Intent Profile (Preview)")
        df = st.session_state["customer_intent_profile_df"]
        st.dataframe(df.head(500), use_container_width=True, height=420)
        st.caption(f"Showing up to 500 rows. Total rows: {len(df):,}")
        df_download_button("📥 Download customer_intent_profile.csv", df, "customer_intent_profile.csv")

    if st.session_state["customer_lifestyle_profile_df"] is not None:
        st.subheader("🧾 Customer Lifestyle Profile (Preview)")
        lsdf = st.session_state["customer_lifestyle_profile_df"]
        st.dataframe(lsdf.head(500), use_container_width=True, height=420)
        st.caption(f"Showing up to 500 rows. Total rows: {len(lsdf):,}")
        df_download_button("📥 Download customer_lifestyle_profile.csv", lsdf, "customer_lifestyle_profile.csv")
# =============================================================================
# STEP 6: CAMPAIGN AUDIENCE BUILDER (RANKING + EXPLAINABILITY)
# =============================================================================
def render_step6():
    st.header("Step 6: Campaign Audience Builder")
    st.caption("Rank customers per campaign by matching Campaign Intent Weights × Customer Intent Weights (with explainability).")

    if st.session_state["customer_intent_profile_df"] is None:
        st.info("Build Customer Intent Profiles in Step 5 first.")
        return

    customer_intent_profile_df = normalize_cols(st.session_state["customer_intent_profile_df"].copy())

    required_cust_cols = {"customer_id", "intent_id", "intent_name", "intent_weight"}
    missing = required_cust_cols - set(customer_intent_profile_df.columns)
    if missing:
        st.error(f"customer_intent_profile_df missing columns: {sorted(list(missing))}")
        return

    st.subheader("6.0 Campaign Intent Profile Source")
    camp_source = st.radio(
        "Choose campaign profile source",
        ["Use Step 3 output (session)", "Upload campaign_intent_profile.csv"],
        horizontal=True,
        key="step6_campaign_source"
    )

    campaign_intent_profile_df = None
    if camp_source == "Use Step 3 output (session)":
        if st.session_state["campaign_intent_profile_df"] is not None:
            campaign_intent_profile_df = normalize_cols(st.session_state["campaign_intent_profile_df"].copy())
            st.success(f"✅ Loaded campaign intent profiles from session: {len(campaign_intent_profile_df):,} rows")
            with st.expander("Preview campaign profiles"):
                st.dataframe(campaign_intent_profile_df.head(25), use_container_width=True)
        else:
            st.warning("No Step 3 output found. Upload CSV instead.")
            return
    else:
        uploaded = st.file_uploader("Upload campaign_intent_profile.csv", type=["csv"], key="step6_campaign_intent_csv")
        if uploaded is not None:
            try:
                campaign_intent_profile_df = pd.read_csv(uploaded)
                campaign_intent_profile_df = normalize_cols(campaign_intent_profile_df)
                st.success(f"✅ Uploaded campaign intent profile: {len(campaign_intent_profile_df):,} rows")
                with st.expander("Preview uploaded campaign profiles"):
                    st.dataframe(campaign_intent_profile_df.head(25), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading campaign intent profile CSV: {e}")
                return
        else:
            st.info("Upload campaign intent profile CSV to continue.")
            return

    required_camp_cols = {"campaign_id", "campaign_name", "intent_id", "intent_name", "weight"}
    missing = required_camp_cols - set(campaign_intent_profile_df.columns)
    if missing:
        st.error(f"campaign_intent_profile_df missing columns: {sorted(list(missing))}")
        st.info("Expected columns: campaign_id, campaign_name, intent_id, intent_name, weight")
        return

    # Normalize types
    campaign_intent_profile_df["campaign_id"] = campaign_intent_profile_df["campaign_id"].astype(str)
    campaign_intent_profile_df["campaign_name"] = campaign_intent_profile_df["campaign_name"].astype(str)
    campaign_intent_profile_df["intent_id"] = campaign_intent_profile_df["intent_id"].astype(str)
    campaign_intent_profile_df["intent_name"] = campaign_intent_profile_df["intent_name"].astype(str)
    campaign_intent_profile_df["weight"] = pd.to_numeric(campaign_intent_profile_df["weight"], errors="coerce").fillna(0.0)

    customer_intent_profile_df["customer_id"] = customer_intent_profile_df["customer_id"].astype(str)
    customer_intent_profile_df["intent_id"] = customer_intent_profile_df["intent_id"].astype(str)
    customer_intent_profile_df["intent_name"] = customer_intent_profile_df["intent_name"].astype(str)
    customer_intent_profile_df["intent_weight"] = pd.to_numeric(customer_intent_profile_df["intent_weight"], errors="coerce").fillna(0.0)

    st.subheader("6.1 Audience Filters (optional)")
    f1, f2, f3 = st.columns(3)
    with f1:
        min_customer_total = st.number_input(
            "Min customer_total_value (if available)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="step6_min_customer_total"
        )
    with f2:
        top_n_customers = st.number_input(
            "Top N customers per campaign",
            min_value=50,
            max_value=200000,
            value=5000,
            step=50,
            key="step6_top_n_customers"
        )
    with f3:
        top_explain = st.number_input(
            "Top matched intents in explanation",
            min_value=1,
            max_value=10,
            value=3,
            key="step6_top_explain"
        )

    # Detect totals column
    cust_total_col = None
    for c in ["customer_total_value", "customer_total_amt", "customer_total_spend"]:
        if c in customer_intent_profile_df.columns:
            cust_total_col = c
            break

    run_btn = st.button("🎯 Rank Audience (Campaign × Customers)", type="primary", use_container_width=True)

    if run_btn:
        try:
            with st.status("Scoring customers per campaign…", expanded=True) as status:
                merged = campaign_intent_profile_df.merge(
                    customer_intent_profile_df,
                    on=["intent_id"],
                    suffixes=("_camp", "_cust"),
                    how="inner"
                )
                if merged.empty:
                    st.error("❌ No matching intent_id between campaign profiles and customer profiles.")
                    return

                merged["score_contribution"] = merged["weight"] * merged["intent_weight"]
                merged["intent_name_final"] = merged["intent_name_camp"].fillna(merged["intent_name_cust"])

                scored = (
                    merged.groupby(["campaign_id", "campaign_name", "customer_id"], as_index=False)
                    .agg(
                        match_score=("score_contribution", "sum"),
                        matched_intents=("intent_id", "nunique")
                    )
                )

                # Optional join totals for filtering
                if cust_total_col is not None:
                    cust_totals = (
                        customer_intent_profile_df.groupby("customer_id", as_index=False)[cust_total_col]
                        .max()
                        .rename(columns={cust_total_col: "customer_total"})
                    )
                    scored = scored.merge(cust_totals, on="customer_id", how="left")
                    scored["customer_total"] = scored["customer_total"].fillna(0.0)
                    if float(min_customer_total) > 0:
                        scored = scored[scored["customer_total"] >= float(min_customer_total)]

                if scored.empty:
                    st.warning("⚠️ No customers remain after filtering.")
                    return

                # Rank per campaign
                scored["rank"] = (
                    scored.sort_values(["campaign_id", "match_score"], ascending=[True, False])
                    .groupby("campaign_id")
                    .cumcount() + 1
                )
                scored = scored[scored["rank"] <= int(top_n_customers)].copy()

                # Explainability: top contributing intents
                merged = merged.merge(scored[["campaign_id", "customer_id"]], on=["campaign_id", "customer_id"], how="inner")
                merged["intent_contrib_rank"] = (
                    merged.sort_values(["campaign_id", "customer_id", "score_contribution"], ascending=[True, True, False])
                    .groupby(["campaign_id", "customer_id"])
                    .cumcount() + 1
                )
                explain = merged[merged["intent_contrib_rank"] <= int(top_explain)].copy()

                explain["explain_piece"] = (
                    explain["intent_name_final"].astype(str)
                    + " (camp="
                    + explain["weight"].round(4).astype(str)
                    + " × cust="
                    + explain["intent_weight"].round(4).astype(str)
                    + " = "
                    + explain["score_contribution"].round(6).astype(str)
                    + ")"
                )

                explain_agg = (
                    explain.groupby(["campaign_id", "customer_id"], as_index=False)
                    .agg(explanation=("explain_piece", lambda s: " | ".join(list(s))))
                )

                campaign_audience_ranked_df = (
                    scored.merge(explain_agg, on=["campaign_id", "customer_id"], how="left")
                    .sort_values(["campaign_id", "rank"])
                    .reset_index(drop=True)
                )

                st.session_state["campaign_audience_ranked_df"] = campaign_audience_ranked_df
                status.update(label="✅ Ranking complete!", state="complete", expanded=False)

            st.success(f"✅ Built campaign_audience_ranked_df: {len(st.session_state['campaign_audience_ranked_df']):,} rows")

        except Exception as e:
            st.error(f"❌ Error ranking audience: {e}")
            st.exception(e)

    if st.session_state["campaign_audience_ranked_df"] is not None:
        st.subheader("🏆 Ranked Audience (Preview)")
        out = st.session_state["campaign_audience_ranked_df"]
        st.dataframe(out.head(500), use_container_width=True, height=440)
        st.caption(f"Showing up to 500 rows. Total rows: {len(out):,}")
        df_download_button("📥 Download campaign_audience_ranked.csv", out, "campaign_audience_ranked.csv")


# =============================================================================
# FINAL ROUTER + UX/UI IMPROVEMENTS
# =============================================================================
def run_app():
    # Sidebar navigation (better than one long page)
    with st.sidebar:
        st.markdown("## 🧠 Semantic Audience Studio")
        st.caption("Navigate steps. Session state persists across pages.")

        pages = [
            "Step 0 — Campaign Input",
            "Step 1 — Upload Data",
            "Step 2 — Ontology",
            "Step 3 — Campaign Intent Profile",
            "Step 4 — Product Labeling",
            "Step 5 — Customer Profile",
            "Step 6 — Audience Builder",
        ]
        page = st.radio("Go to", pages, index=0, key="nav_page")

        st.divider()
        st.markdown("### ✅ Readiness")
        st.write("Campaigns:", "✅" if st.session_state.get("campaigns_df") is not None else "—")
        st.write("Catalog:", "✅" if st.session_state.get("catalog_df") is not None else "—")
        st.write("Transactions:", "✅" if st.session_state.get("txn_df") is not None else "—")
        st.write("Ontology:", "✅" if st.session_state.get("ontology") is not None else "—")
        st.write("Product Labels:", "✅" if st.session_state.get("product_intent_labels_df") is not None else "—")
        st.write("Customer Profile:", "✅" if st.session_state.get("customer_intent_profile_df") is not None else "—")
        st.write("Audience Ranked:", "✅" if st.session_state.get("campaign_audience_ranked_df") is not None else "—")

        st.divider()
        if st.button("🧹 Clear ALL session data", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # Main page
    if page.startswith("Step 0"):
        render_step0()
    elif page.startswith("Step 1"):
        render_step1()
        render_data_summary()  # nice to keep summary near data upload
    elif page.startswith("Step 2"):
        render_step2()
    elif page.startswith("Step 3"):
        render_step3()
    elif page.startswith("Step 4"):
        render_step4()
    elif page.startswith("Step 5"):
        render_step5()
    elif page.startswith("Step 6"):
        render_step6()


# Call the router at the end of the file
run_app()

