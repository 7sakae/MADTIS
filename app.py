import streamlit as st
import pandas as pd
import json
import re
import time
import random

# =========================
# LLM JSON Robust Helpers
# =========================

def trim_to_balanced_json(s: str) -> str:
    """
    Trim to the first complete JSON object/array (balanced braces/brackets),
    respecting quoted strings and escapes.
    """
    stack = []
    in_str = False
    esc = False

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                return s[:i + 1]
            expected = stack[-1]
            if ch != expected:
                return s[:i + 1]
            stack.pop()
            if not stack:
                return s[:i + 1]

    return s


def repair_unescaped_quotes_in_strings(s: str) -> str:
    """
    Repairs the common LLM bug:
    "reason":"The "autumn-scented" aspect..."
                 ^ unescaped quote inside string
    Strategy: while inside a JSON string, if we see a quote that DOESN'T look like
    the end of the string (next non-space not in ,}] ), convert it to a single quote.
    """
    out = []
    in_str = False
    esc = False

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                out.append(ch)
                esc = False
                continue

            if ch == "\\":
                out.append(ch)
                esc = True
                continue

            if ch == '"':
                # Peek ahead to decide if this is end-of-string or an internal quote
                j = i + 1
                while j < len(s) and s[j] in " \r\n\t":
                    j += 1

                if j >= len(s) or s[j] in ",}]":
                    # Looks like a real closing quote
                    in_str = False
                    out.append('"')
                else:
                    # Looks like an unescaped internal quote -> replace
                    out.append("'")
                continue

            out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)

    return "".join(out)


def extract_json_from_text_robust(raw: str):
    """
    Handles:
    - ```json fences
    - leading commentary before JSON
    - trailing commentary after JSON
    - trims to first balanced JSON object/array
    - repairs unescaped quotes inside strings (1 retry)
    """
    text = (raw or "").strip()

    # Remove fences if present
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    # Find first { or [
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise json.JSONDecodeError("No JSON object/array found in model output", text, 0)

    text = text[min(starts):].strip()
    text = trim_to_balanced_json(text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = repair_unescaped_quotes_in_strings(text)
        return json.loads(fixed)


def debug_json_error(raw: str, e: json.JSONDecodeError, context: int = 180) -> str:
    s = raw or ""
    i = getattr(e, "pos", 0)
    start = max(0, i - context)
    end = min(len(s), i + context)
    snippet = s[start:end]
    caret_pos = max(0, i - start)
    return (
        f"{str(e)}\n"
        f"--- context around char {i} ---\n"
        f"{snippet}\n"
        f"{' ' * caret_pos}^\n"
    )


def parse_retry_delay_seconds(err_text: str) -> int:
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}", err_text)
    return int(m.group(1)) if m else 0


def make_gemini_model(api_key: str, model_name: str, json_mode: bool = True, temperature: float = 0.2):
    """
    Creates a Gemini model with JSON mode enabled (strongly reduces broken JSON).
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    if json_mode:
        try:
            gen_cfg = genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=temperature
            )
            return genai.GenerativeModel(model_name, generation_config=gen_cfg)
        except Exception:
            # Fallback if GenerationConfig signature differs
            return genai.GenerativeModel(model_name)

    return genai.GenerativeModel(model_name)


def call_llm_with_retry(model, prompt: str, rpm_limit: int, max_retries: int, last_ts_key: str):
    """
    Throttle + retry 429-safe call. Stores timestamp in st.session_state[last_ts_key].
    """
    min_interval = 60.0 / max(float(rpm_limit), 1.0)

    last_call = st.session_state.get(last_ts_key, 0.0)
    now = time.time()
    wait = (last_call + min_interval) - now
    if wait > 0:
        time.sleep(wait)

    for attempt in range(int(max_retries) + 1):
        try:
            resp = model.generate_content(prompt)
            st.session_state[last_ts_key] = time.time()
            return getattr(resp, "text", "") or ""
        except Exception as e:
            msg = str(e)
            if "429" in msg or "ResourceExhausted" in msg or "quota" in msg.lower():
                retry_s = parse_retry_delay_seconds(msg)
                if retry_s <= 0:
                    retry_s = int(min(60, (2 ** attempt) * 5))
                retry_s = retry_s + random.randint(1, 3)
                st.warning(
                    f"⚠️ Rate limit hit (429). Sleeping {retry_s}s then retrying... "
                    f"(attempt {attempt+1}/{int(max_retries)+1})"
                )
                time.sleep(retry_s)
                continue
            raise

    raise RuntimeError("Exceeded retry attempts due to rate limiting.")


def chunk_df(df, size: int):
    for start in range(0, len(df), size):
        yield df.iloc[start:start + size]


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


def safe_json_snippet(obj, max_chars=16000):
    txt = json.dumps(obj, ensure_ascii=False)
    return txt[:max_chars]
# =========================
# Token / Chunk Advisor (NO API)
# =========================
def approx_tokens_from_text(text: str, mode: str = "chars", chars_per_token: float = 4.0) -> float:
    """
    Cheap token estimate (no API calls).
    mode:
      - "chars": tokens ~= len(text) / chars_per_token
      - "bytes": tokens ~= len(text.encode('utf-8')) / chars_per_token   (better for TH/mixed)
    """
    if text is None:
        text = ""
    s = str(text)
    if mode == "bytes":
        n = len(s.encode("utf-8"))
    else:
        n = len(s)
    return float(n) / max(float(chars_per_token), 0.1)


def build_step2_overhead_prompt(language: str) -> str:
    """
    This MUST mirror your Step 2 prompt template BUT with NO examples injected.
    Used only to estimate fixed prompt overhead.
    """
    return f"""
You are proposing a Lifestyle→Intent ontology for retail marketing.

Input: Product catalog examples (titles + descriptions):

Task:
- Propose Lifestyle parents and Intents under each.
- Each Intent must include:
  intent_name (2–5 words), definition (1 sentence),
  include_examples (2–3), exclude_examples (1–2).
- Output language: {language}

Rules:
- Return JSON ONLY. No markdown. No extra text.
- Do NOT use double quotes inside any string fields. Use parentheses or single quotes if needed.

Return STRICT minified JSON:
{{"lifestyles":[{{"lifestyle_name":"...","definition":"...","intents":[{{"intent_name":"...","definition":"...","include_examples":["..."],"exclude_examples":["..."]}}]}}]}}
""".strip()


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
            "campaign_brief": "Celebrate romance and affection with curated gifting and date-night bundles, featuring fragrances, chocolates, candles, dinner-at-home kits, and self-care sets designed for couples and self-love moments."
        },
        {
            "campaign_id": "CAMP_MOTHERSDAY",
            "campaign_name": "Mother’s Day",
            "campaign_brief": "Focus on gratitude-driven gifting for moms and parents with comfort-forward home items, premium personal care, wellness essentials, and heartfelt gift sets that feel warm, thoughtful, and special."
        },
        {
            "campaign_id": "CAMP_FATHERSDAY",
            "campaign_name": "Father’s Day",
            "campaign_brief": "Highlight practical and utility-led gifting with grooming kits, tech accessories, tools, hobby-related items, and everyday bundles that feel useful while still being celebratory."
        },
        {
            "campaign_id": "CAMP_GRAD",
            "campaign_name": "Graduation Season",
            "campaign_brief": "Support milestone gifting and career transition needs with productivity tech, professional bags, style essentials, desk upgrades, and meaningful keepsakes that mark the next chapter."
        },
        {
            "campaign_id": "CAMP_WEDDING",
            "campaign_name": "Wedding Season",
            "campaign_brief": "Enable wedding and couple-focused gifting with premium bundles, home setup essentials, celebration-ready items, and registry-style picks that help newlyweds build a shared life."
        },
        {
            "campaign_id": "CAMP_LUNAR_NEWYEAR",
            "campaign_name": "Lunar New Year",
            "campaign_brief": "Lean into family reunion and home hosting with auspicious gifting, festive table essentials, home refresh items, and celebration supplies that match tradition, luck, and togetherness."
        },
        {
            "campaign_id": "CAMP_RAMADAN_EID",
            "campaign_name": "Ramadan & Eid",
            "campaign_brief": "Center family gatherings and culturally mindful hosting with food prep staples, modest personal care, home-serving essentials, and Eid gifting traditions that feel respectful and celebratory."
        },
        {
            "campaign_id": "CAMP_DIWALI",
            "campaign_name": "Diwali",
            "campaign_brief": "Capture festive home celebration with lights and decor, new-outfit moments, sweet gifting, and traditional home items that elevate warmth, togetherness, and joyful rituals."
        },
        {
            "campaign_id": "CAMP_EASTER",
            "campaign_name": "Easter",
            "campaign_brief": "Activate family gatherings and spring refresh with light home decor, seasonal sweets, small gifting bundles, and playful hosting items that fit cheerful, pastel, and fresh-start vibes."
        },
        {
            "campaign_id": "CAMP_HALLOWEEN",
            "campaign_name": "Halloween",
            "campaign_brief": "Drive costumes and party hosting with themed decor, spooky accessories, trick-or-treat treats, and fun confectionery bundles that make celebrations easy and photogenic."
        },
        {
            "campaign_id": "CAMP_THANKSGIVING",
            "campaign_name": "Thanksgiving",
            "campaign_brief": "Support hosting-heavy family meals with kitchen prep tools, serveware, comfort-driven home essentials, and warm seasonal items that make gatherings feel cozy and complete."
        },
        {
            "campaign_id": "CAMP_BACKTOSCHOOL",
            "campaign_name": "Back to School",
            "campaign_brief": "Prepare students and parents for the new term with stationery, backpacks, learning gadgets, organizational tools, and study-friendly bundles that improve readiness and productivity."
        },
        {
            "campaign_id": "CAMP_SUMMERTRAVEL",
            "campaign_name": "Summer Travel / Vacation",
            "campaign_brief": "Power travel prep and outdoor living with sun protection, beach gear, hydration essentials, lightweight packing sets, and convenient on-the-go kits for holidays and weekend trips."
        },
        {
            "campaign_id": "CAMP_WINTERWARM",
            "campaign_name": "Winter Warm-Up",
            "campaign_brief": "Emphasize cold-season comfort and wellness with warm clothing, protective gear, soothing personal care, and cozy home items that support staying warm, healthy, and relaxed."
        },
        {
            "campaign_id": "CAMP_SPRINGCLEAN",
            "campaign_name": "Spring Cleaning / Home Refresh",
            "campaign_brief": "Motivate a seasonal reset with cleaning supplies, storage solutions, decluttering tools, and home-organization bundles that make refresh projects feel simple and satisfying."
        },
        {
            "campaign_id": "CAMP_NEWYEAR",
            "campaign_name": "New Year / Fresh Start",
            "campaign_brief": "Tap into renewal and goal-setting with home reset items, wellness routines, productivity upgrades, and “new year, new habits” bundles that help customers start strong and feel refreshed."
        },
        {
            "campaign_id": "CAMP_MIDAUTUMN",
            "campaign_name": "Mid-Autumn Festival",
            "campaign_brief": "Celebrate family togetherness and seasonal traditions with mooncake gifting, tea and table pairings, premium snack bundles, lantern-themed decor, and hosting essentials for reunion moments."
        },
        {
            "campaign_id": "CAMP_CHRISTMAS",
            "campaign_name": "Christmas / Year-End Holidays",
            "campaign_brief": "Enable gift-giving and festive hosting with home decor, family celebration sets, premium style gifts, party-ready bundles, and seasonal treats that fit end-of-year togetherness."
        }
    ]

    json_text = st.text_area(
        "Paste campaign JSON (one campaign object or a list of objects)",
        value=json.dumps(default_json, indent=2, ensure_ascii=False),
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

                catalog_df = product_df[["product_id", "product_title", "product_text","product_category"]].copy()
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

    # NEW: Category tab as FIRST tab
    tab0, tab1, tab2, tab3 = st.tabs(["🗂️ Categories", "📦 Products", "🛒 Transactions", "📈 Analytics"])

    with tab0:
        st.subheader("Category Overview")

        # Auto-detect a category-like column
        candidate_cols = [
            "category_name", "category", "cat_name", "cat",
            "department", "dept", "product_category", "subcategory_name", "subcategory"
        ]
        cat_col = next((c for c in candidate_cols if c in catalog_df.columns), None)

        if not cat_col:
            st.info("👆 No category column found in product catalog. Add a column like `category_name` or `category` to enable this tab.")
        else:
            # Distinct products per category
            cat_summary = (
                catalog_df.assign(_cat=catalog_df[cat_col].fillna("Unknown").astype(str).str.strip())
                .groupby("_cat")["product_id"]
                .nunique()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"_cat": "Category", "product_id": "Distinct Products"})
            )

            total_distinct = int(catalog_df["product_id"].nunique())
            cat_summary["Share"] = (cat_summary["Distinct Products"] / max(total_distinct, 1)).map(lambda x: f"{x*100:.1f}%")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Category Column", cat_col)
            with c2:
                st.metric("Total Categories", f"{cat_summary['Category'].nunique():,}")
            with c3:
                st.metric("Total Distinct Products", f"{total_distinct:,}")

            st.dataframe(cat_summary, use_container_width=True, height=420)

            st.download_button(
                "📥 Download Category Overview",
                data=cat_summary.to_csv(index=False).encode("utf-8"),
                file_name="category_overview.csv",
                mime="text/csv"
            )

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
            .reset_index(name="transaction_count")
        )

        top_products = top_products.merge(
            catalog_df[["product_id", "product_name"]],
            on="product_id",
            how="left"
        )

        top_products = top_products.rename(columns={
            "product_id": "Product ID",
            "product_name": "Product Name",
            "transaction_count": "Transaction Count"
        })

        st.dataframe(top_products[["Product ID", "Product Name", "Transaction Count"]], use_container_width=True)

else:
    st.info("👆 Upload both Product and Transaction CSV files to see the data summary.")

st.divider()

# ============================================================================
# CATEGORY DISTRIBUTION STATS
# ============================================================================


cat_col = "product_category"

if "catalog_df" in st.session_state:
    catalog_df = st.session_state["catalog_df"]

    if cat_col in catalog_df.columns and "product_id" in catalog_df.columns:
        # Normalize category text
        cat_series = (
            catalog_df[cat_col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        # Count DISTINCT products per category (safe even if duplicates exist)
        cat_counts = (
            catalog_df.assign(_cat=cat_series)
            .groupby("_cat")["product_id"]
            .nunique()
            .sort_values(ascending=False)
        )

        total_distinct_products = int(catalog_df["product_id"].nunique())
        n_categories = int(cat_counts.shape[0])

        # Key concentration stats
        top1_share = float(cat_counts.iloc[0] / max(total_distinct_products, 1)) if n_categories > 0 else 0.0
        top3_share = float(cat_counts.iloc[:3].sum() / max(total_distinct_products, 1)) if n_categories > 0 else 0.0

        # "Rare" categories — choose ONE rule (keep it simple)
        # Rule A: fewer than 30 distinct products in a category
        rare_threshold_n = 30
        rare_cat_count = int((cat_counts < rare_threshold_n).sum())

        # Rule B (optional): categories <1% share (uncomment if you prefer)
        # rare_threshold_pct = 0.01
        # rare_cat_count = int(((cat_counts / max(total_distinct_products, 1)) < rare_threshold_pct).sum())

        # --------------------------------------------------------------------
        # UI: show stats block (place this AFTER your category overview table)
        # --------------------------------------------------------------------
        st.markdown("### 🧪 Category Distribution Stats (for chunking decision)")

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total categories", f"{n_categories:,}")
        with s2:
            st.metric("Top-1 share", f"{top1_share*100:.1f}%")
        with s3:
            st.metric("Top-3 share", f"{top3_share*100:.1f}%")
        with s4:
            st.metric(f"Rare categories (<{rare_threshold_n})", f"{rare_cat_count:,}")

        # # Short explanation (keep readable for demo)
        # st.caption(
        #     "Why this matters: If the catalog is heavily concentrated (high Top-1/Top-3 share), "
        #     "LLM chunk prompts will over-see the dominant category and may bias Lifestyle parents. "
        #     "In that case, use imbalance-aware chunking (e.g., stratified / cap) instead of simple shuffle."
        # )

        # Optional: simple recommendation label (no action yet)
        # Thresholds are easy to explain and tweak
        if top1_share <= 0.35:
            reco_mode = "Shuffle chunking (balanced enough)"
        elif top1_share <= 0.55:
            reco_mode = "Imbalance-aware chunking (stratified mix)"
        else:
            reco_mode = "Imbalance-aware chunking (cap + stratified mix)"

        st.info(f"✅ Suggested chunking mode: **{reco_mode}**")

    else:
        st.info("👆 Category stats unavailable: `product_category` (or `product_id`) not found in catalog_df.")


# ============================================================================
# STEP 2: AI-POWERED ONTOLOGY GENERATION (OR REUSE VIA UPLOAD)
# ============================================================================
st.header("Step 2: AI-Powered Ontology Generation")
st.caption("Option A: Generate with Gemini • Option B: Upload a previously-downloaded ontology_v1.json to reuse")

if "catalog_df" in st.session_state:
    catalog_df = st.session_state["catalog_df"]

    # -------------------------
    # Option B: Reuse Ontology (Upload)
    # -------------------------
    st.subheader("♻️ Reuse Existing Ontology (Upload)")

    with st.expander("Upload ontology_v1.json to reuse (recommended for demos / save quota)", expanded=True):
        uploaded_ontology_json = st.file_uploader(
            "Upload ontology JSON (e.g., ontology_v1.json)",
            type=["json"],
            key="upload_ontology_json"
        )

        col_u1, col_u2, col_u3 = st.columns([1, 1, 2])
        with col_u1:
            load_uploaded_ontology_btn = st.button(
                "Load Uploaded Ontology",
                type="primary",
                key="load_uploaded_ontology_btn"
            )
        with col_u2:
            clear_ontology_btn = st.button("Clear Ontology (session)", key="clear_ontology_btn")
        with col_u3:
            st.caption("Loads ontology + rebuilds dim_lifestyle and dim_intent in session_state. No API key needed.")

        if clear_ontology_btn:
            for k in ["ontology", "dim_lifestyle_df", "dim_intent_df"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.success("✅ Cleared ontology from session_state.")

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

                    st.success(
                        f"✅ Loaded ontology from upload: {len(dim_lifestyle_df)} lifestyles, {len(dim_intent_df)} intents"
                    )

                except Exception as e:
                    st.error(f"❌ Failed to load uploaded ontology: {e}")

    st.divider()

    # -------------------------
    # Option A: Generate Ontology with Gemini
    # -------------------------
    st.subheader("🤖 Generate Ontology with AI (Gemini)")

    # =========================
    # Chunking Helpers (CSI)
    # =========================
    def build_order_category_stratified_interleaving(
        df: pd.DataFrame,
        cat_col: str = "product_category",
        seed: int = 42,
    ) -> list:
        """
        Chunking method: Category-Stratified Interleaving (CSI)
        - Shuffle items within each category bucket
        - Then interleave categories (round-robin pick 1 per category) until exhausted
        Result: early/mid chunks have better category diversity, reducing ontology bias.
        Fallback: if cat_col missing -> shuffle all rows.
        Returns: list of row indices in the order we feed the LLM.
        """
        rng = random.Random(int(seed))

        if cat_col not in df.columns:
            idxs = list(df.index)
            rng.shuffle(idxs)
            return idxs

        cats = (
            df[cat_col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

        buckets = {}
        for i, c in zip(df.index.tolist(), cats.tolist()):
            buckets.setdefault(c, []).append(i)

        for c in list(buckets.keys()):
            rng.shuffle(buckets[c])

        # Big categories first (helps them show up early), then interleave
        cat_order = sorted(buckets.keys(), key=lambda k: len(buckets[k]), reverse=True)

        ordered = []
        exhausted = 0
        while exhausted < len(cat_order):
            exhausted = 0
            for c in cat_order:
                if buckets[c]:
                    ordered.append(buckets[c].pop())
                else:
                    exhausted += 1
        return ordered


    def chunk_list(xs, size: int):
        for start in range(0, len(xs), int(size)):
            yield xs[start : start + int(size)]


    # =========================
    # API Key
    # =========================
    gemini_api_key = None
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini API key loaded from secrets")
    else:
        gemini_api_key = st.text_input(
            "Enter your Gemini API Key (only needed if you generate)",
            type="password",
            help="Get your API key from https://aistudio.google.com/apikey",
            key="gemini_key_step2"
        )
        if gemini_api_key:
            st.info("💡 Tip: Add your API key to Streamlit secrets for persistence")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Ontology Settings**")

        n_lifestyles = st.number_input(
            "Number of Lifestyle Categories",
            min_value=3, max_value=150, value=6,
            key="step2_n_lifestyles"
        )

        max_intents_per_lifestyle = st.number_input(
            "Max Intents per Lifestyle",
            min_value=2, max_value=100, value=5,
            key="step2_max_intents"
        )

        chunk_size = st.number_input(
            "Chunk Size (products per API call)",
            min_value=20, max_value=100, value=40,
            key="step2_chunk_size"
        )

        product_slice_chars = st.number_input(
            "Max characters per product sent in Step 2 (slice length)",
            min_value=80, max_value=1200, value=240, step=20,
            key="step2_product_slice_chars"
        )

        chunk_seed = st.number_input(
            "Interleaving seed (reproducible mix)",
            min_value=1, max_value=999999, value=42, step=1,
            key="step2_chunk_seed"
        )

        language = st.selectbox(
            "Output Language",
            ["en", "th", "zh", "ja", "es", "fr"],
            key="step2_lang"
        )

        st.caption("Chunking method: **Category-Stratified Interleaving (CSI)** (interleaves across product_category)")

        # =========================
        # Chunk Size Advisor (EXECUTIVE, NO API) — 2 columns wide
        # =========================
        with st.expander("📏 Chunk Size Advisor (Executive view)", expanded=False):
            adv_left, adv_right = st.columns([1, 1])

            with adv_left:
                st.markdown("#### Inputs")
                est_mode = st.radio(
                    "Estimator mode",
                    ["chars (simple, good for EN)", "bytes (better for TH/mixed)"],
                    index=1,
                    horizontal=True,
                    key="step2_token_est_mode_exec"
                )
                mode_key = "bytes" if str(est_mode).startswith("bytes") else "chars"

                chars_per_token = st.slider(
                    "Chars-per-token factor (lower = more tokens)",
                    min_value=2.0,
                    max_value=6.0,
                    value=4.0,
                    step=0.5,
                    key="step2_chars_per_token_exec"
                )

                token_budget = st.select_slider(
                    "Per-call token budget (estimate)",
                    options=[6000, 8000, 12000, 16000, 20000],
                    value=12000,
                    key="step2_token_budget_exec"
                )

                st.markdown("#### Cost inputs")
                tokens_per_dollar = st.number_input(
                    "Tokens per $1 (your pricing assumption)",
                    min_value=1000.0,
                    max_value=5_000_000.0,
                    value=250_000.0,
                    step=10_000.0,
                    key="step2_tokens_per_dollar_exec"
                )

                safety_multiplier = st.slider(
                    "Safety multiplier (output + variance)",
                    min_value=1.0,
                    max_value=2.0,
                    value=1.3,
                    step=0.05,
                    key="step2_safety_mult_exec"
                )

            with adv_right:
                st.markdown("#### Executive KPIs")

                # Mirrors your prompt examples (slice)
                sent_series = (
                    catalog_df["product_text"]
                    .fillna("")
                    .astype(str)
                    .str.slice(0, int(product_slice_chars))
                )

                avg_tokens_per_product = sent_series.apply(
                    lambda s: approx_tokens_from_text(s, mode=mode_key, chars_per_token=chars_per_token)
                ).mean()

                overhead_prompt = build_step2_overhead_prompt(language)
                overhead_tokens = approx_tokens_from_text(
                    overhead_prompt,
                    mode=mode_key,
                    chars_per_token=chars_per_token
                )

                cs = int(chunk_size)
                n_products = int(len(catalog_df))

                est_tokens_per_call = float(overhead_tokens + avg_tokens_per_product * cs)
                est_calls = int((n_products + cs - 1) // cs)

                reco = int((float(token_budget) - float(overhead_tokens)) / max(float(avg_tokens_per_product), 1.0))
                reco = max(5, min(reco, 100))

                est_total_tokens_prompt = est_tokens_per_call * est_calls
                est_total_tokens_allin = est_total_tokens_prompt * float(safety_multiplier)
                est_cost = est_total_tokens_allin / max(float(tokens_per_dollar), 1.0)

                k1, k2 = st.columns(2)
                with k1:
                    st.metric("Avg tokens / product", f"{avg_tokens_per_product:,.1f}")
                with k2:
                    st.metric("Fixed overhead / call", f"{overhead_tokens:,.0f}")

                k3, k4 = st.columns(2)
                with k3:
                    st.metric("Est tokens / call", f"{est_tokens_per_call:,.0f}")
                with k4:
                    st.metric("Est #calls", f"{est_calls:,}")

                k5, k6 = st.columns(2)
                with k5:
                    st.metric("Recommended chunk size", f"{reco} (under {int(token_budget):,} tokens)")
                with k6:
                    st.metric("Est total cost ($)", f"{est_cost:,.2f}")

                st.caption(
                    "Rough estimate (no API calls). Cost uses your tokens-per-$ assumption and a safety multiplier."
                )

    with col2:
        st.write("**Actions**")
        generate_btn = st.button(
            "🤖 Generate with AI",
            type="primary",
            use_container_width=True,
            key="step2_generate_btn"
        )
        st.info(f"Will analyze {len(catalog_df)} products using Gemini 2.5 Flash")

    if generate_btn:
        if not gemini_api_key:
            st.error("⚠️ Please provide a Gemini API key to generate ontology (or upload an existing ontology above).")
        else:
            try:
                model = make_gemini_model(gemini_api_key, "gemini-2.5-flash", json_mode=True, temperature=0.2)

                order = build_order_category_stratified_interleaving(
                    catalog_df,
                    cat_col="product_category",
                    seed=int(chunk_seed),
                )
                ordered_texts = catalog_df.loc[order, "product_text"].fillna("").astype(str).tolist()

                with st.spinner(f"🤖 Analyzing products in chunks of {chunk_size} (CSI: category-stratified interleaving)..."):
                    chunk_outputs = []
                    progress_bar = st.progress(0)
                    total_chunks = (len(ordered_texts) + int(chunk_size) - 1) // int(chunk_size)

                    for idx, chunk in enumerate(chunk_list(ordered_texts, int(chunk_size))):
                        examples = "\n".join([f"- {t[:int(product_slice_chars)]}" for t in chunk])

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

Rules:
- Return JSON ONLY. No markdown. No extra text.
- Do NOT use double quotes inside any string fields. Use parentheses or single quotes if needed.

Return STRICT minified JSON:
{{"lifestyles":[{{"lifestyle_name":"...","definition":"...","intents":[{{"intent_name":"...","definition":"...","include_examples":["..."],"exclude_examples":["..."]}}]}}]}}
""".strip()

                        raw = call_llm_with_retry(
                            model=model,
                            prompt=prompt,
                            rpm_limit=8,
                            max_retries=4,
                            last_ts_key="_last_llm_call_ts_step2"
                        )

                        try:
                            chunk_outputs.append(extract_json_from_text_robust(raw))
                        except json.JSONDecodeError as e:
                            st.error("❌ Invalid JSON returned by the model (Step 2 chunk).")
                            st.code(debug_json_error(raw, e))
                            st.code((raw or "")[:1200])
                            st.stop()

                        progress_bar.progress((idx + 1) / max(total_chunks, 1))

                    st.success(f"✅ Analyzed {total_chunks} chunks (CSI: category-stratified interleaving)")

                with st.spinner("🔄 Consolidating ontology..."):
                    pool = {}
                    for obj in chunk_outputs:
                        for ls in obj.get("lifestyles", []):
                            ls_name = str(ls.get("lifestyle_name", "")).strip()
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

Rules:
- Return JSON ONLY. No markdown. No extra text.
- Do NOT use double quotes inside any string fields. Use parentheses or single quotes if needed.

Return STRICT minified JSON:
{{"lifestyles":[{{"lifestyle_id":"LS_...","lifestyle_name":"...","definition":"...","intents":[{{"intent_id":"IN_...","intent_name":"...","definition":"...","include_examples":["..."],"exclude_examples":["..."]}}]}}]}}
""".strip()

                    raw2 = call_llm_with_retry(
                        model=model,
                        prompt=prompt2,
                        rpm_limit=8,
                        max_retries=4,
                        last_ts_key="_last_llm_call_ts_step2"
                    )

                    try:
                        ontology_data = extract_json_from_text_robust(raw2)
                    except json.JSONDecodeError as e:
                        st.error("❌ Invalid JSON returned by the model (Step 2 consolidation).")
                        st.code(debug_json_error(raw2, e))
                        st.code((raw2 or "")[:1200])
                        st.stop()

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
                            "n_intents": len(dim_intent_df),
                            "chunking_method": "Category-Stratified Interleaving (CSI)",
                            "chunk_size": int(chunk_size),
                            "product_slice_chars": int(product_slice_chars),
                            "seed": int(chunk_seed),
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

    # -------------------------
    # Display & Downloads (works for both generated or uploaded ontology)
    # -------------------------
    if "ontology" in st.session_state:
        st.divider()
        st.subheader("📥 Ontology Files")

        ontology = st.session_state["ontology"]
        dim_lifestyle_df = st.session_state.get("dim_lifestyle_df", pd.DataFrame())
        dim_intent_df = st.session_state.get("dim_intent_df", pd.DataFrame())

        tab1, tab2, tab3 = st.tabs(["📋 Ontology JSON", "🎨 Lifestyle Dimensions", "🎯 Intent Dimensions"])

        with tab1:
            with st.expander("Preview ontology JSON (collapsed)", expanded=False):
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
    st.info("👆 Upload Product CSV in Step 1 to enable ontology generation or reuse.")

# ============================================================================
# STEP 3: CAMPAIGN → WEIGHTED INTENT PROFILE (LLM)  +  (UPLOAD TO REUSE)
# ============================================================================
st.divider()
st.header("Step 3: Campaign → Weighted Intent Profile (LLM)")
st.caption("Option A: Generate with Gemini • Option B: Upload an existing campaign_intent_profile.csv for reuse (no LLM).")

if "campaigns_df" not in st.session_state:
    st.info("👆 Please load campaign input in Step 0 first.")
    st.stop()

campaigns_df = st.session_state["campaigns_df"].copy()

# Small helper (avoid dependency if you didn’t define it elsewhere)
def chunk_df(df: pd.DataFrame, size: int):
    for start in range(0, len(df), int(size)):
        yield df.iloc[start : start + int(size)]

# -------------------------
# Option B: Upload to reuse
# -------------------------
st.subheader("♻️ Reuse Existing Campaign Intent Profile (Upload)")

with st.expander("Upload campaign_intent_profile.csv to reuse (recommended for demos / save quota)", expanded=True):
    uploaded_campaign_profile = st.file_uploader(
        "Upload campaign intent profile CSV (e.g., campaign_intent_profile.csv)",
        type=["csv"],
        key="upload_campaign_intent_profile_csv"
    )

    col_u1, col_u2, col_u3 = st.columns([1, 1, 2])
    with col_u1:
        load_uploaded_campaign_profile_btn = st.button(
            "Load Uploaded Campaign Profile",
            type="primary",
            key="load_uploaded_campaign_profile_btn"
        )
    with col_u2:
        clear_campaign_profile_btn = st.button(
            "Clear Campaign Profile (session)",
            key="clear_campaign_profile_btn"
        )
    with col_u3:
        st.caption("Loads the profile into session_state so Step 4 can proceed without any campaign-intent matching.")

    if clear_campaign_profile_btn:
        for k in ["campaign_intent_profile_df", "campaign_intent_profiles_json"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("✅ Cleared campaign intent profile from session_state.")

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
                        profiles_json.append({
                            "campaign_id": str(cid),
                            "campaign_name": cname,
                            "top_intents": top_intents
                        })
                    st.session_state["campaign_intent_profiles_json"] = profiles_json

                    st.success(f"✅ Loaded uploaded campaign intent profile: {len(df_up):,} rows")

                    with st.expander("Preview uploaded profile (first 200 rows)", expanded=False):
                        st.dataframe(df_up.head(200), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Failed to load uploaded campaign intent profile: {e}")

st.divider()

# -------------------------
# Option A: Generate with LLM (only if ontology exists)
# -------------------------
st.subheader("🤖 Generate Campaign Intent Profiles with AI (Gemini)")

if "ontology" not in st.session_state or "dim_intent_df" not in st.session_state:
    st.info("👆 To generate profiles with AI, please generate/upload ontology in Step 2 first. (Upload reuse above still works.)")
else:
    dim_intent_df = st.session_state["dim_intent_df"].copy()

    # ✅ Auto-hide default, only what you asked: avg chars per campaign + # campaigns
    with st.expander("📏 Campaign Size Advisor (Step 3)", expanded=False):
        briefs = campaigns_df["campaign_brief"].fillna("").astype(str)
        avg_chars_per_campaign = float(briefs.str.len().mean()) if len(briefs) else 0.0
        n_campaigns = int(len(campaigns_df))
        c1, c2 = st.columns(2)
        c1.metric("Avg characters / campaign brief", f"{avg_chars_per_campaign:,.0f}")
        c2.metric("# campaigns", f"{n_campaigns:,}")

    st.write("**🔑 API Configuration**")
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
        st.warning("⚠️ Provide API key only if you want to generate. (Or upload reuse above.)")

    left, right = st.columns([2, 1])

    with left:
        st.write("**Configuration**")
        top_n = st.number_input("Top intents per campaign", min_value=3, max_value=12, value=6, key="step3_top_n")
        output_language = st.selectbox("Output Language", ["en", "th"], index=0, key="step3_output_language")
        include_lifestyle_context = st.checkbox("Include lifestyle context in prompt", value=True, key="step3_include_lifestyle")

        # ✅ parameter you requested
        intent_snippet_max_chars = st.number_input(
            "Intent list max chars in prompt (intent_snippet_max_chars)",
            min_value=2000,
            max_value=120000,
            value=16000,
            step=1000,
            help="Caps how many intents go into the prompt. Smaller = cheaper but may miss some intents."
        )

    with right:
        st.write("**Rate-limit settings (avoid 429)**")
        model_name = st.text_input("Model name", value="gemini-2.5-flash", key="step3_model_name")
        rpm_limit = st.number_input("Max requests per minute (RPM)", min_value=1, max_value=60, value=8, key="step3_rpm_limit")
        batch_size = st.number_input("Campaigns per request (batch size)", min_value=1, max_value=10, value=3, key="step3_batch_size")
        max_retries = st.number_input("Max retries on 429", min_value=0, max_value=10, value=4, key="step3_max_retries")

        gen_campaign_profile_btn = st.button(
            "🤖 Generate Campaign Intent Profiles",
            type="primary",
            use_container_width=True,
            key="step3_generate_btn"
        )

    # intent candidates
    intent_candidates = []
    for _, r in dim_intent_df.iterrows():
        intent_candidates.append({
            "intent_id": str(r.get("intent_id", "")).strip(),
            "intent_name": str(r.get("intent_name", "")).strip(),
            "definition": str(r.get("definition", "")).strip(),
            "lifestyle_id": str(r.get("lifestyle_id", "")).strip(),
        })

    # ✅ use your parameter here (not hard-coded)
    intent_snippet = safe_json_snippet(intent_candidates, max_chars=int(intent_snippet_max_chars))

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

    if "campaign_intent_profile_df" not in st.session_state:
        st.session_state["campaign_intent_profile_df"] = None
    if "campaign_intent_profiles_json" not in st.session_state:
        st.session_state["campaign_intent_profiles_json"] = None

    if gen_campaign_profile_btn:
        if not gemini_api_key:
            st.error("Please provide a Gemini API key to generate (or use upload reuse above).")
        else:
            try:
                model = make_gemini_model(gemini_api_key, model_name, json_mode=True, temperature=0.2)

                existing_rows = []
                existing_profiles = []
                if st.session_state["campaign_intent_profile_df"] is not None:
                    existing_rows = st.session_state["campaign_intent_profile_df"].to_dict("records")
                if st.session_state["campaign_intent_profiles_json"] is not None:
                    existing_profiles = list(st.session_state["campaign_intent_profiles_json"])

                already_done = set([p["campaign_id"] for p in existing_profiles if "campaign_id" in p])

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
1) Select the TOP {int(top_n)} intents most relevant to the campaign brief.
2) For each selected intent, provide:
   - intent_id
   - intent_name
   - rationale (1–2 sentences, grounded in the brief)
   - weight (positive number; does NOT need to sum to 1 yet)
3) Output language for rationales: {output_language}

Rules:
- Return JSON ONLY. No markdown. No extra text.
- Do NOT use double quotes inside rationale strings. Use parentheses or single quotes if needed.

Return STRICT minified JSON only:
{{"campaign_profiles":[{{"campaign_id":"...","campaign_name":"...","top_intents":[{{"intent_id":"IN_...","intent_name":"...","weight":0.0,"rationale":"..."}}]}}]}}
""".strip()

                            raw = call_llm_with_retry(
                                model=model,
                                prompt=prompt,
                                rpm_limit=int(rpm_limit),
                                max_retries=int(max_retries),
                                last_ts_key="_last_llm_call_ts_step3"
                            )

                            try:
                                data = extract_json_from_text_robust(raw)
                            except json.JSONDecodeError as e:
                                st.error("❌ Invalid JSON returned by the model (Step 3).")
                                st.code(debug_json_error(raw, e))
                                st.code((raw or "")[:1200])
                                st.stop()

                            campaign_profiles = data.get("campaign_profiles", []) or []

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

# -------------------------
# Output preview/download (works for uploaded OR generated)
# -------------------------
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
    with st.expander("Preview JSON (collapsed)", expanded=False):
        st.json(st.session_state["campaign_intent_profiles_json"][:50])
    st.caption("Showing first 50 campaigns in JSON preview.")

    st.download_button(
        "📥 Download campaign_intent_profiles.json",
        data=json.dumps(st.session_state["campaign_intent_profiles_json"], ensure_ascii=False, indent=2),
        file_name="campaign_intent_profiles.json",
        mime="application/json",
        use_container_width=True
    )


# NOTE TO STEP 4:
# Step 4 does not need to match campaign → intents.
# The campaign intent profile exists only as an optional marketing artifact / audit input for later modules.
# Product labeling runs directly against the ontology intent list.

# ============================================================================
# STEP 4: PRODUCT → INTENT LABELING (UPLOAD OR LLM) — CONSISTENT IDs WITH STEP 2
# ============================================================================
st.divider()
st.header("Step 4: Product → Intent Labeling")
st.caption("Mode A: Upload existing labels • Mode B: Run Gemini labeling (JSON-robust)")

if "catalog_df" not in st.session_state:
    st.info("👆 Upload Product CSV in Step 1 first.")
    st.stop()

catalog_df = st.session_state["catalog_df"].copy()

if "product_intent_labels_df" not in st.session_state:
    st.session_state["product_intent_labels_df"] = None
if "product_intent_labels_json" not in st.session_state:
    st.session_state["product_intent_labels_json"] = None

mode = st.radio(
    "Choose labeling mode",
    ["Upload labeled CSV (Mode A)", "Run Gemini labeling (Mode B)"],
    horizontal=True,
    key="step4_mode"
)

# -------------------------
# MODE A: Upload labels CSV
# -------------------------
if mode == "Upload labeled CSV (Mode A)":
    st.subheader("📤 Upload product intent labels CSV")
    st.caption(
        "Accepted format (long): product_id, intent_id (optional: score, intent_name, evidence, reason)\n"
        "Important: intent_id MUST match the ontology in Step 2, or Step 5 will not work."
    )

    up = st.file_uploader("Upload product_intent_labels.csv", type=["csv"], key="step4_upload_labels")
    load_btn = st.button("Load Uploaded Labels", type="primary", key="step4_load_uploaded_labels")

    if load_btn:
        if up is None:
            st.error("Please upload a CSV first.")
        else:
            try:
                df = pd.read_csv(up)
                df.columns = df.columns.str.strip().str.lower()

                required = {"product_id", "intent_id"}
                missing = required - set(df.columns)
                if missing:
                    st.error(f"❌ Missing columns: {sorted(list(missing))}. Required at least: product_id, intent_id")
                    st.info(f"Found columns: {df.columns.tolist()}")
                    st.stop()

                # Normalize
                df["product_id"] = df["product_id"].astype(str).str.strip()
                df["intent_id"] = df["intent_id"].astype(str).str.strip()

                if "score" not in df.columns:
                    df["score"] = 1.0
                df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0).clip(0, 1)

                # Attach product_name if missing
                if "product_name" not in df.columns:
                    df = df.merge(
                        catalog_df[["product_id", "product_name"]],
                        on="product_id",
                        how="left"
                    )
                    df["product_name"] = df["product_name"].fillna("")

                # If ontology exists, enforce that uploaded IDs are valid (optional but strongly recommended)
                if "dim_intent_df" in st.session_state:
                    dim_intent_df = st.session_state["dim_intent_df"].copy()
                    dim_intent_df.columns = dim_intent_df.columns.str.strip().str.lower()
                    dim_intent_df["intent_id"] = dim_intent_df["intent_id"].astype(str).str.strip()
                    dim_intent_df["intent_name"] = dim_intent_df["intent_name"].astype(str).str.strip()

                    valid_ids = set(dim_intent_df["intent_id"].tolist())
                    bad = df.loc[~df["intent_id"].isin(valid_ids), "intent_id"].drop_duplicates().head(30).tolist()

                    if len(bad) > 0:
                        st.error("❌ Uploaded labels contain intent_id values not found in Step 2 ontology.")
                        st.write("Examples of unknown intent_id:", bad)
                        st.info("Fix: upload labels built from the same ontology OR rerun Step 4 Mode B after Step 2.")
                        st.stop()

                    # Attach canonical intent_name
                    id_to_name = dict(zip(dim_intent_df["intent_id"], dim_intent_df["intent_name"]))
                    df["intent_name"] = df.get("intent_name", "").astype(str)
                    df["intent_name"] = df["intent_id"].map(id_to_name).fillna(df["intent_name"]).fillna("")

                # Fill optional fields
                if "evidence" not in df.columns:
                    df["evidence"] = ""
                if "reason" not in df.columns:
                    df["reason"] = ""

                # Create rank per product by score desc
                df = df.sort_values(["product_id", "score"], ascending=[True, False]).copy()
                df["rank"] = df.groupby("product_id").cumcount() + 1

                # Add metadata columns
                ontology_version = "v1"
                if "ontology" in st.session_state:
                    ontology_version = str(st.session_state["ontology"].get("version", "v1"))

                df["ontology_version"] = df.get("ontology_version", ontology_version)
                df["labeling_run_id"] = df.get(
                    "labeling_run_id",
                    f"upload_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                )
                df["model"] = df.get("model", "uploaded")
                df["created_at"] = df.get("created_at", pd.Timestamp.now().isoformat())

                # Build JSON structure per product
                labels_json = []
                for pid, g in df.sort_values(["product_id", "rank"]).groupby("product_id"):
                    pname = ""
                    if "product_name" in g.columns and g["product_name"].notna().any():
                        pname = str(g["product_name"].dropna().iloc[0])

                    top_intents = []
                    for _, r in g.iterrows():
                        top_intents.append({
                            "intent_id": str(r.get("intent_id", "")),
                            "intent_name": str(r.get("intent_name", "")),
                            "score": float(r.get("score", 0.0)),
                            "evidence": str(r.get("evidence", "")),
                            "reason": str(r.get("reason", "")),
                        })

                    labels_json.append({
                        "product_id": str(pid),
                        "product_name": pname,
                        "top_intents": top_intents,
                        "ontology_version": ontology_version,
                        "labeling_run_id": str(g.get("labeling_run_id", "").iloc[0]) if "labeling_run_id" in g.columns else "",
                        "model": str(g.get("model", "").iloc[0]) if "model" in g.columns else "uploaded",
                        "created_at": str(g.get("created_at", "").iloc[0]) if "created_at" in g.columns else pd.Timestamp.now().isoformat(),
                    })

                st.session_state["product_intent_labels_df"] = df
                st.session_state["product_intent_labels_json"] = labels_json

                st.success(f"✅ Loaded uploaded labels: {len(df):,} rows, {df['product_id'].nunique():,} products")
                with st.expander("Preview labels (first 200 rows)", expanded=False):
                    st.dataframe(df.head(200), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Failed to load uploaded labels: {e}")
                st.exception(e)

# -------------------------
# MODE B: Run Gemini labeling (ENFORCE CONSISTENT IDs)
# -------------------------
else:
    # Preconditions
    if "dim_intent_df" not in st.session_state or "ontology" not in st.session_state:
        st.info("👆 Generate or upload the ontology in Step 2 first (needed for LLM labeling).")
        st.stop()

    dim_intent_df = st.session_state["dim_intent_df"].copy()
    dim_intent_df.columns = dim_intent_df.columns.str.strip().str.lower()
    ontology = st.session_state["ontology"]
    ontology_version = str(ontology.get("version", "v1"))

    # --- Canonical lookups (from Step 2) ---
    dim_intent_df["intent_id"] = dim_intent_df["intent_id"].astype(str).str.strip()
    dim_intent_df["intent_name"] = dim_intent_df["intent_name"].astype(str).str.strip()

    canon_id_set = set(dim_intent_df["intent_id"].tolist())
    canon_id_to_name = dict(zip(dim_intent_df["intent_id"], dim_intent_df["intent_name"]))

    def _norm_name(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^a-z0-9ก-๙\s\-_]", "", s)
        return s

    canon_name_to_id = {}
    for _, r in dim_intent_df.iterrows():
        canon_name_to_id[_norm_name(r["intent_name"])] = r["intent_id"]

    def canonicalize_intent(iid: str, iname: str):
        """Return (canonical_intent_id, canonical_intent_name) or (None, None) if cannot map."""
        iid = str(iid or "").strip()
        iname = str(iname or "").strip()

        # If ID already valid, accept and canonicalize name
        if iid and iid in canon_id_set:
            return iid, canon_id_to_name.get(iid, iname)

        # Fallback: map by intent_name
        key = _norm_name(iname)
        if key and key in canon_name_to_id:
            cid = canon_name_to_id[key]
            return cid, canon_id_to_name.get(cid, iname)

        return None, None

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
        min_score = st.slider("Minimum score threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05, key="step4_minscore")
        product_text_chars = st.number_input("Max characters of product_text sent to LLM", min_value=120, max_value=2000, value=600, step=60, key="step4_textchars")

    with c2:
        st.subheader("Rate-limit settings (avoid 429)")
        model_name = st.text_input("Model name", value="gemini-2.5-flash", key="step4_model_name")
        rpm_limit = st.number_input("Max requests per minute (RPM)", min_value=1, max_value=60, value=8, key="step4_rpm_limit")
        products_per_request = st.number_input("Products per request (batch size)", min_value=1, max_value=12, value=4, key="step4_products_per_request")
        max_retries = st.number_input("Max retries on 429", min_value=0, max_value=10, value=4, key="step4_max_retries")

    # Optional: Keep prompt small (reuse Step 3 param if exists)
    intent_snippet_max_chars = int(st.session_state.get("step3_intent_snippet_max_chars", 16000)) if "step3_intent_snippet_max_chars" in st.session_state else 16000

    label_btn = st.button("🏷️ Run Product → Intent Labeling", type="primary", use_container_width=True, key="step4_run_btn")

    # Prepare concise intent list (canonical)
    intent_candidates = []
    for _, r in dim_intent_df.iterrows():
        intent_candidates.append({
            "intent_id": str(r.get("intent_id", "")).strip(),
            "intent_name": str(r.get("intent_name", "")).strip(),
            "definition": str(r.get("definition", "")).strip(),
        })
    intent_snippet = safe_json_snippet(intent_candidates, max_chars=int(intent_snippet_max_chars))

    if label_btn:
        try:
            model = make_gemini_model(gemini_api_key, model_name, json_mode=True, temperature=0.2)

            existing_rows = []
            existing_json = []
            if st.session_state.get("product_intent_labels_df") is not None:
                existing_rows = st.session_state["product_intent_labels_df"].to_dict("records")
            if st.session_state.get("product_intent_labels_json") is not None:
                existing_json = list(st.session_state["product_intent_labels_json"])

            already_done = set([x.get("product_id") for x in existing_json if x.get("product_id")])
            todo_df = catalog_df[~catalog_df["product_id"].astype(str).isin(already_done)].reset_index(drop=True)

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
                            ptext = str(r.get("product_text", "") or "")[:int(product_text_chars)]
                            batch_products.append({
                                "product_id": pid,
                                "product_name": pname,
                                "product_text": ptext
                            })

                        prompt = f"""
You are labeling retail products into a fixed Intent Ontology for marketing.

You MUST choose intents ONLY from this canonical list (do not invent new intents):
{intent_snippet}

Input products (JSON):
{json.dumps(batch_products, ensure_ascii=False)}

Task for EACH product:
1) Select TOP {int(top_k_intents)} intents most relevant to the product.
2) For each selected intent, output:
   - intent_id (MUST come from the canonical list)
   - intent_name (MUST match the canonical list)
   - score (0.0 to 1.0, higher = more relevant)
   - evidence (short phrase copied or paraphrased from product text)
   - reason (1 short sentence)
3) Return JSON ONLY. No markdown. No extra text. No trailing commas.
   IMPORTANT: Do NOT put double quotes (") inside evidence/reason strings.
              Use parentheses or single quotes instead.

Return STRICT minified JSON:
{{"labels":[{{"product_id":"...","product_name":"...","top_intents":[{{"intent_id":"IN_...","intent_name":"...","score":0.0,"evidence":"...","reason":"..."}}]}}]}}
""".strip()

                        raw = call_llm_with_retry(
                            model=model,
                            prompt=prompt,
                            rpm_limit=int(rpm_limit),
                            max_retries=int(max_retries),
                            last_ts_key="_last_llm_call_ts_step4"
                        )

                        try:
                            data = extract_json_from_text_robust(raw)
                        except json.JSONDecodeError as e:
                            st.error("❌ Invalid JSON returned by the model (Step 4).")
                            st.code(debug_json_error(raw, e))
                            st.write("Raw model output (first 1200 chars):")
                            st.code((raw or "")[:1200])
                            st.stop()

                        batch_labels = data.get("labels", []) or []

                        for item in batch_labels:
                            pid = str(item.get("product_id", "")).strip()
                            pname = str(item.get("product_name", "")).strip()
                            intents = item.get("top_intents", []) or []

                            cleaned = []
                            for it in intents:
                                iid_raw = str(it.get("intent_id", "")).strip()
                                iname_raw = str(it.get("intent_name", "")).strip()
                                score = clamp01(it.get("score", 0.0))
                                evidence = str(it.get("evidence", "")).strip()
                                reason = str(it.get("reason", "")).strip()

                                # ✅ Canonicalize to Step 2 IDs
                                cid, cname = canonicalize_intent(iid_raw, iname_raw)
                                if cid is None:
                                    continue  # drop non-mappable intents

                                cleaned.append({
                                    "intent_id": cid,
                                    "intent_name": cname,
                                    "score": score,
                                    "evidence": evidence,
                                    "reason": reason
                                })

                            # De-dup by intent_id, keep best score
                            best = {}
                            for x in cleaned:
                                if (x["intent_id"] not in best) or (x["score"] > best[x["intent_id"]]["score"]):
                                    best[x["intent_id"]] = x
                            cleaned = list(best.values())

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

                            for rank, it2 in enumerate(cleaned, start=1):
                                results_rows.append({
                                    "product_id": pid,
                                    "product_name": pname,
                                    "rank": rank,
                                    "intent_id": it2["intent_id"],
                                    "intent_name": it2["intent_name"],
                                    "score": it2["score"],
                                    "evidence": it2["evidence"],
                                    "reason": it2["reason"],
                                    "ontology_version": ontology_version,
                                    "labeling_run_id": labeling_run_id,
                                    "model": model_name,
                                    "created_at": created_at
                                })

                        # persist per batch
                        st.session_state["product_intent_labels_df"] = pd.DataFrame(results_rows)
                        st.session_state["product_intent_labels_json"] = labels_json

                        progress.progress((b_idx + 1) / max(total_batches, 1))

                st.success("✅ Product → Intent labeling completed (canonical IDs enforced).")

            st.session_state["product_intent_labels_df"] = pd.DataFrame(results_rows)
            st.session_state["product_intent_labels_json"] = labels_json

            # Optional: show health check
            with st.expander("✅ Step 4 Health Check (ID overlap)", expanded=False):
                lbl = st.session_state["product_intent_labels_df"].copy()
                lbl["intent_id"] = lbl["intent_id"].astype(str).str.strip()
                overlap = lbl["intent_id"].isin(dim_intent_df["intent_id"]).mean() * 100.0
                st.write(f"Intent ID overlap vs Step 2 ontology: {overlap:.1f}% (should be ~100%)")

        except ImportError:
            st.error("❌ Missing library: google-generativeai. Please add it to requirements.txt")
        except Exception as e:
            st.error(f"❌ Error labeling products: {str(e)}")
            st.exception(e)

# -------------------------
# Step 4 Outputs
# -------------------------
if st.session_state.get("product_intent_labels_df") is not None:
    st.subheader("🏷️ Product → Intent Labels (Preview)")
    labels_df = st.session_state["product_intent_labels_df"].copy()
    st.dataframe(labels_df, use_container_width=True, height=420)

    st.download_button(
        "📥 Download product_intent_labels.csv",
        data=labels_df.to_csv(index=False).encode("utf-8"),
        file_name="product_intent_labels.csv",
        mime="text/csv",
        use_container_width=True
    )

if st.session_state.get("product_intent_labels_json") is not None:
    st.subheader("📦 Product Intent Labels (JSON)")
    with st.expander("Preview JSON (collapsed)", expanded=False):
        st.json(st.session_state["product_intent_labels_json"][:25])
    st.caption("Showing first 25 products in JSON preview.")

    st.download_button(
        "📥 Download product_intent_labels.json",
        data=json.dumps(st.session_state["product_intent_labels_json"], ensure_ascii=False, indent=2),
        file_name="product_intent_labels.json",
        mime="application/json",
        use_container_width=True
    )

# st.write("labels intents:", labels_df["intent_id"].nunique())
# st.write("ontology intents:", dim_intent_df["intent_id"].nunique())
# st.write("overlap intents:", labels_df["intent_id"].isin(dim_intent_df["intent_id"]).sum())

# ============================================================================
# STEP 5: CUSTOMER INTENT / LIFESTYLE PROFILE BUILDER  (FULL REPLACE)
# ============================================================================
st.divider()
st.header("Step 5: Customer Lifestyle Profile Builder")
st.caption("Build customer_intent_profile and customer_lifestyle_profile from transactions + product intent labels.")

# -------------------------
# Preconditions
# -------------------------
if "txn_df" not in st.session_state:
    st.info("👆 Upload Transaction CSV in Step 1 first.")
    st.stop()

if st.session_state.get("product_intent_labels_df") is None:
    st.info("👆 Complete Step 4 (Product → Intent labeling) first.")
    st.stop()

txn_df = st.session_state["txn_df"].copy()
labels_df = st.session_state["product_intent_labels_df"].copy()

# Normalize columns
txn_df.columns = txn_df.columns.str.strip().str.lower()
labels_df.columns = labels_df.columns.str.strip().str.lower()

# Basic required checks
req_tx = {"customer_id", "product_id"}
if not req_tx.issubset(set(txn_df.columns)):
    st.error(f"Transaction table missing required columns: {sorted(list(req_tx - set(txn_df.columns)))}")
    st.stop()

req_lb = {"product_id", "intent_id"}
if not req_lb.issubset(set(labels_df.columns)):
    st.error(f"Product labels missing required columns: {sorted(list(req_lb - set(labels_df.columns)))}")
    st.stop()

# Ensure types
txn_df["customer_id"] = txn_df["customer_id"].astype(str).str.strip()
txn_df["product_id"] = txn_df["product_id"].astype(str).str.strip()

labels_df["product_id"] = labels_df["product_id"].astype(str).str.strip()
labels_df["intent_id"] = labels_df["intent_id"].astype(str).str.strip()

# Ensure score exists
if "score" not in labels_df.columns:
    labels_df["score"] = 1.0
labels_df["score"] = pd.to_numeric(labels_df["score"], errors="coerce").fillna(0.0).clip(0, 1)

# Optional: make sure amt exists if user wants amt mode
if "amt" not in txn_df.columns:
    # Try reconstruct from qty * price if possible; otherwise keep 1.0 fallback later
    if "qty" in txn_df.columns and "price" in txn_df.columns:
        txn_df["qty"] = pd.to_numeric(txn_df["qty"], errors="coerce").fillna(0.0)
        txn_df["price"] = pd.to_numeric(txn_df["price"], errors="coerce").fillna(0.0)
        txn_df["amt"] = txn_df["qty"] * txn_df["price"]
    else:
        txn_df["amt"] = 0.0

# Ontology mapping (intent -> lifestyle)
has_ontology = ("dim_intent_df" in st.session_state)
dim_intent_df = None
dim_lifestyle_df = None

if has_ontology:
    dim_intent_df = st.session_state["dim_intent_df"].copy()
    dim_intent_df.columns = dim_intent_df.columns.str.strip().str.lower()
    if "intent_id" in dim_intent_df.columns:
        dim_intent_df["intent_id"] = dim_intent_df["intent_id"].astype(str).str.strip()

    if "dim_lifestyle_df" in st.session_state:
        dim_lifestyle_df = st.session_state["dim_lifestyle_df"].copy()
        dim_lifestyle_df.columns = dim_lifestyle_df.columns.str.strip().str.lower()
        if "lifestyle_id" in dim_lifestyle_df.columns:
            dim_lifestyle_df["lifestyle_id"] = dim_lifestyle_df["lifestyle_id"].astype(str).str.strip()

# Session slots
if "customer_intent_profile_df" not in st.session_state:
    st.session_state["customer_intent_profile_df"] = None
if "customer_lifestyle_profile_df" not in st.session_state:
    st.session_state["customer_lifestyle_profile_df"] = None

# -------------------------
# Controls
# -------------------------
c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
with c1:
    intent_weight_mode = st.selectbox(
        "Transaction contribution mode",
        ["amt * label_score", "qty * label_score", "label_score only"],
        index=0,
        key="step5_weight_mode"
    )
with c2:
    normalize_customer = st.checkbox(
        "Normalize per customer (sum intent share = 1)",
        value=True,
        key="step5_norm_customer"
    )
with c3:
    topn_keep = st.number_input(
        "Keep top-N intents per customer (after scoring)",
        min_value=5, max_value=80, value=20,
        key="step5_topn_keep"
    )

build_profiles_btn = st.button(
    "🧩 Build Customer Profiles",
    type="primary",
    use_container_width=True,
    key="step5_build_btn"
)

def _compute_base(row, mode: str) -> float:
    if mode == "amt * label_score":
        return float(row.get("amt", 0.0) or 0.0)
    if mode == "qty * label_score":
        return float(row.get("qty", 0.0) or 0.0)
    return 1.0

def _get_lifestyle_name_map_from_ontology():
    """Fallback if dim_lifestyle_df missing lifestyle_name"""
    ontology = st.session_state.get("ontology", {}) or {}
    rows = []
    for ls in (ontology.get("lifestyles", []) or []):
        rows.append({
            "lifestyle_id": str(ls.get("lifestyle_id", "")).strip(),
            "lifestyle_name": str(ls.get("lifestyle_name", "")).strip(),
        })
    df = pd.DataFrame(rows).drop_duplicates()
    if len(df) == 0:
        return None
    df.loc[df["lifestyle_name"] == "", "lifestyle_name"] = "Unknown"
    return df

# -------------------------
# Build
# -------------------------
if build_profiles_btn:
    try:
        # Join txn -> labels on product_id
        merged = txn_df.merge(
            labels_df[["product_id", "intent_id", "score"]],
            on="product_id",
            how="left"
        )

        # Drop rows with no intent_id (no labels for those products)
        merged["intent_id"] = merged["intent_id"].fillna("").astype(str).str.strip()
        merged = merged[merged["intent_id"] != ""].copy()

        if len(merged) == 0:
            st.error("❌ No transaction rows matched any product intent labels. Check Step 4 output and product_id formats.")
            st.stop()

        merged["base"] = merged.apply(lambda r: _compute_base(r, intent_weight_mode), axis=1)
        merged["intent_points"] = merged["base"] * pd.to_numeric(merged["score"], errors="coerce").fillna(0.0).clip(0, 1)

        # Aggregate customer-intent points
        cust_int = (
            merged.groupby(["customer_id", "intent_id"], as_index=False)["intent_points"]
            .sum()
            .rename(columns={"intent_points": "intent_points_raw"})
        )

        # Attach intent_name (prefer ontology dim_intent_df if available)
        cust_int["intent_name"] = ""

        if has_ontology and dim_intent_df is not None and "intent_id" in dim_intent_df.columns:
            cols = ["intent_id"]
            if "intent_name" in dim_intent_df.columns:
                cols.append("intent_name")
            dim_i = dim_intent_df[cols].drop_duplicates()
            if "intent_name" not in dim_i.columns:
                dim_i["intent_name"] = ""
            cust_int = cust_int.merge(dim_i, on="intent_id", how="left", suffixes=("", "_dim"))
            # coalesce
            cust_int["intent_name"] = cust_int["intent_name_dim"].fillna(cust_int["intent_name"]).fillna("")
            if "intent_name_dim" in cust_int.columns:
                cust_int.drop(columns=["intent_name_dim"], inplace=True)

        # Normalize per customer
        if normalize_customer:
            sums = cust_int.groupby("customer_id")["intent_points_raw"].sum().reset_index(name="cust_sum")
            cust_int = cust_int.merge(sums, on="customer_id", how="left")
            cust_int["intent_share"] = cust_int.apply(
                lambda r: float(r["intent_points_raw"]) / float(r["cust_sum"]) if float(r.get("cust_sum", 0) or 0) > 0 else 0.0,
                axis=1
            )
        else:
            cust_int["intent_share"] = cust_int["intent_points_raw"]

        # Keep top-N intents per customer
        cust_int = cust_int.sort_values(["customer_id", "intent_share"], ascending=[True, False])
        cust_int["rank"] = cust_int.groupby("customer_id").cumcount() + 1
        cust_int_top = cust_int[cust_int["rank"] <= int(topn_keep)].copy()

        st.session_state["customer_intent_profile_df"] = cust_int_top

        # -------------------------
        # Lifestyle aggregation (requires ontology intent -> lifestyle mapping)
        # -------------------------
        cust_ls_agg = None

        if has_ontology and dim_intent_df is not None and "lifestyle_id" in dim_intent_df.columns and "intent_id" in dim_intent_df.columns:
            map_cols = ["intent_id", "lifestyle_id"]
            if "lifestyle_name" in dim_intent_df.columns:
                map_cols.append("lifestyle_name")
            if "intent_name" in dim_intent_df.columns:
                map_cols.append("intent_name")

            map_df = dim_intent_df[map_cols].drop_duplicates().copy()
            map_df["intent_id"] = map_df["intent_id"].astype(str).str.strip()
            map_df["lifestyle_id"] = map_df["lifestyle_id"].fillna("").astype(str).str.strip()

            cust_ls = cust_int_top.merge(map_df, on="intent_id", how="left", suffixes=("", "_dim"))

            # Ensure lifestyle_name exists by joining dim_lifestyle_df or ontology
            if "lifestyle_name" not in cust_ls.columns:
                cust_ls["lifestyle_name"] = ""

            need_name = cust_ls["lifestyle_name"].isna() | (cust_ls["lifestyle_name"].astype(str).str.strip() == "")
            if need_name.any():
                # Try dim_lifestyle_df first
                if dim_lifestyle_df is not None and "lifestyle_name" in dim_lifestyle_df.columns and "lifestyle_id" in dim_lifestyle_df.columns:
                    cust_ls = cust_ls.merge(
                        dim_lifestyle_df[["lifestyle_id", "lifestyle_name"]].drop_duplicates(),
                        on="lifestyle_id",
                        how="left",
                        suffixes=("", "_ls")
                    )
                    cust_ls["lifestyle_name"] = cust_ls["lifestyle_name"].fillna(cust_ls.get("lifestyle_name_ls", ""))
                    if "lifestyle_name_ls" in cust_ls.columns:
                        cust_ls.drop(columns=["lifestyle_name_ls"], inplace=True)
                else:
                    # Fallback to ontology JSON
                    fallback_map = _get_lifestyle_name_map_from_ontology()
                    if fallback_map is not None:
                        cust_ls = cust_ls.merge(fallback_map, on="lifestyle_id", how="left", suffixes=("", "_onto"))
                        cust_ls["lifestyle_name"] = cust_ls["lifestyle_name"].fillna(cust_ls.get("lifestyle_name_onto", ""))
                        if "lifestyle_name_onto" in cust_ls.columns:
                            cust_ls.drop(columns=["lifestyle_name_onto"], inplace=True)

            # Clean
            cust_ls["lifestyle_id"] = cust_ls["lifestyle_id"].fillna("").astype(str).str.strip()
            cust_ls["lifestyle_name"] = cust_ls["lifestyle_name"].fillna("Unknown").astype(str).str.strip()
            cust_ls.loc[cust_ls["lifestyle_name"] == "", "lifestyle_name"] = "Unknown"

            # Diagnostics: mapping quality
            missing_ls = (cust_ls["lifestyle_id"].fillna("").astype(str).str.strip() == "")
            pct_missing = float(missing_ls.mean() * 100.0) if len(cust_ls) else 0.0

            if pct_missing >= 1.0:
                st.warning(f"⚠️ {pct_missing:.1f}% of intent rows could not be mapped to a lifestyle (intent_id mismatch or missing in dim_intent_df).")
                sample_missing = cust_ls.loc[missing_ls, "intent_id"].dropna().astype(str).unique().tolist()[:20]
                if sample_missing:
                    st.caption("Sample missing intent_ids (up to 20):")
                    st.code(sample_missing)

            # Aggregate to customer-lifestyle
            cust_ls_agg = (
                cust_ls[~missing_ls].groupby(["customer_id", "lifestyle_id", "lifestyle_name"], as_index=False)["intent_share"]
                .sum()
                .rename(columns={"intent_share": "lifestyle_share"})
            )

            cust_ls_agg = cust_ls_agg.sort_values(["customer_id", "lifestyle_share"], ascending=[True, False])
            cust_ls_agg["rank"] = cust_ls_agg.groupby("customer_id").cumcount() + 1

            st.session_state["customer_lifestyle_profile_df"] = cust_ls_agg
        else:
            st.session_state["customer_lifestyle_profile_df"] = None
            st.warning("⚠️ Lifestyle profile not built because ontology mapping (dim_intent_df with lifestyle_id) is missing.")

        st.success("✅ Customer profiles built successfully.")

    except Exception as e:
        st.error(f"❌ Failed to build customer profiles: {e}")
        st.exception(e)

# ============================================================================
# Step 5 Visual: Treemap drilldown (Lifestyle -> Intent) 
# ============================================================================
st.subheader("🧩 Lifestyle → Intent Treemap (click to drill down)")
st.caption("One rectangle = one node • size = # customers (distinct). Click a lifestyle to zoom into its intents.")

df_ci = st.session_state.get("customer_intent_profile_df")

if df_ci is None or len(df_ci) == 0:
    st.info("No customer intent profile yet. Click **Build Customer Profiles** above.")
else:
    if (
        not has_ontology
        or dim_intent_df is None
        or "lifestyle_id" not in (dim_intent_df.columns if dim_intent_df is not None else [])
        or "intent_id" not in (dim_intent_df.columns if dim_intent_df is not None else [])
    ):
        st.info("Treemap drilldown needs ontology mapping (dim_intent_df with intent_id + lifestyle_id). Generate/upload ontology in Step 2.")
    else:
        import plotly.express as px

        # -------------------------
        # Build mapping tables
        # -------------------------
        dim_i = dim_intent_df.copy()
        dim_i.columns = dim_i.columns.str.strip().str.lower()

        # Required fields
        dim_i["intent_id"] = dim_i["intent_id"].astype(str).str.strip()
        dim_i["lifestyle_id"] = dim_i["lifestyle_id"].fillna("").astype(str).str.strip()

        # Prefer human-readable names
        if "intent_name" not in dim_i.columns:
            dim_i["intent_name"] = ""
        dim_i["intent_name"] = dim_i["intent_name"].fillna("").astype(str).str.strip()

        # Small mapping table
        dim_i_small = dim_i[["intent_id", "intent_name", "lifestyle_id"]].drop_duplicates().copy()

        # Lifestyle name map (prefer dim_lifestyle_df then ontology JSON)
        ls_map = None
        if (
            "dim_lifestyle_df" in st.session_state
            and st.session_state["dim_lifestyle_df"] is not None
        ):
            dim_lifestyle_df = st.session_state["dim_lifestyle_df"].copy()
            dim_lifestyle_df.columns = dim_lifestyle_df.columns.str.strip().str.lower()

            if "lifestyle_id" in dim_lifestyle_df.columns and "lifestyle_name" in dim_lifestyle_df.columns:
                ls_map = dim_lifestyle_df[["lifestyle_id", "lifestyle_name"]].drop_duplicates().copy()
                ls_map["lifestyle_id"] = ls_map["lifestyle_id"].astype(str).str.strip()
                ls_map["lifestyle_name"] = ls_map["lifestyle_name"].fillna("Unknown").astype(str).str.strip()

        if ls_map is None or len(ls_map) == 0:
            # Fallback: build from ontology JSON (helper defined earlier in Step 5)
            ls_map = _get_lifestyle_name_map_from_ontology()

        # -------------------------
        # Start from customer intent profile
        # -------------------------
        ci = df_ci.copy()
        ci.columns = ci.columns.str.strip().str.lower()

        ci["customer_id"] = ci["customer_id"].astype(str).str.strip()
        ci["intent_id"] = ci["intent_id"].astype(str).str.strip()

        if "intent_share" not in ci.columns:
            ci["intent_share"] = 0.0
        ci["intent_share"] = pd.to_numeric(ci["intent_share"], errors="coerce").fillna(0.0)

        # -------------------------
        # Controls
        # -------------------------
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            treemap_rank_threshold = st.selectbox(
                "Include intent if customer rank ≤",
                options=[3, 5, 10, 20, 999],
                index=0,
                key="step5_treemap_rank_threshold"
            )
        with cc2:
            min_customers_node = st.number_input(
                "Hide nodes with < customers",
                min_value=1, value=5, step=1,
                key="step5_treemap_min_customers"
            )

        if "rank" in ci.columns and int(treemap_rank_threshold) != 999:
            ci = ci[ci["rank"].fillna(999).astype(int) <= int(treemap_rank_threshold)].copy()

        # -------------------------
        # Attach lifestyle + intent names
        # -------------------------
        j = ci.merge(dim_i_small, on="intent_id", how="left")

        # Clean lifestyle_id
        if "lifestyle_id" not in j.columns:
            j["lifestyle_id"] = ""
        j["lifestyle_id"] = j["lifestyle_id"].fillna("").astype(str).str.strip()

        # Filter only mappable intents (prevents Unknown noise)
        j = j[j["lifestyle_id"] != ""].copy()

        # Lifestyle names
        if ls_map is not None and len(ls_map) > 0:
            j = j.merge(ls_map, on="lifestyle_id", how="left")
        if "lifestyle_name" not in j.columns:
            j["lifestyle_name"] = "Unknown"
        j["lifestyle_name"] = j["lifestyle_name"].fillna("Unknown").astype(str).str.strip()
        j.loc[j["lifestyle_name"] == "", "lifestyle_name"] = "Unknown"

        # Intent label: show intent_name (fallback to intent_id if missing)
        if "intent_name" not in j.columns:
            j["intent_name"] = ""
        j["intent_name"] = j["intent_name"].fillna("").astype(str).str.strip()

        j["intent_label"] = j["intent_name"]
        j.loc[j["intent_label"] == "", "intent_label"] = j["intent_id"].astype(str)

        # -------------------------
        # Aggregate distinct customer count per lifestyle-intent
        # -------------------------
        treemap_df = (
            j.groupby(["lifestyle_name", "intent_label"], as_index=False)
            .agg(
                customer_count=("customer_id", "nunique"),
                avg_share=("intent_share", "mean"),
            )
        )

        # Filter small nodes
        treemap_df = treemap_df[treemap_df["customer_count"] >= int(min_customers_node)].copy()

        if len(treemap_df) == 0:
            st.info("Nothing to show (filtered out). Lower the minimum customer threshold or increase rank threshold.")
        else:
            fig = px.treemap(
                treemap_df,
                path=["lifestyle_name", "intent_label"],  # click lifestyle => drill down
                values="customer_count",
                hover_data={
                    "customer_count": True,
                    "avg_share": ":.3f",
                },
            )
            fig.update_layout(height=560, margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Step 5 Outputs (tables + downloads)
# ============================================================================
if st.session_state.get("customer_intent_profile_df") is not None:
    st.subheader("🧠 Customer Intent Profile (Preview)")
    df_out_ci = st.session_state["customer_intent_profile_df"].copy()
    st.dataframe(df_out_ci, use_container_width=True, height=420)
    st.download_button(
        "📥 Download customer_intent_profile.csv",
        data=df_out_ci.to_csv(index=False).encode("utf-8"),
        file_name="customer_intent_profile.csv",
        mime="text/csv",
        use_container_width=True
    )

if st.session_state.get("customer_lifestyle_profile_df") is not None:
    st.subheader("🏠 Customer Lifestyle Profile (Preview)")
    df_out_cl = st.session_state["customer_lifestyle_profile_df"].copy()
    st.dataframe(df_out_cl, use_container_width=True, height=420)
    st.download_button(
        "📥 Download customer_lifestyle_profile.csv",
        data=df_out_cl.to_csv(index=False).encode("utf-8"),
        file_name="customer_lifestyle_profile.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.info("Lifestyle profile is not available (missing ontology mapping or 100% unmapped intents).")

# ============================================================================
# STEP 6: CAMPAIGN AUDIENCE BUILDER (MATCH CAMPAIGN → CUSTOMERS)
# ============================================================================
st.divider()
st.header("Step 6: Campaign Audience Builder")
st.caption("Rank customers for a selected campaign using (campaign intent weights) × (customer intent profile).")

if st.session_state.get("customer_intent_profile_df") is None:
    st.info("👆 Build customer profiles in Step 5 first.")
    st.stop()

df_ci = st.session_state["customer_intent_profile_df"].copy()
df_ci["customer_id"] = df_ci["customer_id"].astype(str)
df_ci["intent_id"] = df_ci["intent_id"].astype(str)
df_ci["intent_share"] = pd.to_numeric(df_ci["intent_share"], errors="coerce").fillna(0.0)

campaigns_df = st.session_state["campaigns_df"].copy()
campaigns_df["campaign_id"] = campaigns_df["campaign_id"].astype(str)

# Choose campaign
campaign_options = campaigns_df["campaign_id"].dropna().astype(str).str.strip().tolist()
campaign_options = [c for c in campaign_options if c != ""]

if len(campaign_options) == 0:
    st.warning("No campaigns found. Upload/paste campaigns in Step 0 first.")
    st.stop()

def _fmt_campaign(cid: str) -> str:
    try:
        row = campaigns_df[campaigns_df["campaign_id"].astype(str).str.strip() == str(cid)].iloc[0]
        nm = str(row.get("campaign_name", "")).strip()
        return f"{cid} — {nm}" if nm else str(cid)
    except Exception:
        return str(cid)

campaign_id = st.selectbox(
    "Select campaign",
    options=campaign_options,
    format_func=_fmt_campaign,
    key="step6_campaign_select_v2"  # ✅ unique key
)


campaign_row = campaigns_df[campaigns_df["campaign_id"] == campaign_id].iloc[0]
st.write(f"**Campaign brief:** {campaign_row['campaign_brief']}")

# Decide weight source
weight_source = st.radio(
    "Intent weight source",
    ["Use Step 3 campaign_intent_profile (if available)", "Manual: pick intents + weights"],
    horizontal=True,
    key="step6_weight_source"
)

intent_weights = None

if weight_source.startswith("Use Step 3"):
    df_cp = st.session_state.get("campaign_intent_profile_df")
    if df_cp is None:
        st.warning("No campaign_intent_profile found in Step 3. Switch to Manual mode or upload profile in Step 3.")
    else:
        df_cp = df_cp.copy()
        df_cp.columns = df_cp.columns.str.strip().str.lower()
        if "campaign_id" not in df_cp.columns or "intent_id" not in df_cp.columns or "weight" not in df_cp.columns:
            st.warning("campaign_intent_profile_df missing required columns. Switch to Manual mode.")
        else:
            w = df_cp[df_cp["campaign_id"].astype(str) == str(campaign_id)].copy()
            if len(w) == 0:
                st.warning("No intents found for this campaign in profile. Switch to Manual mode.")
            else:
                w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)
                s = w["weight"].sum()
                if s <= 0:
                    w["weight"] = 1.0 / max(len(w), 1)
                else:
                    w["weight"] = w["weight"] / s
                intent_weights = w[["intent_id", "weight"]].copy()

                with st.expander("Campaign intent weights (from Step 3)", expanded=False):
                    show_cols = ["intent_id", "weight"]
                    if "intent_name" in w.columns:
                        show_cols = ["intent_id", "intent_name", "weight"]
                    if "rationale" in w.columns:
                        show_cols += ["rationale"]
                    st.dataframe(w[show_cols].sort_values("weight", ascending=False), use_container_width=True)

if intent_weights is None:
    # Manual mode
    if "dim_intent_df" in st.session_state:
        dim_intent_df = st.session_state["dim_intent_df"].copy()
        dim_intent_df.columns = dim_intent_df.columns.str.strip().str.lower()
        intent_options = dim_intent_df[["intent_id", "intent_name"]].drop_duplicates()
        intent_options["label"] = intent_options["intent_id"].astype(str) + " — " + intent_options["intent_name"].astype(str)
        label_to_id = dict(zip(intent_options["label"], intent_options["intent_id"]))
        labels = intent_options["label"].tolist()
    else:
        # fallback to what exists in customer profile
        intent_ids = sorted(df_ci["intent_id"].unique().tolist())
        labels = intent_ids
        label_to_id = {x: x for x in intent_ids}

    selected = st.multiselect("Pick intents for this campaign", options=labels, default=labels[:6], key="step6_manual_intents")
    if len(selected) == 0:
        st.warning("Pick at least 1 intent to rank customers.")
        st.stop()

    st.write("Set weights (they will be normalized to sum = 1):")
    weights = []
    cols = st.columns(min(3, len(selected)))
    for idx, lab in enumerate(selected):
        with cols[idx % len(cols)]:
            w = st.number_input(f"Weight: {lab}", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key=f"step6_w_{idx}")
            weights.append((label_to_id[lab], float(w)))

    wdf = pd.DataFrame(weights, columns=["intent_id", "weight"])
    s = wdf["weight"].sum()
    if s <= 0:
        wdf["weight"] = 1.0 / max(len(wdf), 1)
    else:
        wdf["weight"] = wdf["weight"] / s
    intent_weights = wdf.copy()

# Filters (minimal but useful)
st.subheader("🔎 Optional Filters")
f1, f2, f3 = st.columns(3)

with f1:
    min_txn_amt = st.number_input("Min total spend (amt) in history", min_value=0.0, value=0.0, step=10.0, key="step6_min_spend")
with f2:
    min_txn_count = st.number_input("Min transaction count", min_value=0, value=0, step=1, key="step6_min_txn_count")
with f3:
    recency_days = st.number_input("Recency window (days): last purchase within", min_value=0, value=0, step=10, key="step6_recency_days")

rank_btn = st.button("🎯 Build Ranked Audience", type="primary", use_container_width=True, key="step6_rank_btn")

if "campaign_audience_ranked_df" not in st.session_state:
    st.session_state["campaign_audience_ranked_df"] = None

if rank_btn:
    try:
        # Compute match score: sum_i customer_intent_share(i) * campaign_weight(i)
        w = intent_weights.copy()
        w["intent_id"] = w["intent_id"].astype(str)
        w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)

        # join
        j = df_ci.merge(w, on="intent_id", how="inner")
        j["match_component"] = j["intent_share"] * j["weight"]

        score = (
            j.groupby("customer_id", as_index=False)["match_component"]
            .sum()
            .rename(columns={"match_component": "match_score"})
        )

        # Add a simple explanation: top contributing intents per customer
        j = j.sort_values(["customer_id", "match_component"], ascending=[True, False])
        j["comp_rank"] = j.groupby("customer_id").cumcount() + 1
        top_comp = j[j["comp_rank"] <= 5].copy()

        # Make a compact explanation string
        def _mk_explain(g):
            parts = []
            for _, r in g.iterrows():
                nm = str(r.get("intent_name", "")) if "intent_name" in g.columns else ""
                if nm:
                    parts.append(f"{r['intent_id']} ({nm}): {r['match_component']:.3f}")
                else:
                    parts.append(f"{r['intent_id']}: {r['match_component']:.3f}")
            return " | ".join(parts)

        explain = top_comp.groupby("customer_id").apply(_mk_explain).reset_index(name="top_contributors")

        out = score.merge(explain, on="customer_id", how="left")

        # Attach spend, count, recency from txn_df
        tx = txn_df.copy()
        tx["customer_id"] = tx["customer_id"].astype(str)
        tx["tx_date"] = pd.to_datetime(tx["tx_date"], errors="coerce")

        agg = tx.groupby("customer_id").agg(
            total_spend=("amt", "sum"),
            txn_count=("tx_id", "count"),
            last_tx_date=("tx_date", "max")
        ).reset_index()

        out = out.merge(agg, on="customer_id", how="left")

        # Apply filters
        if float(min_txn_amt) > 0:
            out = out[out["total_spend"].fillna(0.0) >= float(min_txn_amt)]
        if int(min_txn_count) > 0:
            out = out[out["txn_count"].fillna(0) >= int(min_txn_count)]
        if int(recency_days) > 0:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(recency_days))
            out = out[out["last_tx_date"].fillna(pd.Timestamp("1900-01-01")) >= cutoff]

        out = out.sort_values("match_score", ascending=False).reset_index(drop=True)
        out["rank"] = out.index + 1
        out["campaign_id"] = str(campaign_id)
        out["campaign_name"] = str(campaign_row["campaign_name"])
        out["built_at"] = pd.Timestamp.now().isoformat()

        st.session_state["campaign_audience_ranked_df"] = out

        st.success(f"✅ Ranked audience built: {len(out):,} customers")

    except Exception as e:
        st.error(f"❌ Failed to build ranked audience: {e}")
        st.exception(e)

if st.session_state.get("campaign_audience_ranked_df") is not None:
    st.subheader("🏆 Ranked Audience (Preview)")
    aud = st.session_state["campaign_audience_ranked_df"]
    st.dataframe(aud.head(500), use_container_width=True, height=420)

    st.download_button(
        "📥 Download campaign_audience_ranked.csv",
        data=aud.to_csv(index=False).encode("utf-8"),
        file_name=f"campaign_audience_ranked_{campaign_id}.csv",
        mime="text/csv",
        use_container_width=True
    )
# ============================================================================
# STEP 6: CAMPAIGN AUDIENCE BUILDER (MATCH CAMPAIGN → CUSTOMERS)
# ============================================================================
st.divider()
st.header("Step 6: Campaign Audience Builder")
st.caption("Rank customers for a selected campaign using (campaign intent weights) × (customer intent profile).")

if st.session_state.get("customer_intent_profile_df") is None:
    st.info("👆 Build customer profiles in Step 5 first.")
    st.stop()

# Pull txn_df for filters (Step 1 prerequisite)
if "txn_df" not in st.session_state:
    st.info("👆 Upload Transaction CSV in Step 1 first.")
    st.stop()

txn_df = st.session_state["txn_df"].copy()

df_ci = st.session_state["customer_intent_profile_df"].copy()
df_ci.columns = df_ci.columns.str.strip().str.lower()
df_ci["customer_id"] = df_ci["customer_id"].astype(str).str.strip()
df_ci["intent_id"] = df_ci["intent_id"].astype(str).str.strip()
if "intent_share" not in df_ci.columns:
    df_ci["intent_share"] = 0.0
df_ci["intent_share"] = pd.to_numeric(df_ci["intent_share"], errors="coerce").fillna(0.0)

campaigns_df = st.session_state["campaigns_df"].copy()
campaigns_df.columns = campaigns_df.columns.str.strip().str.lower()
campaigns_df["campaign_id"] = campaigns_df["campaign_id"].astype(str).str.strip()

# Choose campaign
campaign_id = st.selectbox(
    "Select campaign",
    options=campaigns_df["campaign_id"].tolist(),
    format_func=lambda cid: f"{cid} — {campaigns_df.loc[campaigns_df['campaign_id'] == cid, 'campaign_name'].iloc[0]}",
    key="step6_campaign_select"
)

campaign_row = campaigns_df[campaigns_df["campaign_id"] == campaign_id].iloc[0]
st.write(f"**Campaign brief:** {campaign_row['campaign_brief']}")

# Decide weight source
weight_source = st.radio(
    "Intent weight source",
    ["Use Step 3 campaign_intent_profile (if available)", "Manual: pick intents + weights"],
    horizontal=True,
    key="step6_weight_source"
)

intent_weights = None

if weight_source.startswith("Use Step 3"):
    df_cp = st.session_state.get("campaign_intent_profile_df")
    if df_cp is None:
        st.warning("No campaign_intent_profile found in Step 3. Switch to Manual mode or upload profile in Step 3.")
    else:
        df_cp = df_cp.copy()
        df_cp.columns = df_cp.columns.str.strip().str.lower()
        if "campaign_id" not in df_cp.columns or "intent_id" not in df_cp.columns or "weight" not in df_cp.columns:
            st.warning("campaign_intent_profile_df missing required columns. Switch to Manual mode.")
        else:
            w = df_cp[df_cp["campaign_id"].astype(str).str.strip() == str(campaign_id)].copy()
            if len(w) == 0:
                st.warning("No intents found for this campaign in profile. Switch to Manual mode.")
            else:
                w["intent_id"] = w["intent_id"].astype(str).str.strip()
                w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)
                s = w["weight"].sum()
                if s <= 0:
                    w["weight"] = 1.0 / max(len(w), 1)
                else:
                    w["weight"] = w["weight"] / s
                intent_weights = w[["intent_id", "weight"]].copy()

                with st.expander("Campaign intent weights (from Step 3)", expanded=False):
                    show_cols = ["intent_id", "weight"]
                    if "intent_name" in w.columns:
                        show_cols = ["intent_id", "intent_name", "weight"]
                    if "rationale" in w.columns:
                        show_cols += ["rationale"]
                    st.dataframe(w[show_cols].sort_values("weight", ascending=False), use_container_width=True)

if intent_weights is None:
    # Manual mode
    if "dim_intent_df" in st.session_state and st.session_state["dim_intent_df"] is not None:
        dim_intent_df = st.session_state["dim_intent_df"].copy()
        dim_intent_df.columns = dim_intent_df.columns.str.strip().str.lower()

        if "intent_name" not in dim_intent_df.columns:
            dim_intent_df["intent_name"] = ""
        dim_intent_df["intent_id"] = dim_intent_df["intent_id"].astype(str).str.strip()
        dim_intent_df["intent_name"] = dim_intent_df["intent_name"].fillna("").astype(str).str.strip()

        intent_options = dim_intent_df[["intent_id", "intent_name"]].drop_duplicates()
        intent_options["label"] = intent_options["intent_id"].astype(str) + " — " + intent_options["intent_name"].astype(str)
        label_to_id = dict(zip(intent_options["label"], intent_options["intent_id"]))
        labels = intent_options["label"].tolist()
    else:
        # fallback to what exists in customer profile
        intent_ids = sorted(df_ci["intent_id"].unique().tolist())
        labels = intent_ids
        label_to_id = {x: x for x in intent_ids}

    selected = st.multiselect(
        "Pick intents for this campaign",
        options=labels,
        default=labels[:6],
        key="step6_manual_intents"
    )
    if len(selected) == 0:
        st.warning("Pick at least 1 intent to rank customers.")
        st.stop()

    st.write("Set weights (they will be normalized to sum = 1):")
    weights = []
    cols = st.columns(min(3, len(selected)))
    for idx, lab in enumerate(selected):
        with cols[idx % len(cols)]:
            ww = st.number_input(
                f"Weight: {lab}",
                min_value=0.0, max_value=10.0,
                value=1.0, step=0.1,
                key=f"step6_w_{idx}"
            )
            weights.append((label_to_id[lab], float(ww)))

    wdf = pd.DataFrame(weights, columns=["intent_id", "weight"])
    s = wdf["weight"].sum()
    if s <= 0:
        wdf["weight"] = 1.0 / max(len(wdf), 1)
    else:
        wdf["weight"] = wdf["weight"] / s
    intent_weights = wdf.copy()

# Filters (minimal but useful)
st.subheader("🔎 Optional Filters")
f1, f2, f3 = st.columns(3)

with f1:
    min_txn_amt = st.number_input(
        "Min total spend (amt) in history",
        min_value=0.0, value=0.0, step=10.0,
        key="step6_min_spend"
    )
with f2:
    min_txn_count = st.number_input(
        "Min transaction count",
        min_value=0, value=0, step=1,
        key="step6_min_txn_count"
    )
with f3:
    recency_days = st.number_input(
        "Recency window (days): last purchase within",
        min_value=0, value=0, step=10,
        key="step6_recency_days"
    )

rank_btn = st.button("🎯 Build Ranked Audience", type="primary", use_container_width=True, key="step6_rank_btn")

if "campaign_audience_ranked_df" not in st.session_state:
    st.session_state["campaign_audience_ranked_df"] = None

# keep the join detail for treemap after building audience
if "step6_join_detail_df" not in st.session_state:
    st.session_state["step6_join_detail_df"] = None

if rank_btn:
    try:
        # Compute match score: sum_i customer_intent_share(i) * campaign_weight(i)
        w = intent_weights.copy()
        w["intent_id"] = w["intent_id"].astype(str).str.strip()
        w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)

        # join customer intents with weights
        j = df_ci.merge(w, on="intent_id", how="inner")
        j["match_component"] = j["intent_share"] * j["weight"]

        score = (
            j.groupby("customer_id", as_index=False)["match_component"]
            .sum()
            .rename(columns={"match_component": "match_score"})
        )

        # Explanation: top contributing intents per customer
        j = j.sort_values(["customer_id", "match_component"], ascending=[True, False])
        j["comp_rank"] = j.groupby("customer_id").cumcount() + 1
        top_comp = j[j["comp_rank"] <= 5].copy()

        def _mk_explain(g):
            parts = []
            for _, r in g.iterrows():
                nm = str(r.get("intent_name", "")) if "intent_name" in g.columns else ""
                if nm:
                    parts.append(f"{r['intent_id']} ({nm}): {r['match_component']:.3f}")
                else:
                    parts.append(f"{r['intent_id']}: {r['match_component']:.3f}")
            return " | ".join(parts)

        explain = top_comp.groupby("customer_id").apply(_mk_explain).reset_index(name="top_contributors")

        out = score.merge(explain, on="customer_id", how="left")

        # Attach spend, count, recency from txn_df
        tx = txn_df.copy()
        tx.columns = tx.columns.str.strip().str.lower()
        tx["customer_id"] = tx["customer_id"].astype(str).str.strip()
        tx["tx_date"] = pd.to_datetime(tx["tx_date"], errors="coerce")

        agg = tx.groupby("customer_id").agg(
            total_spend=("amt", "sum"),
            txn_count=("tx_id", "count"),
            last_tx_date=("tx_date", "max")
        ).reset_index()

        out = out.merge(agg, on="customer_id", how="left")

        # Apply filters
        if float(min_txn_amt) > 0:
            out = out[out["total_spend"].fillna(0.0) >= float(min_txn_amt)]
        if int(min_txn_count) > 0:
            out = out[out["txn_count"].fillna(0).astype(int) >= int(min_txn_count)]
        if int(recency_days) > 0:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(recency_days))
            out = out[out["last_tx_date"].fillna(pd.Timestamp("1900-01-01")) >= cutoff]

        out = out.sort_values("match_score", ascending=False).reset_index(drop=True)
        out["rank"] = out.index + 1
        out["campaign_id"] = str(campaign_id)
        out["campaign_name"] = str(campaign_row["campaign_name"])
        out["built_at"] = pd.Timestamp.now().isoformat()

        st.session_state["campaign_audience_ranked_df"] = out

        # Save join detail for treemap (only rows with customers that survived filters)
        kept_customers = set(out["customer_id"].astype(str).str.strip().tolist())
        j_keep = j[j["customer_id"].astype(str).str.strip().isin(kept_customers)].copy()
        st.session_state["step6_join_detail_df"] = j_keep

        st.success(f"✅ Ranked audience built: {len(out):,} customers")

    except Exception as e:
        st.error(f"❌ Failed to build ranked audience: {e}")
        st.exception(e)

if st.session_state.get("campaign_audience_ranked_df") is not None:
    st.subheader("🏆 Ranked Audience (Preview)")
    aud = st.session_state["campaign_audience_ranked_df"]
    st.dataframe(aud.head(500), use_container_width=True, height=420)

    st.download_button(
        "📥 Download campaign_audience_ranked.csv",
        data=aud.to_csv(index=False).encode("utf-8"),
        file_name=f"campaign_audience_ranked_{campaign_id}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================================
# STEP 6 Visual: Treemap drilldown (Lifestyle -> Intent) UNDER SELECTED CAMPAIGN
# ============================================================================
st.divider()
st.subheader("🧩 Campaign Audience: Lifestyle → Intent Treemap (drilldown)")
st.caption("This treemap reflects ONLY customers in the ranked audience (after filters). Size = # customers (distinct). Click a lifestyle to zoom into its intents.")

aud = st.session_state.get("campaign_audience_ranked_df")
j_detail = st.session_state.get("step6_join_detail_df")

if aud is None or len(aud) == 0 or j_detail is None or len(j_detail) == 0:
    st.info("Build a ranked audience first (click **Build Ranked Audience** above).")
else:
    if "dim_intent_df" not in st.session_state or st.session_state["dim_intent_df"] is None:
        st.info("Treemap needs ontology mapping (dim_intent_df). Generate/upload ontology in Step 2.")
    else:
        import plotly.express as px

        dim_i = st.session_state["dim_intent_df"].copy()
        dim_i.columns = dim_i.columns.str.strip().str.lower()

        # Required
        dim_i["intent_id"] = dim_i["intent_id"].astype(str).str.strip()
        if "intent_name" not in dim_i.columns:
            dim_i["intent_name"] = ""
        dim_i["intent_name"] = dim_i["intent_name"].fillna("").astype(str).str.strip()

        if "lifestyle_id" not in dim_i.columns:
            dim_i["lifestyle_id"] = ""
        dim_i["lifestyle_id"] = dim_i["lifestyle_id"].fillna("").astype(str).str.strip()

        dim_i_small = dim_i[["intent_id", "intent_name", "lifestyle_id"]].drop_duplicates().copy()

        # Lifestyle name map
        ls_map = None
        if "dim_lifestyle_df" in st.session_state and st.session_state["dim_lifestyle_df"] is not None:
            dim_lifestyle_df = st.session_state["dim_lifestyle_df"].copy()
            dim_lifestyle_df.columns = dim_lifestyle_df.columns.str.strip().str.lower()
            if "lifestyle_id" in dim_lifestyle_df.columns and "lifestyle_name" in dim_lifestyle_df.columns:
                ls_map = dim_lifestyle_df[["lifestyle_id", "lifestyle_name"]].drop_duplicates().copy()
                ls_map["lifestyle_id"] = ls_map["lifestyle_id"].astype(str).str.strip()
                ls_map["lifestyle_name"] = ls_map["lifestyle_name"].fillna("Unknown").astype(str).str.strip()

        if ls_map is None or len(ls_map) == 0:
            # Reuse ontology JSON fallback helper if available
            try:
                ls_map = _get_lifestyle_name_map_from_ontology()
            except Exception:
                ls_map = pd.DataFrame([{"lifestyle_id": "", "lifestyle_name": "Unknown"}])

        # Base from join detail (already filtered to kept customers)
        jj = j_detail.copy()
        jj.columns = jj.columns.str.strip().str.lower()
        jj["customer_id"] = jj["customer_id"].astype(str).str.strip()
        jj["intent_id"] = jj["intent_id"].astype(str).str.strip()

        # Optional controls for campaign treemap
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            min_customers_node = st.number_input(
                "Hide nodes with < customers (campaign view)",
                min_value=1, value=5, step=1,
                key="step6_treemap_min_customers"
            )
        with cc2:
            comp_rank_threshold = st.selectbox(
                "Include customer-intent rows if comp_rank ≤ (explain depth)",
                options=[1, 3, 5, 999],
                index=2,
                key="step6_treemap_comp_rank_threshold"
            )

        if "comp_rank" in jj.columns and int(comp_rank_threshold) != 999:
            jj = jj[jj["comp_rank"].fillna(999).astype(int) <= int(comp_rank_threshold)].copy()

        # Attach lifestyle_id + intent_name
        jj = jj.merge(dim_i_small, on="intent_id", how="left")

        if "lifestyle_id" not in jj.columns:
            jj["lifestyle_id"] = ""
        jj["lifestyle_id"] = jj["lifestyle_id"].fillna("").astype(str).str.strip()

        jj = jj[jj["lifestyle_id"] != ""].copy()

        # Lifestyle name
        jj = jj.merge(ls_map, on="lifestyle_id", how="left")
        if "lifestyle_name" not in jj.columns:
            jj["lifestyle_name"] = "Unknown"
        jj["lifestyle_name"] = jj["lifestyle_name"].fillna("Unknown").astype(str).str.strip()
        jj.loc[jj["lifestyle_name"] == "", "lifestyle_name"] = "Unknown"

        # Intent label: show name (fallback to id)
        if "intent_name" not in jj.columns:
            jj["intent_name"] = ""
        jj["intent_name"] = jj["intent_name"].fillna("").astype(str).str.strip()
        jj["intent_label"] = jj["intent_name"]
        jj.loc[jj["intent_label"] == "", "intent_label"] = jj["intent_id"].astype(str)

        # Aggregate distinct customers per lifestyle-intent in audience
        treemap_df = (
            jj.groupby(["lifestyle_name", "intent_label"], as_index=False)
            .agg(
                customer_count=("customer_id", "nunique"),
                avg_match_component=("match_component", "mean"),
            )
        )

        treemap_df = treemap_df[treemap_df["customer_count"] >= int(min_customers_node)].copy()

        if len(treemap_df) == 0:
            st.info("Nothing to show (filtered out). Lower the minimum customer threshold.")
        else:
            fig = px.treemap(
                treemap_df,
                path=["lifestyle_name", "intent_label"],
                values="customer_count",
                hover_data={
                    "customer_count": True,
                    "avg_match_component": ":.4f",
                },
            )
            fig.update_layout(height=560, margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


st.divider()
st.caption("End of app.")

