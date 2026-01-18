import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Semantic Audience Studio (Lean)", layout="wide")

# ============================================================================
# LEAN PRINCIPLES
# - Minimal UI, mostly st.file_uploader + st.button + st.dataframe
# - Pure functions for each step (easy to modify & test)
# - All outputs stored in st.session_state
# ============================================================================

# ----------------------------
# Helpers (small + consistent)
# ----------------------------
def ss_get(key, default=None):
    return st.session_state[key] if key in st.session_state else default

def ss_set(key, value):
    st.session_state[key] = value

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df

def require_cols(df: pd.DataFrame, required: list[str], name: str):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(list(missing))}")

# ============================================================================
# STEP 0: Load Campaigns (JSON paste OR CSV)
# ============================================================================
def load_campaigns_from_json(json_text: str) -> pd.DataFrame:
    obj = json.loads(json_text)
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list) or len(obj) == 0:
        raise ValueError("Campaign JSON must be object or non-empty list.")
    df = pd.DataFrame(obj)
    df = normalize_cols(df)
    require_cols(df, ["campaign_id", "campaign_name", "campaign_brief"], "campaigns_df")
    df["campaign_id"] = df["campaign_id"].astype(str)
    df["campaign_name"] = df["campaign_name"].astype(str)
    df["campaign_brief"] = df["campaign_brief"].astype(str)
    return df

def load_campaigns_from_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = normalize_cols(df)
    require_cols(df, ["campaign_id", "campaign_name", "campaign_brief"], "campaigns_df")
    df["campaign_id"] = df["campaign_id"].astype(str)
    df["campaign_name"] = df["campaign_name"].astype(str)
    df["campaign_brief"] = df["campaign_brief"].astype(str)
    return df

# ============================================================================
# STEP 1: Load Product + Transaction CSV
# ============================================================================
def load_products(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = normalize_cols(df)
    require_cols(df, ["product_id", "product_title", "product_description"], "products_df")

    df["product_id"] = df["product_id"].astype(str)
    df["product_title"] = df["product_title"].fillna("").astype(str)
    df["product_description"] = df["product_description"].fillna("").astype(str)

    df["product_text"] = (df["product_title"] + " | " + df["product_description"]).str.lower()
    out = df[["product_id", "product_title", "product_text"]].drop_duplicates("product_id").reset_index(drop=True)
    out["product_name"] = out["product_title"]
    return out

def load_transactions(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = normalize_cols(df)
    require_cols(df, ["tx_id", "customer_id", "product_id", "tx_date", "qty", "price"], "txn_df")

    df["tx_id"] = df["tx_id"].astype(str)
    df["customer_id"] = df["customer_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    df["tx_date"] = pd.to_datetime(df["tx_date"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["amt"] = df["qty"] * df["price"]
    return df

# ============================================================================
# APP UI (LEAN)
# ============================================================================
st.title("Semantic Audience Studio (Lean)")
st.caption("No fancy UI. Pure functions + clean pipeline. UX comes later.")

with st.expander("Step 0 — Campaign Input", expanded=True):
    mode = st.radio("Campaign input mode", ["Paste JSON", "Upload CSV"], horizontal=True)

    if mode == "Paste JSON":
        default_json =  [
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

        json_text = st.text_area("Campaign JSON", value=json.dumps(default_json, indent=2), height=180)
        if st.button("Load Campaigns (JSON)"):
            try:
                df = load_campaigns_from_json(json_text)
                ss_set("campaigns_df", df)
                st.success(f"Loaded campaigns: {len(df)} rows")
            except Exception as e:
                st.error(str(e))
    else:
        up = st.file_uploader("Upload campaigns.csv", type=["csv"])
        if st.button("Load Campaigns (CSV)"):
            try:
                if up is None:
                    raise ValueError("Upload campaigns.csv first.")
                df = load_campaigns_from_csv(up)
                ss_set("campaigns_df", df)
                st.success(f"Loaded campaigns: {len(df)} rows")
            except Exception as e:
                st.error(str(e))

    if ss_get("campaigns_df") is not None:
        st.dataframe(ss_get("campaigns_df").head(20), use_container_width=True)

with st.expander("Step 1 — Upload Data (Product + Transactions)", expanded=True):
    pfile = st.file_uploader("Upload products.csv", type=["csv"], key="lean_products")
    tfile = st.file_uploader("Upload transactions.csv", type=["csv"], key="lean_txns")

    if st.button("Load Product + Transaction Data"):
        try:
            if pfile is None or tfile is None:
                raise ValueError("Upload BOTH products.csv and transactions.csv")
            catalog_df = load_products(pfile)
            txn_df = load_transactions(tfile)
            ss_set("catalog_df", catalog_df)
            ss_set("txn_df", txn_df)
            st.success(f"Loaded products={len(catalog_df):,} | txns={len(txn_df):,}")
        except Exception as e:
            st.error(str(e))

    if ss_get("catalog_df") is not None:
        st.write("catalog_df (head)")
        st.dataframe(ss_get("catalog_df").head(10), use_container_width=True)

    if ss_get("txn_df") is not None:
        st.write("txn_df (head)")
        st.dataframe(ss_get("txn_df").head(10), use_container_width=True)

st.divider()
st.caption("Next: Part 2 = Ontology + Campaign Intent Profile (lean, still function-first).")

# ============================================================================
# STEP 2: Ontology (Upload OR LLM)
# ============================================================================
def load_ontology_json(uploaded_json) -> dict:
    raw = uploaded_json.read().decode("utf-8")
    ontology = json.loads(raw)
    if not isinstance(ontology, dict) or "lifestyles" not in ontology:
        raise ValueError("Ontology must be a dict containing 'lifestyles'.")
    return ontology

def build_dim_tables_from_ontology(ontology: dict):
    lifestyles = ontology.get("lifestyles", [])
    version = str(ontology.get("version", "v1"))

    dim_lifestyle_rows, dim_intent_rows = [], []
    for ls in lifestyles:
        ls_id = ls.get("lifestyle_id", "")
        dim_lifestyle_rows.append({
            "lifestyle_id": ls_id,
            "lifestyle_name": ls.get("lifestyle_name", ""),
            "definition": ls.get("definition", ""),
            "version": version
        })
        for it in (ls.get("intents", []) or []):
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
    return dim_lifestyle_df, dim_intent_df

def extract_json_from_text(text: str) -> dict:
    import re
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return json.loads(text.strip())

def generate_ontology_with_gemini(catalog_df: pd.DataFrame, api_key: str, n_lifestyles=6, max_intents=5, language="en"):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    sample = catalog_df["product_text"].head(60).tolist()
    examples = "\n".join([f"- {t[:240]}" for t in sample])

    prompt = f"""
You are proposing a Lifestyle→Intent ontology for retail marketing.

Input product examples:
{examples}

Task:
- Produce EXACTLY {n_lifestyles} Lifestyle parents
- Under each, up to {max_intents} intents
- Each intent: intent_id, intent_name, definition, include_examples, exclude_examples
- Each lifestyle: lifestyle_id, lifestyle_name, definition
- Output language: {language}

Return STRICT minified JSON only:
{{
  "version":"v1",
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

    resp = model.generate_content(prompt)
    ontology = extract_json_from_text(resp.text)
    return ontology

# ============================================================================
# STEP 3: Campaign Intent Profile (Upload OR LLM)
# ============================================================================
def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df = df.sort_values(["campaign_id", "weight"], ascending=[True, False])
    df["rank"] = df.groupby("campaign_id").cumcount() + 1

    def _norm(g):
        s = g["weight"].sum()
        if s <= 0:
            g["weight"] = 1.0 / max(len(g), 1)
        else:
            g["weight"] = g["weight"] / s
        return g

    df = df.groupby("campaign_id", group_keys=False).apply(_norm)
    return df

def load_campaign_intent_profile_csv(uploaded_file, campaigns_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = normalize_cols(df)
    require_cols(df, ["campaign_id", "intent_id", "weight"], "campaign_intent_profile_df")
    df["campaign_id"] = df["campaign_id"].astype(str)
    df["intent_id"] = df["intent_id"].astype(str)

    if "campaign_name" not in df.columns and campaigns_df is not None:
        df = df.merge(campaigns_df[["campaign_id", "campaign_name"]], on="campaign_id", how="left")

    if "intent_name" not in df.columns:
        df["intent_name"] = ""
    if "rationale" not in df.columns:
        df["rationale"] = ""

    df = normalize_weights(df)
    return df

def generate_campaign_intent_profile_with_gemini(campaigns_df: pd.DataFrame, dim_intent_df: pd.DataFrame, api_key: str, top_n=6, output_language="en"):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    intent_candidates = dim_intent_df[["intent_id", "intent_name", "definition"]].fillna("").to_dict("records")
    intent_snippet = json.dumps(intent_candidates, ensure_ascii=False)[:16000]

    batch_campaigns = campaigns_df[["campaign_id", "campaign_name", "campaign_brief"]].to_dict("records")

    prompt = f"""
You are a marketing analyst. Map EACH campaign brief into a weighted intent profile.

You MUST choose intents ONLY from this intent list:
{intent_snippet}

Input campaigns:
{json.dumps(batch_campaigns, ensure_ascii=False)}

Task for EACH campaign:
- Select TOP {top_n} intents
- For each: intent_id, intent_name, weight (>0), rationale (1–2 sentences)
- Rationale language: {output_language}

Return STRICT minified JSON only:
{{
  "campaign_profiles":[
    {{
      "campaign_id":"...",
      "campaign_name":"...",
      "top_intents":[
        {{"intent_id":"IN_...","intent_name":"...","weight":0.0,"rationale":"..."}}
      ]
    }}
  ]
}}
""".strip()

    resp = model.generate_content(prompt)
    data = extract_json_from_text(resp.text)

    rows = []
    for cp in data.get("campaign_profiles", []):
        cid = str(cp.get("campaign_id", ""))
        cname = str(cp.get("campaign_name", ""))
        intents = cp.get("top_intents", []) or []
        for it in intents:
            rows.append({
                "campaign_id": cid,
                "campaign_name": cname,
                "intent_id": str(it.get("intent_id", "")),
                "intent_name": str(it.get("intent_name", "")),
                "weight": float(it.get("weight", 0.0)),
                "rationale": str(it.get("rationale", "")),
            })

    df = pd.DataFrame(rows)
    df = normalize_cols(df)
    df = normalize_weights(df)
    return df

# ============================================================================
# APP UI (LEAN)
# ============================================================================
st.subheader("Step 2 — Ontology")
ontology_upload = st.file_uploader("Upload ontology_v1.json (optional)", type=["json"], key="lean_ontology")
api_key_step2 = st.text_input("Gemini API Key (only if generating ontology)", type="password", key="lean_gemini_step2")
gen_ont = st.button("Generate Ontology (LLM)", key="lean_gen_ontology")
load_ont = st.button("Load Ontology (Upload)", key="lean_load_ontology")

if load_ont:
    try:
        if ontology_upload is None:
            raise ValueError("Upload ontology_v1.json first.")
        ontology = load_ontology_json(ontology_upload)
        dim_lifestyle_df, dim_intent_df = build_dim_tables_from_ontology(ontology)

        ss_set("ontology", ontology)
        ss_set("dim_lifestyle_df", dim_lifestyle_df)
        ss_set("dim_intent_df", dim_intent_df)
        st.success(f"Loaded ontology: lifestyles={len(dim_lifestyle_df)} intents={len(dim_intent_df)}")
    except Exception as e:
        st.error(str(e))

if gen_ont:
    try:
        catalog_df = ss_get("catalog_df")
        if catalog_df is None:
            raise ValueError("Load products first (Step 1).")
        if not api_key_step2:
            raise ValueError("Provide Gemini API key.")
        ontology = generate_ontology_with_gemini(catalog_df, api_key_step2)
        dim_lifestyle_df, dim_intent_df = build_dim_tables_from_ontology(ontology)

        ss_set("ontology", ontology)
        ss_set("dim_lifestyle_df", dim_lifestyle_df)
        ss_set("dim_intent_df", dim_intent_df)
        st.success(f"Generated ontology: lifestyles={len(dim_lifestyle_df)} intents={len(dim_intent_df)}")
    except Exception as e:
        st.error(str(e))

if ss_get("dim_intent_df") is not None:
    st.dataframe(ss_get("dim_intent_df").head(20), use_container_width=True)

st.divider()

st.subheader("Step 3 — Campaign Intent Profile")
camp_profile_upload = st.file_uploader("Upload campaign_intent_profile.csv (optional)", type=["csv"], key="lean_camp_profile")
api_key_step3 = known = st.text_input("Gemini API Key (only if generating campaign profile)", type="password", key="lean_gemini_step3")
gen_cp = st.button("Generate Campaign Intent Profile (LLM)", key="lean_gen_cp")
load_cp = st.button("Load Campaign Intent Profile (Upload)", key="lean_load_cp")

if load_cp:
    try:
        if camp_profile_upload is None:
            raise ValueError("Upload campaign_intent_profile.csv first.")
        campaigns_df = ss_get("campaigns_df")
        df = load_campaign_intent_profile_csv(camp_profile_upload, campaigns_df=campaigns_df)
        ss_set("campaign_intent_profile_df", df)
        st.success(f"Loaded campaign intent profile: {len(df)} rows")
    except Exception as e:
        st.error(str(e))

if gen_cp:
    try:
        campaigns_df = ss_get("campaigns_df")
        dim_intent_df = ss_get("dim_intent_df")
        if campaigns_df is None:
            raise ValueError("Load campaigns first (Step 0).")
        if dim_intent_df is None:
            raise ValueError("Load/generate ontology first (Step 2).")
        if not api_key_step3:
            raise ValueError("Provide Gemini API key.")
        df = generate_campaign_intent_profile_with_gemini(campaigns_df, dim_intent_df, api_key_step3)
        ss_set("campaign_intent_profile_df", df)
        st.success(f"Generated campaign intent profile: {len(df)} rows")
    except Exception as e:
        st.error(str(e))

if ss_get("campaign_intent_profile_df") is not None:
    st.dataframe(ss_get("campaign_intent_profile_df").head(30), use_container_width=True)

st.caption("Next: Part 3 = Product labeling + Customer profiles + Audience ranking (still lean).")

# ============================================================================
# STEP 4: Product → Intent Labels (Upload OR LLM)
# ============================================================================
def load_product_intent_labels_csv(uploaded_file, catalog_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = normalize_cols(df)
    require_cols(df, ["product_id", "intent_id"], "product_intent_labels_df")

    if "score" not in df.columns and "weight" in df.columns:
        df["score"] = df["weight"]
    if "score" not in df.columns:
        df["score"] = 1.0

    df["product_id"] = df["product_id"].astype(str)
    df["intent_id"] = df["intent_id"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

    if "rank" not in df.columns:
        df = df.sort_values(["product_id", "score"], ascending=[True, False])
        df["rank"] = df.groupby("product_id").cumcount() + 1

    if "product_name" not in df.columns and catalog_df is not None and "product_name" in catalog_df.columns:
        df = df.merge(catalog_df[["product_id", "product_name"]], on="product_id", how="left")

    if "intent_name" not in df.columns:
        df["intent_name"] = ""

    return df

def generate_product_intent_labels_with_gemini(catalog_df: pd.DataFrame, dim_intent_df: pd.DataFrame, api_key: str, top_k=3, min_score=0.25):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    intent_candidates = dim_intent_df[["intent_id", "intent_name", "definition"]].fillna("").to_dict("records")
    intent_snippet = json.dumps(intent_candidates, ensure_ascii=False)[:16000]

    # keep it lean: label first N products (you can expand later)
    rows = []
    for _, r in catalog_df.iterrows():
        pid = str(r["product_id"])
        pname = str(r.get("product_name", ""))
        ptext = str(r.get("product_text", ""))[:600]

        prompt = f"""
You are labeling retail products into a fixed Intent Ontology for marketing.
Choose intents ONLY from this list:
{intent_snippet}

Product:
{{"product_id":"{pid}","product_name":"{pname}","product_text":"{ptext}"}}

Return STRICT minified JSON only:
{{
  "top_intents":[
    {{"intent_id":"IN_...","intent_name":"...","score":0.0,"evidence":"...","reason":"..."}}
  ]
}}
""".strip()

        resp = model.generate_content(prompt)
        data = extract_json_from_text(resp.text)
        intents = data.get("top_intents", []) or []
        intents = sorted(intents, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        intents = [x for x in intents if float(x.get("score", 0.0)) >= float(min_score)][:int(top_k)]

        for rank, it in enumerate(intents, start=1):
            rows.append({
                "product_id": pid,
                "product_name": pname,
                "rank": rank,
                "intent_id": str(it.get("intent_id", "")),
                "intent_name": str(it.get("intent_name", "")),
                "score": float(it.get("score", 0.0)),
                "evidence": str(it.get("evidence", "")),
                "reason": str(it.get("reason", "")),
            })

    df = pd.DataFrame(rows)
    return df

# ============================================================================
# STEP 5: Customer Intent Profiles (Deterministic)
# ============================================================================
def build_customer_intent_profile(txn_df: pd.DataFrame, labels_df: pd.DataFrame, weight_method="amt", apply_score=True, normalize="sum_to_1"):
    t = txn_df.copy()
    l = labels_df.copy()

    t["amt"] = t["amt"] if "amt" in t.columns else (t["qty"] * t["price"])
    joined = t.merge(l[["product_id", "intent_id", "intent_name", "score"]], on="product_id", how="inner")
    if joined.empty:
        raise ValueError("No matches between txn.product_id and labels.product_id")

    if weight_method == "amt":
        base = joined["amt"]
    elif weight_method == "qty":
        base = joined["qty"]
    else:
        base = 1.0

    joined["contribution"] = base * (joined["score"] if apply_score else 1.0)

    agg = (
        joined.groupby(["customer_id", "intent_id", "intent_name"], as_index=False)
        .agg(intent_value=("contribution", "sum"),
             txn_count=("tx_id", "nunique"),
             last_tx_date=("tx_date", "max"))
    )

    totals = (
        joined.groupby("customer_id", as_index=False)
        .agg(customer_total_value=("contribution", "sum"),
             customer_txn_count=("tx_id", "nunique"))
    )

    agg = agg.merge(totals, on="customer_id", how="left")

    if normalize == "sum_to_1":
        agg["intent_weight"] = agg["intent_value"] / agg["customer_total_value"].replace({0: pd.NA})
        agg["intent_weight"] = agg["intent_weight"].fillna(0.0)
    else:
        agg["intent_weight"] = agg["intent_value"]

    agg = agg.sort_values(["customer_id", "intent_weight"], ascending=[True, False])
    agg["intent_rank"] = agg.groupby("customer_id").cumcount() + 1
    return agg.reset_index(drop=True)

# ============================================================================
# STEP 6: Audience Ranking (Campaign × Customer)
# ============================================================================
def rank_audience(camp_df: pd.DataFrame, cust_df: pd.DataFrame, top_explain=3):
    camp = camp_df.copy()
    cust = cust_df.copy()

    camp["weight"] = pd.to_numeric(camp["weight"], errors="coerce").fillna(0.0)
    cust["intent_weight"] = pd.to_numeric(cust["intent_weight"], errors="coerce").fillna(0.0)

    merged = camp.merge(cust, on="intent_id", suffixes=("_camp", "_cust"), how="inner")
    if merged.empty:
        raise ValueError("No matching intent_id between campaign profile and customer profile.")

    merged["score_contribution"] = merged["weight"] * merged["intent_weight"]

    scored = (
        merged.groupby(["campaign_id", "campaign_name", "customer_id"], as_index=False)
        .agg(match_score=("score_contribution", "sum"), matched_intents=("intent_id", "nunique"))
    )

    scored = scored.sort_values(["campaign_id", "match_score"], ascending=[True, False])
    scored["rank"] = scored.groupby("campaign_id").cumcount() + 1

    merged = merged.merge(scored[["campaign_id", "customer_id"]], on=["campaign_id", "customer_id"], how="inner")
    merged = merged.sort_values(["campaign_id", "customer_id", "score_contribution"], ascending=[True, True, False])
    merged["intent_contrib_rank"] = merged.groupby(["campaign_id", "customer_id"]).cumcount() + 1

    explain = merged[merged["intent_contrib_rank"] <= int(top_explain)].copy()
    explain["explain_piece"] = (
        explain["intent_name_camp"].astype(str)
        + f" (camp×cust="
        + explain["score_contribution"].round(6).astype(str)
        + ")"
    )

    explain_agg = (
        explain.groupby(["campaign_id", "customer_id"], as_index=False)
        .agg(explanation=("explain_piece", lambda s: " | ".join(list(s))))
    )

    out = scored.merge(explain_agg, on=["campaign_id", "customer_id"], how="left")
    return out.reset_index(drop=True)

# ============================================================================
# APP UI (LEAN)
# ============================================================================
st.subheader("Step 4 — Product → Intent Labels")
labels_upload = st.file_uploader("Upload product_intent_labels.csv (optional)", type=["csv"], key="lean_labels")
api_key_step4 = st.text_input("Gemini API Key (only if generating labels)", type="password", key="lean_gemini_step4")

load_labels = st.button("Load Labels (Upload)", key="lean_load_labels")
gen_labels = st.button("Generate Labels (LLM)", key="lean_gen_labels")

if load_labels:
    try:
        if labels_upload is None:
            raise ValueError("Upload product_intent_labels.csv first.")
        catalog_df = ss_get("catalog_df")
        df = load_product_intent_labels_csv(labels_upload, catalog_df=catalog_df)
        ss_set("product_intent_labels_df", df)
        st.success(f"Loaded labels: {len(df)} rows")
    except Exception as e:
        st.error(str(e))

if gen_labels:
    try:
        catalog_df = ss_get("catalog_df")
        dim_intent_df = ss_get("dim_intent_df")
        if catalog_df is None or dim_intent_df is None:
            raise ValueError("Need products (Step 1) + ontology (Step 2).")
        if not api_key_step4:
            raise ValueError("Provide Gemini API key.")
        df = generate_product_intent_labels_with_gemini(catalog_df.head(30), dim_intent_df, api_key_step4)  # lean default: 30 products
        ss_set("product_intent_labels_df", df)
        st.success(f"Generated labels: {len(df)} rows (lean default: first 30 products)")
    except Exception as e:
        st.error(str(e))

if ss_get("product_intent_labels_df") is not None:
    st.dataframe(ss_get("product_intent_labels_df").head(30), use_container_width=True)

st.divider()

st.subheader("Step 5 — Build Customer Intent Profiles")
build_profiles_btn = st.button("Build customer_intent_profile_df", key="lean_build_customer_profiles")

if build_profiles_btn:
    try:
        txn_df = ss_get("txn_df")
        labels_df = ss_get("product_intent_labels_df")
        if txn_df is None or labels_df is None:
            raise ValueError("Need txn_df (Step 1) + product_intent_labels_df (Step 4).")
        cust_df = build_customer_intent_profile(txn_df, labels_df, weight_method="amt", apply_score=True, normalize="sum_to_1")
        ss_set("customer_intent_profile_df", cust_df)
        st.success(f"Built customer_intent_profile_df: {len(cust_df)} rows")
    except Exception as e:
        st.error(str(e))

if ss_get("customer_intent_profile_df") is not None:
    st.dataframe(ss_get("customer_intent_profile_df").head(30), use_container_width=True)

st.divider()

st.subheader("Step 6 — Rank Campaign Audience")
rank_btn = st.button("Rank campaign_audience_ranked_df", key="lean_rank_audience")

if rank_btn:
    try:
        camp_df = ss_get("campaign_intent_profile_df")
        cust_df = ss_get("customer_intent_profile_df")
        if camp_df is None or cust_df is None:
            raise ValueError("Need campaign_intent_profile_df (Step 3) + customer_intent_profile_df (Step 5).")
        ranked = rank_audience(camp_df, cust_df, top_explain=3)
        ss_set("campaign_audience_ranked_df", ranked)
        st.success(f"Built campaign_audience_ranked_df: {len(ranked)} rows")
    except Exception as e:
        st.error(str(e))

if ss_get("campaign_audience_ranked_df") is not None:
    st.dataframe(ss_get("campaign_audience_ranked_df").head(50), use_container_width=True)
