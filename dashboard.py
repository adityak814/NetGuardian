# dashboard.py
import streamlit as st
import pandas as pd
import random
import requests
import re

st.set_page_config(page_title="SQLi & XSS Attack Detection", layout="wide")
st.title("🔍 Web Attack Detection Dashboard")

# ── 1) Load & label the CSIC dataset ──
df = pd.read_csv("csic_database.csv")
df["text"] = (
    df["Method"].fillna("") + " " +
    df["URL"].fillna("") + " " +
    df["content"].fillna("") + " " +
    df["User-Agent"].fillna("")
)

def detect_attack_type(text):
    sqli_patterns = [
        r"(?i)(\bUNION\b.*\bSELECT\b|\bSELECT\b.*\bUNION\b)",
        r"(?i)(1\s*=\s*1|0\s*=\s*0|\btrue\b|\bfalse\b)",
        r"(?i)(\'\s*=\s*\'|\"\s*=\s*\")",
        r"(--|\#|\/\*|\*\/)",
        r"(?i)\bexec(ute)?\s*[\(\s]",
        r"(?i)(char\(|ascii\(|substring\(|length\(|version\()",
        r"(?i)(waitfor\s+delay|benchmark\(|sleep\()",
        r"(?i)(extractvalue\(|updatexml\(|exp\()",
        r";\s*(drop|insert|update|delete|create|alter)",
        r"[\'\"][^\'\"]*[\'\"]",
        r"(?i)\bhaving\b.*\bcount\b",
        r"(?i)\border\s+by\b.*\d+"
    ]
    xss_patterns = [
        r"(?i)<\s*script[^>]*>.*?</script>",
        r"(?i)<\s*script[^>]*>",
        r"(?i)on(load|error|click|mouseover|focus|blur|change|submit)\s*=",
        r"(?i)javascript\s*:",
        r"(?i)&\#(x)?[0-9a-f]+;",
        r"(?i)<\s*img[^>]*src\s*=\s*[\"']?javascript:",
        r"(?i)<\s*iframe[^>]*>",
        r"(?i)<\s*(object|embed)[^>]*>",
        r"(?i)(alert|confirm|prompt)\s*\(",
        r"(?i)document\.(write|writeln|cookie)",
        r"(?i)(expression\(|eval\()",
        r"(?i)<\s*meta[^>]*refresh"
    ]
    if any(re.search(p, text) for p in sqli_patterns):
        return 1
    if any(re.search(p, text) for p in xss_patterns):
        return 2
    if text.strip() != "":
        return 3
    return 0

df["attack_type"] = df.apply(
    lambda r: 0 if r.classification == 0 else detect_attack_type(r.text),
    axis=1
)

# ── 2) Sampling helper (up to n real samples) ──
def sample_rows(mask, label, n=5):
    subset = df[mask]
    k = min(len(subset), n)
    if k == 0:
        return []
    chosen = subset.sample(k, random_state=42)
    out = []
    for _, r in chosen.iterrows():
        out.append({
            "method":  r.Method,
            "url":     r.URL,
            "content": r.content if isinstance(r.content, str) else "",
            "ua":      r["User-Agent"],
            "category": label
        })
    return out

# ── 3) Build 5 samples per category ──
benign = sample_rows(df.classification == 0, "Benign", n=5)
sqli   = sample_rows(df.attack_type    == 1, "SQLi",   n=5)
xss    = sample_rows(df.attack_type    == 2, "XSS",    n=5)
other  = sample_rows(df.attack_type    == 3, "Other",  n=5)

# ── 4) Sidebar: availability & four buttons ──
st.sidebar.markdown("### Available Samples")
st.sidebar.write(f"- Benign: {len(benign)}/5")
st.sidebar.write(f"- SQLi:   {len(sqli)}/5")
st.sidebar.write(f"- XSS:    {len(xss)}/5")
st.sidebar.write(f"- Other:  {len(other)}/5")

st.sidebar.markdown("---")
btn_b = st.sidebar.button("🎲 Random Benign Sample")
btn_s = st.sidebar.button("🎲 Random SQLi Sample")
btn_x = st.sidebar.button("🎲 Random XSS Sample")
btn_o = st.sidebar.button("🎲 Random Other Attack Samples")

# pick the right list
if btn_b:
    chosen = benign
elif btn_s:
    chosen = sqli
elif btn_x:
    chosen = xss
elif btn_o:
    chosen = other
else:
    chosen = None

if chosen:
    s = random.choice(chosen)
    st.markdown(f"### 🔸 Category: **{s['category']}**")
    st.code(f"{s['method']} {s['url']}", language="http")
    if s["content"]:
        st.markdown(f"**Body:** `{s['content']}`")
    st.markdown(f"**User-Agent:** `{s['ua']}`")

    payload = {
        "method":     s['method'],
        "url":        s['url'],
        "content":    s['content'],
        "user_agent": s['ua']
    }

    PROXY = "http://localhost:5000"

    # — Anomaly Detection —
    try:
        r = requests.post(f"{PROXY}/predict_anomaly", json=payload); r.raise_for_status()
        score = r.json()['anomaly_score']
        st.subheader("🚨 Anomaly Detection")
        # st.write(f"**Anomaly Score:** {score:.4f}")
        # st.write("Higher scores indicate more deviation from normal traffic patterns.")
        st.write(f"**Anomaly Score:** {score:.4f}   (threshold = 0.0)")
        if score > 0:
            st.error("🚨 This is flagged as an anomaly (score > 0).")
        else:
            st.success("✅ This is within normal bounds (score ≤ 0).")
    except Exception as e:
        st.error(f"Anomaly detection error: {e}")

    # — Binary Classification —
    try:
        r = requests.post(f"{PROXY}/predict_binary", json=payload); r.raise_for_status()
        lbl = r.json()['label']
        st.subheader("🛡️ Binary Classification")
        if lbl == 0:
            st.success("✅ Normal traffic")
        else:
            st.error("⚠️ Attack detected")

    except Exception as e:
        st.error(f"Binary classification error: {e}")

    # — Multi-Class Classification —
    try:
        r = requests.post(f"{PROXY}/predict_multiclass", json=payload); r.raise_for_status()
        m_lbl = r.json()['label']
        names = {0:"Normal",1:"SQL Injection",2:"XSS",3:"Other"}
        desc  = {
            0: "Normal application traffic",
            1: "SQL Injection attempt",
            2: "XSS attempt",
            3: "Other anomalous request"
        }
        st.subheader("🔢 Multi-Class Classification")
        if m_lbl == 0:
            st.success(f"✅ {names[m_lbl]} traffic")
        elif m_lbl in [1, 2]:
            st.error(f"⚠️ {names[m_lbl]} attempt")
        elif m_lbl == 3:
            st.error(f"⚠️ {names[m_lbl]} anomalous request")
    except Exception as e:
        st.error(f"Multi-class classification error: {e}")
