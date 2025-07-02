
from flask import Flask, request, jsonify
import pickle
import numpy as np
import re, urllib.parse, html
from collections import Counter
from scipy.sparse import hstack

app = Flask(__name__)

# Load Models
with open('vectorizers/vectorizer.pkl','rb') as f:
    bin_vectorizer = pickle.load(f)
with open('models/classifier.pkl','rb') as f:
    bin_clf = pickle.load(f)
with open('models/anomaly_detector.pkl','rb') as f:
    ocsvm = pickle.load(f)

with open('vectorizers/tfidf_char.pkl','rb') as f:
    tfidf_char = pickle.load(f)
with open('vectorizers/tfidf_word.pkl','rb') as f:
    tfidf_word = pickle.load(f)
with open('vectorizers/count_vec.pkl','rb') as f:
    count_vec = pickle.load(f)
with open('vectorizers/scaler.pkl','rb') as f:
    scaler    = pickle.load(f)
with open('models/lightgbm_model.pkl','rb') as f:
    multi_clf = pickle.load(f)

#  SQLi / XSS regex lists
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

def preprocess_text(text):

    if not text: return ""
    s = text
    for _ in range(3):
        try:
            new = urllib.parse.unquote_plus(s)
            if new==s: break
            s=new
        except:
            break
    s = html.unescape(s)
    try:
        s = bytes(s,"utf-8").decode("unicode_escape")
    except:
        pass
    return ' '.join(s.split()).lower()

def extract_numeric_features(method, url, content, ua):

    fields = [method, url, content, ua]
    combined = ' '.join(preprocess_text(f) for f in fields)
    feats = {}
    feats["request_length"] = len(combined)
    feats["url_length"]     = len(url)
    feats["content_length"] = len(content)
    feats["user_agent_length"] = len(ua)
    if combined:
        feats["special_char_ratio"] = sum(c in '!@#$%^&*()[]{}|\\:";\'<>?,./`~' for c in combined)/len(combined)
        feats["digit_ratio"] = sum(c.isdigit() for c in combined)/len(combined)
        feats["alpha_ratio"] = sum(c.isalpha() for c in combined)/len(combined)
        feats["space_ratio"] = combined.count(' ')/len(combined)
        feats["quote_count"]   = combined.count("'")+combined.count('"')
        feats["bracket_count"] = combined.count('<')+combined.count('>')
        feats["semicolon_count"] = combined.count(';')
        feats["equals_count"] = combined.count('=')
        feats["ampersand_count"] = combined.count('&')
    else:
        for k in ["special_char_ratio","digit_ratio","alpha_ratio","space_ratio",
                  "quote_count","bracket_count","semicolon_count","equals_count","ampersand_count"]:
            feats[k] = 0
    feats["url_param_count"] = url.count('&') + url.count('=')
    feats["url_query_length"] = len(url.split('?')[-1]) if '?' in url else 0
    feats["url_path_depth"] = url.count('/')
    feats["sqli_pattern_score"] = sum(bool(re.search(p,combined)) for p in sqli_patterns)
    feats["xss_pattern_score"]  = sum(bool(re.search(p,combined)) for p in xss_patterns)
    # entropy
    if combined:
        cnt = Counter(combined)
        total = len(combined)
        feats["entropy"] = -sum((c/total)*np.log2(c/total) for c in cnt.values())
    else:
        feats["entropy"] = 0
    return np.array([feats[k] for k in [
        "request_length","url_length","content_length","user_agent_length",
        "special_char_ratio","digit_ratio","alpha_ratio","space_ratio",
        "quote_count","bracket_count","semicolon_count","equals_count","ampersand_count",
        "url_param_count","url_query_length","url_path_depth",
        "sqli_pattern_score","xss_pattern_score","entropy"
    ]])

def build_feature_text(data):
    return f"{data.get('method','')} {data.get('url','')} {data.get('content','')} {data.get('user_agent','')}"

@app.route('/predict_binary', methods=['POST'])
def predict_binary():
    data = request.get_json(force=True)
    text = build_feature_text(data)
    Xb = bin_vectorizer.transform([text]).toarray()
    lbl = int(bin_clf.predict(Xb)[0])
    return jsonify({'label': lbl})

@app.route('/predict_anomaly', methods=['POST'])
def predict_anomaly():
    data = request.get_json(force=True)
    text = build_feature_text(data)
    Xb = bin_vectorizer.transform([text]).toarray()
    Xs = ocsvm.decision_function(Xb) if hasattr(ocsvm,'decision_function') else ocsvm._decision_function(Xb)
    return jsonify({'anomaly_score': float(-Xs[0])})

@app.route('/predict_multiclass', methods=['POST'])
def predict_multiclass():
    data = request.get_json(force=True)
    m, u, c, ua = data['method'], data['url'], data['content'], data['user_agent']
    # text features
    tc = tfidf_char.transform([preprocess_text(f"{m} {u} {c} {ua}")])
    tw = tfidf_word.transform([preprocess_text(f"{m} {u} {c} {ua}")])
    ct = count_vec.transform([preprocess_text(f"{m} {u} {c} {ua}")])

    # numeric features
    num = extract_numeric_features(m,u,c,ua).reshape(1,-1)
    num_s = scaler.transform(num)

    # combine
    Xmc = hstack([tc, tw, ct, num_s])
    lbl = int(multi_clf.predict(Xmc)[0])
    return jsonify({'label': lbl})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
