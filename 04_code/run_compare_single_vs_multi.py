import os
import json
import re
import numpy as np
from datetime import datetime
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ===============================
# LOAD CONFIG
# ===============================
with open("config.json", "r", encoding="utf-8") as f:
    CFG = json.load(f)

MODEL = CFG["model"]
TEMPERATURES = [float(t) for t in CFG["temperatures"]]
N_REPEATS = int(CFG["n_repeats"])
TAU = float(CFG["tau_grounding"])
QUESTION_ID = CFG["question_id"]

RUN_ROOT = CFG["run_root"]
RUN_NAME = f"{CFG['experiment_name']}_COMPARE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR = os.path.join(RUN_ROOT, RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)

RUNS_DIR = os.path.join(RUN_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# ===============================
# CLIENTS
# ===============================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ===============================
# LOAD KEYWORDS
# ===============================
with open("00_protocol/rubric_keywords_v1.json", "r", encoding="utf-8") as f:
    RUBRIC_KEYWORDS = json.load(f)

EXPECTED_KEYWORDS = RUBRIC_KEYWORDS.get(QUESTION_ID, {})

# ===============================
# LOAD ANSWERS
# ===============================
with open("00_protocol/pilot_answers_v1.json", "r", encoding="utf-8") as f:
    DATA = json.load(f)

question_text = DATA["question_text"]
rubric = DATA["rubric"]
answers = DATA["answers"]

# ===============================
# TRACE-ED C — Grounding
# ===============================
ABSENCE_CUES = ["tidak", "belum", "kurang", "terbatas", "tanpa", "tidak menyebut"]

def is_absence_claim(claim: str):
    c = claim.lower()
    return any(cue in c for cue in ABSENCE_CUES)

def absence_grounded(indicator_id, answer, threshold=0.5):
    expected = EXPECTED_KEYWORDS.get(indicator_id, [])
    if not expected:
        return False
    ans = answer.lower()
    present = sum(1 for kw in expected if kw.lower() in ans)
    missing = len(expected) - present
    return (missing / len(expected)) >= threshold

def compute_grounding(explanation_text, student_answer):
    claims = re.findall(r"^\s*CLAIM\s*:\s*(.+)\s*$",
                        explanation_text,
                        flags=re.MULTILINE)

    sent_segments = [s.strip()
                     for s in re.split(r"[.?!]\s*", student_answer)
                     if s.strip()]

    tokens = [t for t in re.findall(r"\w+|[^\w\s]",
                                    student_answer,
                                    flags=re.UNICODE) if t.strip()]

    window_size = 8
    win_segments = [
        " ".join(tokens[i:i+window_size])
        for i in range(0, max(1, len(tokens)-window_size+1))
    ]

    segments, seen = [], set()
    for s in sent_segments + win_segments:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            segments.append(s)

    if not claims or not segments:
        return 0.0

    claim_emb = embedding_model.encode(claims, normalize_embeddings=True)
    seg_emb = embedding_model.encode(segments, normalize_embeddings=True)

    sims = claim_emb @ seg_emb.T

    claim_indicator_ids = [r["indicator_id"] for r in rubric][:len(claims)]
    grounded_flags = []

    for i, claim in enumerate(claims):
        best_sim = float(np.max(sims[i]))

        if is_absence_claim(claim):
            grounded = (
                best_sim >= TAU or
                absence_grounded(claim_indicator_ids[i], student_answer)
            )
        else:
            grounded = best_sim >= TAU

        grounded_flags.append(grounded)

    return sum(grounded_flags) / len(grounded_flags)

# ===============================
# TRACE-ED E — Coherence
# ===============================
POSITIVE_WORDS = ["tepat", "baik", "benar", "lengkap", "sesuai", "jelas", "akurat"]
NEGATIVE_WORDS = ["kurang", "tidak", "belum", "terbatas", "lemah", "salah"]

def lexical_valence(text):
    text = text.lower()
    pos = sum(text.count(w) for w in POSITIVE_WORDS)
    neg = sum(text.count(w) for w in NEGATIVE_WORDS)
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total

def score_to_valence(score, max_score):
    return 2 * (score / max_score) - 1

def coherence(S_prime, s_val):
    return 1 - (abs(S_prime - s_val) / 2)

def compute_coherence(explanation_text):
    pattern = r"INDICATOR:\s*(I\d+)\s*\nSCORE:\s*(\d+)\s*/\s*(\d+)\s*\nCLAIM:\s*(.+)"
    matches = re.findall(pattern, explanation_text)

    coherences, contradictions = [], []

    for indicator_id, score, max_score, claim in matches:
        S_prime = score_to_valence(int(score), int(max_score))
        s_val = lexical_valence(claim)
        cs = coherence(S_prime, s_val)

        coherences.append(cs)
        contradictions.append(
            (S_prime >= 0.5 and s_val < 0) or
            (S_prime <= -0.5 and s_val > 0)
        )

    mean_cs = float(np.mean(coherences)) if coherences else 0.0
    contr_rate = float(sum(contradictions)/len(contradictions)) if contradictions else 0.0
    return mean_cs, contr_rate

def parse_total_score(scoring_json):
    total = scoring_json.get("total_score")
    if total is None and "indicators" in scoring_json:
        total = sum(int(x.get("score", 0)) for x in scoring_json["indicators"])
    return total

# ===============================
# PROMPTS
# ===============================
system_msg_score = (
    "You are a STRICT grading agent.\n"
    "You MUST return valid JSON only.\n"
    "No markdown. No explanation. No commentary.\n"
    "JSON format must be:\n"
    "{\n"
    '  "question_id": "<id>",\n'
    '  "indicators": [\n'
    '    {"indicator_id": "I1", "score": <int>, "max_score": <int>},\n'
    '    ...\n'
    "  ],\n"
    '  "total_score": <int>,\n'
    '  "max_total_score": <int>\n'
    "}\n"
    "All fields are mandatory."
)

system_msg_exp = (
    "You are an Explanation Agent.\n"
    "OUTPUT MUST FOLLOW FORMAT EXACTLY.\n"
    "Write in Indonesian.\n\n"
    "QUESTION_ID: <id>\n\n"
    "INDICATOR: I1\n"
    "SCORE: <x>/<max>\n"
    "CLAIM: <one sentence>\n"
    "JUSTIFICATION: <short reasoning>\n\n"
    "(repeat)\n\n"
    "OVERALL_SUMMARY:\n"
    "<2-3 sentences>\n"
)

system_msg_single = (
    "You must score AND explain.\n"
    "OUTPUT FORMAT:\n"
    "RAW JSON OUTPUT:\n"
    "{valid JSON}\n\n"
    "QUESTION_ID: ...\n"
    "(then same explanation structure as above)"
)

def split_single_output(text):
    m = re.search(r"RAW JSON OUTPUT:\s*(\{.*?\})\s*QUESTION_ID:",
                  text,
                  flags=re.DOTALL)
    if not m:
        return None, text

    json_str = m.group(1)
    rest = text[m.end()-len("QUESTION_ID:"):]

    try:
        scoring = json.loads(json_str)
    except:
        scoring = None

    return scoring, rest

# ===============================
# RUN EXPERIMENT
# ===============================
records = []
ARCHS = ["single", "multi"]

for arch in ARCHS:
    for ans in answers:
        answer_id = ans["answer_id"]
        student_answer = ans["text"]

        for temp in TEMPERATURES:
            for rep in range(1, N_REPEATS + 1):

                temp_tag = f"{temp:.1f}"
                run_id = f"{arch}_{answer_id}_temp_{temp_tag}_rep_{rep}"
                run_path = os.path.join(RUNS_DIR, run_id)
                os.makedirs(run_path, exist_ok=True)

                if arch == "multi":

                    resp = client.chat.completions.create(
                        model=MODEL,
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": system_msg_score},
                            {"role": "user", "content": json.dumps({
                                "question_id": QUESTION_ID,
                                "question_text": question_text,
                                "rubric": rubric,
                                "student_answer": student_answer
                            }, ensure_ascii=False)}
                        ]
                    )

                    raw_text = resp.choices[0].message.content.strip()

                    try:
                        scoring_json = json.loads(raw_text)
                    except Exception:
                        # Attempt to extract JSON block if extra text exists
                        m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
                        if m:
                            try:
                                scoring_json = json.loads(m.group(0))
                            except Exception:
                                scoring_json = {"question_id": QUESTION_ID, "indicators": []}
                        else:
                            scoring_json = {"question_id": QUESTION_ID, "indicators": []}


                    resp_exp = client.chat.completions.create(
                        model=MODEL,
                        temperature=temp,
                        messages=[
                            {"role": "system", "content": system_msg_exp},
                            {"role": "user", "content": json.dumps({
                                "question_id": QUESTION_ID,
                                "question_text": question_text,
                                "rubric": rubric,
                                "student_answer": student_answer,
                                "scoring_json": scoring_json
                            }, ensure_ascii=False)}
                        ]
                    )

                    explanation_text = resp_exp.choices[0].message.content

                else:

                    resp = client.chat.completions.create(
                        model=MODEL,
                        temperature=temp,
                        messages=[
                            {"role": "system", "content": system_msg_single},
                            {"role": "user", "content": json.dumps({
                                "question_id": QUESTION_ID,
                                "question_text": question_text,
                                "rubric": rubric,
                                "student_answer": student_answer
                            }, ensure_ascii=False)}
                        ]
                    )

                    full_text = resp.choices[0].message.content
                    scoring_json, explanation_text = split_single_output(full_text)

                    if scoring_json is None:
                        scoring_json = {"question_id": QUESTION_ID, "indicators": []}

                    with open(os.path.join(run_path, "full_output.txt"),
                              "w", encoding="utf-8") as f:
                        f.write(full_text)

                # Save outputs
                with open(os.path.join(run_path, "scoring.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(scoring_json, f, indent=2, ensure_ascii=False)

                with open(os.path.join(run_path, "explanation.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(explanation_text)

                # Metrics
                gr = compute_grounding(explanation_text, student_answer)
                mean_cs, contr_rate = compute_coherence(explanation_text)
                total_score = parse_total_score(scoring_json)

                records.append({
                    "answer_id": answer_id,
                    "arch": arch,
                    "temperature": temp,
                    "repeat": rep,
                    "total_score": total_score,
                    "grounding_rate": gr,
                    "mean_coherence": mean_cs,
                    "contradiction_rate": contr_rate
                })

# ===============================
# SAVE RESULTS
# ===============================
with open(os.path.join(RUN_DIR, "records.json"),
          "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

multi_subset = [r for r in records if r["arch"] == "multi"]
missing_scores = sum(1 for r in multi_subset if r["total_score"] is None)
print("Multi-agent missing total_score count:", missing_scores)

# ===============================
# ICC
# ===============================
def icc_1k(X):
    X = np.array(X, dtype=float)
    if X.ndim != 2:
        return None

    n, k = X.shape
    if n < 2 or k < 2:
        return None

    row_means = X.mean(axis=1, keepdims=True)
    grand_mean = X.mean()

    SSR = k * np.sum((row_means - grand_mean) ** 2)
    SSE = np.sum((X - row_means) ** 2)

    MSR = SSR / (n - 1)
    MSE = SSE / (n * (k - 1))

    denom = MSR + (k - 1) * MSE
    if denom == 0:
        return None

    return float((MSR - MSE) / denom)

print("\n=== ICC REPORT (total_score) ===")

for arch in ARCHS:
    subset = [r for r in records if r["arch"] == arch]
    answer_ids = sorted({r["answer_id"] for r in subset})

    k_expected = len(TEMPERATURES) * N_REPEATS
    matrix = []

    for aid in answer_ids:
        row = [r["total_score"]
               for r in subset
               if r["answer_id"] == aid]

        if len(row) == k_expected and all(v is not None for v in row):
            matrix.append(row)

    if len(matrix) >= 2:
        icc_val = icc_1k(np.array(matrix))
        if icc_val is None:
            print(f"ICC(1,k) total_score — {arch}: Undefined (zero variance or perfect stability)")
        else:
            print(f"ICC(1,k) total_score — {arch}: {icc_val:.4f}")
    else:
        print(f"ICC(1,k) total_score — {arch}: Not enough data")


print("\nCOMPARE run saved to:", RUN_DIR)
print("Total runs:", len(records))