"""
HOPE-EXP @ IberLEF 2026 — Full solution (all 4 tracks)
=======================================================
Arquitectura en dos capas:

  Capa 1 — ML clásico (siempre se ejecuta, offline, rápido):
    • Task A  — TF-IDF + LogisticRegression(C=5, balanced)   → ~0.84 macro-F1 CV
    • Task B  — TF-IDF + OVR-LogisticRegression              → ~0.63 macro-F1 CV
    • Task C  — Extracción de spans por reglas + sentence-split
    • Task D  — Reglas léxicas para stance y actor

  Capa 2 — LLM (opcional, mejora Tasks A/B/C/D):
    • Reemplaza la predicción ML cuando el LLM responde correctamente
    • Backends: ollama-local | ollama-web | huggingface
    • Activar con --llm-backend

Uso:
  # Solo ML (sin LLM, rápido, reproducible)
  python hope_exp_solver.py

  # ML + LLM para refinar
  python hope_exp_solver.py --llm-backend ollama-local --llm-model llama3.1
  python hope_exp_solver.py --llm-backend ollama-web   --llm-model llama3.3 --api-key KEY
  python hope_exp_solver.py --llm-backend huggingface  --llm-model mistralai/Mistral-7B-Instruct-v0.3 --api-key hf_TOKEN

  # Evaluar sobre muestra del train
  python hope_exp_solver.py --eval 200
  python hope_exp_solver.py --llm-backend ollama-local --llm-model llama3.1 --eval 200
"""

import os, json, re, zipfile, time, argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_FILE = "HopeEXP_Train.jsonl"
TEST_FILE  = "HopeEXP_Test_unlabeled.jsonl"
OUTPUT_DIR = Path("outputs_hllm")

TEAM_NAME  = "TeamName"
RUN_NUMBER = 1

# ── Valid label sets (case-sensitive) ─────────────────────────────────────────
VALID_PRIMARY = [
    "General Hope", "Realistic Hope", "Unrealistic Hope",
    "Sarcastic Hope", "Hopelessness", "Not Hope"
]
HOPE_LABELS = {"General Hope","Realistic Hope","Unrealistic Hope","Sarcastic Hope"}

EMOTIONS_ALL = ["sadness","joy","love","anger","fear","surprise","Nuetral/unclear"]
VALID_STANCE = {"Desired","Avoided"}
VALID_ACTOR  = {"Self","Other","World/System","Unclear"}

# ══════════════════════════════════════════════════════════════════════════════
#  I/O  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_jsonl(path: str) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def build_text(row: dict) -> str:
    return (str(row.get("title") or "") + "\n\n" +
            str(row.get("selftext") or "")).strip()

# ══════════════════════════════════════════════════════════════════════════════
#  NORMALIZATION HELPERS  (shared by ML + LLM layers)
# ══════════════════════════════════════════════════════════════════════════════
_LABEL_MAP = {re.sub(r"[^a-z]", "", l.lower()): l for l in VALID_PRIMARY}
_LABEL_MAP.update({
    "nothope":"Not Hope", "nope":"Not Hope",
    "generalhope":"General Hope",
    "realistichope":"Realistic Hope",
    "unrealistichope":"Unrealistic Hope",
    "sarcastichope":"Sarcastic Hope",
    "hopelessness":"Hopelessness",
})

_EMOTION_MAP = {e.lower(): e for e in EMOTIONS_ALL}
_EMOTION_MAP.update({
    "neutral/unclear":"Nuetral/unclear", "neutral":"Nuetral/unclear",
    "nuetral":"Nuetral/unclear", "unclear":"Nuetral/unclear",
    "happiness":"joy","happy":"joy","hate":"anger",
    "anxiety":"fear","disgust":"anger","worried":"fear",
})

_STANCE_MAP = {s.lower(): s for s in VALID_STANCE}
_ACTOR_MAP  = {a.lower(): a for a in VALID_ACTOR}
_ACTOR_MAP.update({
    "world/system":"World/System","world":"World/System",
    "system":"World/System","society":"World/System",
    "government":"World/System","others":"Other",
})

def norm_label(label: str) -> str:
    return _LABEL_MAP.get(re.sub(r"[^a-z]", "", str(label).lower()), "Not Hope")

def norm_emotions(emotions) -> list:
    if not isinstance(emotions, list) or not emotions:
        return ["Nuetral/unclear"]
    out = []
    for e in emotions:
        n = _EMOTION_MAP.get(str(e).lower().strip())
        if n and n not in out:
            out.append(n)
    if not out:
        return ["Nuetral/unclear"]
    # drop Neutral/unclear when real emotions present
    if "Nuetral/unclear" in out and len(out) > 1:
        out = [x for x in out if x != "Nuetral/unclear"]
    return out

def validate_spans(spans: list, title: str, selftext: str) -> list:
    """Keep ≤3 spans that are exact substrings of the source text."""
    if not isinstance(spans, list):
        return []
    full      = f"{title}\n{selftext}"
    full_norm = re.sub(r"\s+", " ", full)
    valid = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        txt = str(s.get("span","")).strip()
        if not txt:
            continue
        if txt in full:
            matched = txt
        else:
            n = re.sub(r"\s+", " ", txt)
            if n not in full_norm:
                continue
            matched = n
        stance = _STANCE_MAP.get(str(s.get("outcome_stance","")).lower().strip(), "Desired")
        actor  = _ACTOR_MAP.get(str(s.get("actor","")).lower().strip(), "Unclear")
        valid.append({"span": matched, "outcome_stance": stance, "actor": actor})
        if len(valid) == 3:
            break
    return valid

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1-A  Task A & B — TF-IDF + Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════
class MLModels:
    """
    Trains Task A (primary label) and Task B (emotions) with TF-IDF + LR.
    CV performance (5-fold):
      Task A  macro-F1 ≈ 0.84  (LR C=5, class_weight=balanced, ngram 1-2)
      Task B  macro-F1 ≈ 0.63  (OVR-LR C=5, balanced)
    """

    def __init__(self):
        self.vec   = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=80_000,
            sublinear_tf=True,
            analyzer="word",
            strip_accents=None,     # keep accented chars for Spanish
        )
        self.clf_A = LogisticRegression(
            C=5, max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        self.mlb   = MultiLabelBinarizer(classes=EMOTIONS_ALL)
        self.clf_B = OneVsRestClassifier(
            LogisticRegression(C=5, max_iter=1000, class_weight="balanced")
        )

    def fit(self, rows: list) -> "MLModels":
        texts   = [build_text(r) for r in rows]
        labels  = [r["primary_label"] for r in rows]
        emotions = [norm_emotions(r.get("trigger_emotions", [])) for r in rows]

        X  = self.vec.fit_transform(texts)
        self.clf_A.fit(X, labels)

        Y = self.mlb.fit_transform(emotions)
        self.clf_B.fit(X, Y)
        return self

    def predict(self, rows: list) -> list:
        """Returns list of (primary_label, [emotions]) tuples."""
        texts = [build_text(r) for r in rows]
        X     = self.vec.transform(texts)

        pred_A = self.clf_A.predict(X).tolist()

        pred_B_bin = self.clf_B.predict(X)
        pred_B = []
        for row in pred_B_bin:
            labs = [self.mlb.classes_[i] for i, v in enumerate(row) if v == 1]
            pred_B.append(norm_emotions(labs))

        return list(zip(pred_A, pred_B))

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1-B  Tasks C & D — Rule-based span extraction + actor/stance
# ══════════════════════════════════════════════════════════════════════════════

# ── Cue patterns (EN + ES) ────────────────────────────────────────────────────
_HOPE_CUES_EN = [
    r"\bi\s+hope\b", r"\bhoping\b", r"\bhopefully\b",
    r"\bi\s+wish\b", r"\bwishing\b",
    r"\bi\s+want\b", r"\bi\s+'d\s+love\b", r"\bi\s+would\s+love\b",
    r"\bpraying\b", r"\bpraying\s+for\b",
    r"\bi\s+expect\b", r"\bi\s+plan\s+to\b",
    r"\bmy\s+goal\b", r"\bmy\s+dream\b",
    r"\bif\s+only\b", r"\bi\s+can't\s+wait\b",
    r"\bpls\b", r"\bplease\b",
]
_HOPE_CUES_ES = [
    r"\bespero\s+que\b", r"\bespero\b", r"\besperar\b",
    r"\bojal[aá]\b", r"\bdeseo\s+que\b", r"\bdeseo\b",
    r"\bquiero\s+que\b", r"\bquiero\b",
    r"\bme\s+gustar[íi]a\b", r"\bsuer[t]e\s+que\b",
    r"\bpido\s+que\b", r"\bpido\b",
    r"\bcon\s+suerte\b", r"\bmi\s+sue[ñn]o\b",
]
_ALL_CUES = _HOPE_CUES_EN + _HOPE_CUES_ES

# ── Sentence splitter ─────────────────────────────────────────────────────────
_SENT_SPLIT = re.compile(
    r'(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÜÑ"\'])'      # EN/ES sentence boundary
    r'|(?<=\.)\s*\n+\s*'                        # newline after period
    r'|\n{2,}',                                 # blank line
    re.UNICODE
)

def _split_sentences(text: str) -> list:
    sents = _SENT_SPLIT.split(text)
    return [s.strip() for s in sents if len(s.strip()) >= 5]

# ── Negation / avoided cues ───────────────────────────────────────────────────
_NEG = re.compile(
    r"\b(not|don't|dont|doesn't|doesnt|never|no\b|won't|wont|"
    r"avoid|prevent|stop|ruin|die|fail|lose|sin que|que no|evitar|impedir)\b",
    re.IGNORECASE
)

# ── Actor keyword sets ────────────────────────────────────────────────────────
_SELF_RE  = re.compile(r"\b(i|i'm|im|i'll|ill|i will|my|me|myself|yo|mi|me\b|voy a)\b",
                        re.IGNORECASE)
_WORLD_KW = {
    "rain","weather","economy","inflation","rent","surgery","meds","medicine",
    "treatment","appointment","doctor","clinic","therapy","visa","approval",
    "government","election","system","society","market","insurance","company",
    "lluvia","clima","tiempo","economía","renta","cirugía","medicamento",
    "tratamiento","cita","gobierno","elección","sistema","sociedad","mercado",
}
_OTHER_RE = re.compile(
    r"\b(they|them|he|she|landlord|boss|manager|teacher|doctor|friend|"
    r"family|parent|partner|school|university|hospital|police|bank|"
    r"ellos|ella|él|jefe|maestro|familia|amigo|pareja|escuela|universidad)\b",
    re.IGNORECASE
)

def _stance_rule(span: str) -> str:
    return "Avoided" if _NEG.search(span) else "Desired"

def _actor_rule(span: str) -> str:
    s = span.lower().strip()
    if re.match(r"^(it|this|that|eso|esto|aquello)", s, re.IGNORECASE):
        return "Unclear"
    # World/System wins when domain keywords present (rain, economy, government)
    if any(k in s for k in _WORLD_KW):
        return "World/System"
    # Other wins when explicit third-person entity named
    if _OTHER_RE.search(s):
        return "Other"
    # Self: first-person references
    if _SELF_RE.search(s):
        return "Self"
    return "Unclear"
    if _SELF_RE.search(s):
        return "Self"
    if any(k in s for k in _WORLD_KW):
        return "World/System"
    if _OTHER_RE.search(s):
        return "Other"
    return "Unclear"

def _best_window(text: str, cue_match, window: int = 120) -> str:
    """Extract text window after cue match, up to next sentence boundary."""
    start = cue_match.start()
    tail  = text[cue_match.end(): cue_match.end() + window].strip(" :,—- \n\t")
    # cut at sentence boundary
    cut = re.search(r"[.!?\n]", tail)
    if cut and cut.start() > 10:
        tail = tail[:cut.start()].strip()
    # strip stray punctuation
    tail = tail.strip(" .,;:!?\"'()[]{}—-")
    return tail

def extract_spans_rules(title: str, selftext: str, primary_label: str,
                         max_spans: int = 3) -> list:
    """
    Task C+D:  extract ≤3 span dicts from the post.
    Strategy:
      1. Search for hope-cue patterns → extract the continuation window
      2. If not enough spans, fall back to sentences containing cue keywords
      3. Validate spans are exact substrings before returning
    """
    if primary_label not in HOPE_LABELS:
        return []

    full  = f"{title}\n{selftext}"
    # prefer body, but include title
    source_priority = selftext if len(selftext.strip()) > 10 else full
    sources = [source_priority, title]

    candidates = []  # list of raw string candidates

    # ── Pass 1: cue-continuation windows ──────────────────────────────────
    for src in sources:
        for pat in _ALL_CUES:
            for m in re.finditer(pat, src, re.IGNORECASE):
                window = _best_window(src, m)
                if len(window) >= 6:
                    candidates.append(window)

    # ── Pass 2: full sentences containing a cue ───────────────────────────
    if len(candidates) < max_spans:
        for src in sources:
            for sent in _split_sentences(src):
                for pat in _ALL_CUES:
                    if re.search(pat, sent, re.IGNORECASE):
                        candidates.append(sent.strip(" .!?"))
                        break

    # ── Deduplicate, filter, validate substring ───────────────────────────
    seen   = set()
    result = []
    for cand in candidates:
        cand = cand.strip()
        if not cand or len(cand) < 5:
            continue
        # must be exact substring of source
        if cand not in full:
            cand_norm = re.sub(r"\s+", " ", cand)
            full_norm = re.sub(r"\s+", " ", full)
            if cand_norm not in full_norm:
                continue
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "span":            cand,
            "outcome_stance":  _stance_rule(cand),
            "actor":           _actor_rule(cand),
        })
        if len(result) == max_spans:
            break

    return result

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — LLM BACKENDS  (optional, replaces ML predictions when available)
# ══════════════════════════════════════════════════════════════════════════════

# ── Prompts ───────────────────────────────────────────────────────────────────
_SYSTEM = """You are an expert annotator for the HOPE-EXP task at IberLEF 2026.
Analyze social media posts (English or Spanish) and return structured JSON.

## TASK A — primary_label (exactly ONE):
"Realistic Hope"   — desire for a PLAUSIBLE, achievable future outcome
"Unrealistic Hope" — desire for an IMPOSSIBLE or fantastical outcome
"General Hope"     — VAGUE optimism, no concrete defined outcome
"Hopelessness"     — absence of hope; pessimism or resignation
"Sarcastic Hope"   — IRONIC expression implying disbelief/criticism
"Not Hope"         — no forward-looking intent; neutral/factual/descriptive

## TASK B — trigger_emotions (ONE OR MORE):
sadness, joy, love, anger, fear, surprise, Nuetral/unclear

## TASK C — span_annotations:
Only for Hope categories. Up to 3 EXACT verbatim substrings from the post.
For "Not Hope" or "Hopelessness": []

## TASK D — per span:
outcome_stance: "Desired" or "Avoided"
actor: "Self" | "Other" | "World/System" | "Unclear"

Return ONLY valid JSON, no markdown, no explanation:
{"primary_label":"...","trigger_emotions":[...],"span_annotations":[{"span":"...","outcome_stance":"...","actor":"..."}]}
"""

_FEWSHOT = """\
EXAMPLE 1 (EN, Sarcastic Hope):
Title: The Silicon Valley Dream That Keeps Crashing
Text: ...I'm sure everything will work out fine in the end, because that's definitely how tech careers work out for everyone.
{"primary_label":"Sarcastic Hope","trigger_emotions":["sadness","anger"],"span_annotations":[{"span":"I'm sure everything will work out fine in the end, because that's definitely how tech careers work out for everyone.","outcome_stance":"Desired","actor":"Self"}]}

EXAMPLE 2 (EN, General Hope):
Title: Parenting while drowning
Text: Some days I just hope things will somehow improve without me having to figure out how.
{"primary_label":"General Hope","trigger_emotions":["sadness"],"span_annotations":[{"span":"I just hope things will somehow improve without me having to figure out how","outcome_stance":"Desired","actor":"World/System"}]}

EXAMPLE 3 (ES, Not Hope):
Title: Cómo es el plan de estudios de la Maestría?
Text: Not Hope
{"primary_label":"Not Hope","trigger_emotions":["Nuetral/unclear"],"span_annotations":[]}

EXAMPLE 4 (EN, Hopelessness):
Title: Nothing ever changes
Text: I've given up expecting things to get better. There's no point.
{"primary_label":"Hopelessness","trigger_emotions":["sadness","fear"],"span_annotations":[]}

EXAMPLE 5 (ES, Realistic Hope):
Title: Quiero mejorar mi español
Text: Espero poder mantener una conversación fluida antes de que acabe el año.
{"primary_label":"Realistic Hope","trigger_emotions":["joy"],"span_annotations":[{"span":"Espero poder mantener una conversación fluida antes de que acabe el año","outcome_stance":"Desired","actor":"Self"}]}
"""


class BackendBase:
    def __init__(self, model, api_key=None, temperature=0.1, timeout=90):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system: str, user: str) -> Optional[str]:
        raise NotImplementedError

    def parse_json(self, raw: str) -> Optional[dict]:
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw).strip()
        m   = re.search(r"\{[\s\S]*\}", raw)
        if m:
            raw = m.group(0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None


class OllamaBackend(BackendBase):
    _URLS = {
        "local": "http://localhost:11434/api/chat",
        "web":   "https://api.ollama.com/api/chat",
    }

    def __init__(self, model, mode="local", api_key=None,
                 base_url=None, temperature=0.1, timeout=90):
        super().__init__(model, api_key, temperature, timeout)
        self.url = base_url or self._URLS.get(mode, self._URLS["local"])

    def complete(self, system, user):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model, "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "options": {"temperature": self.temperature, "num_predict": 600},
        }
        try:
            r = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.ConnectionError:
            print(f"    [Ollama] Cannot connect to {self.url}")
        except requests.HTTPError as e:
            print(f"    [Ollama {e.response.status_code}] {e.response.text[:150]}")
        except (KeyError, ValueError) as e:
            print(f"    [Ollama parse] {e}")
        return None

    def list_models(self):
        url = self.url.replace("/api/chat", "/api/tags")
        h   = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            return [m["name"] for m in
                    requests.get(url, headers=h, timeout=8).json().get("models", [])]
        except Exception:
            return []


class HuggingFaceBackend(BackendBase):
    _CHAT = "https://router.huggingface.co/hf-inference/models/{}/v1/chat/completions"
    _TEXT = "https://router.huggingface.co/hf-inference/models/{}/v1/completions"

    RECOMMENDED = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "meta-llama&/Meta-Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "HuggingFaceH4/zephyr-7b-beta",
    ]

    def __init__(self, model, api_key, use_chat=True, temperature=0.1, timeout=120):
        super().__init__(model, api_key, temperature, timeout)
        self.use_chat = use_chat
        self._h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def complete(self, system, user):
        return self._chat(system, user) if self.use_chat else self._text(system, user)

    def _chat(self, system, user):
        url = self._CHAT.format(self.model)
        payload = {
            "model": self.model,
            "messages": [{"role":"system","content":system},
                         {"role":"user","content":user}],
            "max_tokens": 600, "temperature": self.temperature, "stream": False,
        }
        try:
            r = requests.post(url, headers=self._h, json=payload, timeout=self.timeout)
            if r.status_code == 404:
                print("    [HF] Switching to text-gen endpoint...")
                self.use_chat = False
                return self._text(system, user)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            print(f"    [HF {e.response.status_code}] {e.response.text[:200]}")
        except (KeyError, ValueError, requests.RequestException) as e:
            print(f"    [HF] {e}")
        return None

    def _text(self, system, user):
        url  = self._TEXT.format(self.model)
        body = f"<s>[INST] {system}\n\n{user} [/INST]"
        payload = {
            "inputs": body,
            "parameters": {
                "max_new_tokens": 600,
                "temperature": max(self.temperature, 0.01),
                "return_full_text": False, "do_sample": True,
            },
            "options": {"wait_for_model": True, "use_cache": False},
        }
        try:
            r = requests.post(url, headers=self._h, json=payload, timeout=self.timeout)
            if r.status_code == 503:
                wait = r.json().get("estimated_time", 25)
                print(f"    [HF] Loading model ({wait:.0f}s)…")
                time.sleep(min(float(wait), 45))
                r = requests.post(url, headers=self._h, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return (data[0] if isinstance(data, list) else data).get("generated_text","")
        except requests.HTTPError as e:
            print(f"    [HF {e.response.status_code}] {e.response.text[:200]}")
        except (KeyError, ValueError, requests.RequestException) as e:
            print(f"    [HF] {e}")
        return None


def build_llm_backend(args) -> Optional[BackendBase]:
    if not args.llm_backend:
        return None
    if args.llm_backend == "ollama-local":
        b = OllamaBackend(args.llm_model, mode="local",
                          base_url=args.base_url,
                          temperature=args.temperature, timeout=args.timeout)
        models = b.list_models()
        if models:
            print(f"  Ollama local — modelos: {models}")
        else:
            print(f"  ⚠  Ollama no responde en {b.url}. Ejecuta: ollama serve")
        return b
    if args.llm_backend == "ollama-web":
        if not args.api_key:
            raise ValueError("--api-key obligatorio para ollama-web")
        return OllamaBackend(args.llm_model, mode="web", api_key=args.api_key,
                             temperature=args.temperature, timeout=args.timeout)
    if args.llm_backend == "huggingface":
        if not args.api_key:
            raise ValueError("--api-key obligatorio para huggingface")
        return HuggingFaceBackend(args.llm_model, args.api_key,
                                  use_chat=not args.hf_text_gen,
                                  temperature=args.temperature, timeout=args.timeout)
    raise ValueError(f"Backend desconocido: {args.llm_backend}")


def llm_predict(backend: BackendBase, post: dict, retries: int = 3) -> Optional[dict]:
    """Call LLM → parse JSON → return raw dict or None."""
    text     = f"Title: {post['title']}\nText: {post['selftext']}"
    user_msg = f"{_FEWSHOT}\n--- YOUR TURN ({post['lang']}) ---\nReturn ONLY JSON:\n\n{text}"
    for attempt in range(retries):
        raw = backend.complete(_SYSTEM, user_msg)
        if raw:
            parsed = backend.parse_json(raw)
            if parsed and "primary_label" in parsed:
                return parsed
            if attempt < retries - 1:
                print(f"    [retry {attempt+1}] bad JSON: {(raw or '')[:80]}")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return None

# ══════════════════════════════════════════════════════════════════════════════
#  ASSEMBLER  — merge ML + LLM into final prediction dict
# ══════════════════════════════════════════════════════════════════════════════

def assemble(post: dict,
             ml_label: str,
             ml_emotions: list,
             llm_raw: Optional[dict] = None) -> dict:
    """
    Priority rules:
      • If LLM returned a valid JSON: use LLM for A, B, C, D
      • Otherwise: use ML for A & B, rules for C & D
    """
    title    = str(post.get("title")    or "")
    selftext = str(post.get("selftext") or "")

    if llm_raw is not None:
        primary  = norm_label(llm_raw.get("primary_label", ml_label))
        emotions = norm_emotions(llm_raw.get("trigger_emotions", ml_emotions))
        spans    = ([] if primary not in HOPE_LABELS
                    else validate_spans(llm_raw.get("span_annotations", []),
                                        title, selftext))
        # if LLM gave 0 valid spans but predicted a Hope label → fallback to rules
        if primary in HOPE_LABELS and not spans:
            spans = extract_spans_rules(title, selftext, primary)
    else:
        primary  = ml_label
        emotions = ml_emotions
        spans    = extract_spans_rules(title, selftext, primary)

    return {
        "row_id":           int(post["row_id"]),
        "lang":             str(post.get("lang") or ""),
        "title":            title,
        "selftext":         selftext,
        "primary_label":    primary,
        "trigger_emotions": emotions,
        "span_annotations": spans,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION  (self-eval on training sample)
# ══════════════════════════════════════════════════════════════════════════════

def macro_f1_multiclass(golds: list, preds: list, labels: list) -> float:
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for g, p in zip(golds, preds):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1; fn[g] += 1
    f1s = []
    for lbl in labels:
        pr = tp[lbl] / (tp[lbl]+fp[lbl]) if (tp[lbl]+fp[lbl]) > 0 else 0
        rc = tp[lbl] / (tp[lbl]+fn[lbl]) if (tp[lbl]+fn[lbl]) > 0 else 0
        f1 = 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0
        f1s.append(f1)
    return sum(f1s)/len(f1s)

def macro_f1_multilabel(golds: list, preds: list, labels: list) -> float:
    """Per-label binary F1 then macro average."""
    f1s = []
    for lbl in labels:
        tp = sum(1 for g,p in zip(golds,preds) if lbl in g and lbl in p)
        fp = sum(1 for g,p in zip(golds,preds) if lbl not in g and lbl in p)
        fn = sum(1 for g,p in zip(golds,preds) if lbl in g and lbl not in p)
        pr = tp/(tp+fp) if (tp+fp) > 0 else 0
        rc = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1 = 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0
        f1s.append(f1)
    return sum(f1s)/len(f1s)

def rouge1_f(gold_span: str, pred_span: str) -> float:
    """Token-level ROUGE-1 F1 between two strings."""
    g = gold_span.lower().split()
    p = pred_span.lower().split()
    if not g or not p:
        return 0.0
    common = Counter(g) & Counter(p)
    overlap = sum(common.values())
    prec = overlap / len(p)
    rec  = overlap / len(g)
    return 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0

def evaluate_sample(model: MLModels, train_rows: list,
                    llm_backend: Optional[BackendBase] = None,
                    sample_size: int = 200, seed: int = 42,
                    delay: float = 0.2, retries: int = 3):
    import random
    random.seed(seed)
    sample = random.sample(train_rows, min(sample_size, len(train_rows)))

    print(f"\n{'='*65}")
    print(f"SELF-EVAL  n={len(sample)}  "
          f"llm={'OFF' if llm_backend is None else llm_backend.model}")
    print(f"{'='*65}")

    ml_preds  = model.predict(sample)
    gold_A    = [r["primary_label"] for r in sample]
    gold_B    = [norm_emotions(r.get("trigger_emotions",[])) for r in sample]

    preds_A, preds_B, preds_spans = [], [], []

    for i, (row, (ml_a, ml_b)) in enumerate(zip(sample, ml_preds)):
        llm_raw = None
        if llm_backend:
            llm_raw = llm_predict(llm_backend, row, retries=retries)
            time.sleep(delay)
        pred = assemble(row, ml_a, ml_b, llm_raw)
        preds_A.append(pred["primary_label"])
        preds_B.append(pred["trigger_emotions"])
        preds_spans.append(pred["span_annotations"])

        ok = pred["primary_label"] == row["primary_label"]
        print(f"  [{i+1:3d}/{len(sample)}] {'✓' if ok else '✗'}"
              f"  GOLD={row['primary_label']:20s}"
              f"  PRED={pred['primary_label']:20s}"
              f"  spans={len(pred['span_annotations'])}")

    # ── Task A ──
    f1_A = macro_f1_multiclass(gold_A, preds_A, VALID_PRIMARY)
    acc_A = sum(g==p for g,p in zip(gold_A, preds_A)) / len(gold_A)

    # ── Task B ──
    f1_B = macro_f1_multilabel(gold_B, preds_B, EMOTIONS_ALL)

    # ── Task C — ROUGE-1 on aligned spans ──
    rouge_scores = []
    for row, pred_spans in zip(sample, preds_spans):
        gold_spans = [s["span"] for s in row.get("span_annotations", [])]
        pred_texts = [s["span"] for s in pred_spans]
        if not gold_spans and not pred_texts:
            rouge_scores.append(1.0)  # both empty = correct
        elif not gold_spans or not pred_texts:
            rouge_scores.append(0.0)
        else:
            # Best match for each gold span
            for gs in gold_spans:
                best = max((rouge1_f(gs, ps) for ps in pred_texts), default=0.0)
                rouge_scores.append(best)

    rouge_C = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    # ── Task D — stance + actor on gold-aligned spans ──
    stance_golds, stance_preds = [], []
    actor_golds, actor_preds   = [], []
    for row, pred_spans in zip(sample, preds_spans):
        for gs in row.get("span_annotations", []):
            # align to nearest predicted span
            if not pred_spans:
                stance_preds.append("Desired"); stance_golds.append(gs["outcome_stance"])
                actor_preds.append("Unclear");  actor_golds.append(gs["actor"])
                continue
            best_ps = max(pred_spans, key=lambda ps: rouge1_f(gs["span"], ps["span"]))
            if rouge1_f(gs["span"], best_ps["span"]) > 0.3:
                stance_golds.append(gs["outcome_stance"])
                stance_preds.append(best_ps["outcome_stance"])
                actor_golds.append(gs["actor"])
                actor_preds.append(best_ps["actor"])

    f1_stance = (macro_f1_multiclass(stance_golds, stance_preds, list(VALID_STANCE))
                 if stance_golds else 0.0)
    f1_actor  = (macro_f1_multiclass(actor_golds,  actor_preds,  list(VALID_ACTOR))
                 if actor_golds else 0.0)

    overall = np.mean([f1_A, f1_B, rouge_C, f1_stance, f1_actor])

    print(f"\n{'─'*65}")
    print(f"  Task A  primary_label  Acc={acc_A:.3f}  MacroF1={f1_A:.3f}")
    print(f"  Task B  emotions               MacroF1={f1_B:.3f}")
    print(f"  Task C  spans                  ROUGE-1 ={rouge_C:.3f}")
    print(f"  Task D  stance                 MacroF1={f1_stance:.3f}")
    print(f"  Task D  actor                  MacroF1={f1_actor:.3f}")
    print(f"{'─'*65}")
    print(f"  Overall (mean of 5)            = {overall:.3f}")
    print(f"{'='*65}")
    return overall

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    train_rows = load_jsonl(TRAIN_FILE)
    test_rows  = load_jsonl(TEST_FILE)
    print(f"Train: {len(train_rows)}  |  Test: {len(test_rows)}")

    # ── Train ML models ──
    print("Training ML models (Task A + B)…")
    ml = MLModels()
    ml.fit(train_rows)
    print("  ✓ TF-IDF + LR ready")

    # ── LLM backend (optional) ──
    llm = build_llm_backend(args) if args.llm_backend else None
    if llm:
        print(f"  LLM backend: {args.llm_backend}  model={args.llm_model}")

    # ── Eval mode ──────────────────────────────────────────────────────────
    if args.eval > 0:
        evaluate_sample(ml, train_rows, llm,
                        sample_size=args.eval,
                        delay=args.delay, retries=args.retries)
        return

    # ── Inference ──────────────────────────────────────────────────────────
    ml_preds = ml.predict(test_rows)   # batch ML predictions (fast)
    total    = len(test_rows)
    preds    = []
    errors   = 0

    print(f"\nProcessing {total} test examples…")
    for i, (row, (ml_a, ml_b)) in enumerate(zip(test_rows, ml_preds)):
        print(f"  [{i+1:4d}/{total}] id={row['row_id']} lang={row['lang']}", end="  ", flush=True)

        llm_raw = None
        if llm:
            llm_raw = llm_predict(llm, row, retries=args.retries)
            if llm_raw is None:
                errors += 1
            time.sleep(args.delay)

        pred = assemble(row, ml_a, ml_b, llm_raw)
        preds.append(pred)

        src  = "LLM" if llm_raw else "ML "
        emos = ",".join(pred["trigger_emotions"])
        print(f"[{src}] {pred['primary_label']:20s} | {emos:30s} | spans={len(pred['span_annotations'])}")

    # ── Save JSONL ──
    pred_name = args.output
    pred_file = OUTPUT_DIR / pred_name
    with open(pred_file, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ── Create ZIP ──
    zip_file = OUTPUT_DIR / f"{args.team_name}_Run{args.run_number}.zip"
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_file, pred_name)    # flat, no subdirectory

    # ── Summary ──
    label_dist = Counter(p["primary_label"]         for p in preds)
    lang_dist  = Counter(p["lang"]                  for p in preds)
    span_dist  = Counter(len(p["span_annotations"]) for p in preds)
    print("\n" + "═"*65)
    print(f"Predicciones: {pred_file}")
    print(f"ZIP:          {zip_file}")
    if llm:
        print(f"LLM fallbacks: {errors}/{total}")
    print(f"Etiquetas:  {dict(label_dist)}")
    print(f"Idiomas:    {dict(lang_dist)}")
    print(f"Spans (n):  {dict(span_dist)}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="HOPE-EXP IberLEF 2026 — TF-IDF+LR baseline + LLM refinement",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ejemplos:
  # Solo ML (rápido, sin API keys)
  python hope_exp_solver.py

  # ML + Ollama local
  python hope_exp_solver.py --llm-backend ollama-local --llm-model llama3.1

  # ML + Ollama local en servidor remoto
  python hope_exp_solver.py --llm-backend ollama-local --llm-model llama3.1 \\
      --base-url http://192.168.1.10:11434/api/chat

  # ML + Ollama Cloud
  python hope_exp_solver.py --llm-backend ollama-web --llm-model llama3.3 \\
      --api-key OLLAMA_KEY

  # ML + HuggingFace chat completions
  python hope_exp_solver.py --llm-backend huggingface \\
      --llm-model mistralai/Mistral-7B-Instruct-v0.3 --api-key hf_TOKEN

  # HuggingFace serverless (free tier)
  python hope_exp_solver.py --llm-backend huggingface \\
      --llm-model HuggingFaceH4/zephyr-7b-beta --api-key hf_TOKEN --hf-text-gen

  # Self-eval (todos los tracks)
  python hope_exp_solver.py --eval 200
  python hope_exp_solver.py --llm-backend ollama-local --llm-model llama3.1 --eval 100
        """
    )

    # ── LLM (opcional) ──
    p.add_argument("--llm-backend", default=None, dest="llm_backend",
                   choices=["ollama-local","ollama-web","huggingface"],
                   help="Backend LLM (opcional; sin él solo se usa ML)")
    p.add_argument("--llm-model",   default="llama3.1", dest="llm_model",
                   help="Modelo para el LLM backend (default: llama3.1)")
    p.add_argument("--api-key",     default=None, dest="api_key",
                   help="API key para ollama-web / huggingface")
    p.add_argument("--base-url",    default=None, dest="base_url",
                   help="URL custom para Ollama (ej: http://host:11434/api/chat)")
    p.add_argument("--hf-text-gen", action="store_true", dest="hf_text_gen",
                   help="HF: usar endpoint serverless en lugar de chat completions")

    # ── Generación ──
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--timeout",     type=int,   default=90)
    p.add_argument("--retries",     type=int,   default=3)
    p.add_argument("--delay",       type=float, default=0.3,
                   help="Pausa entre llamadas LLM (segundos)")

    # ── Salida ──
    p.add_argument("--output",     default="pred.jsonl",
                   help="Nombre del JSONL de salida (default: pred.jsonl)")
    p.add_argument("--team-name",  default=TEAM_NAME,  dest="team_name")
    p.add_argument("--run-number", default=RUN_NUMBER, dest="run_number", type=int)

    # ── Eval ──
    p.add_argument("--eval", type=int, default=0, metavar="N",
                   help="Evaluar sobre N ejemplos del train (todos los tracks)")

    main(p.parse_args())