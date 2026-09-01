"""
Shared paths, constants, and the model registry.

Human reference: the line-break-free prose-poem corpus (N=4,642), whose frozen
25-motif model lives in data/atlas/instrument_frozen/ and is used as the instrument
(paper §2.2.1). Six generation systems: three from US developers and three from
Korean developers (SI S4).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

# ---------------------------------------------------------------- paths
DATA = ROOT / "data"
ATLAS = DATA / "atlas"                       # human corpus (from prior work)
INSTRUMENT = ATLAS / "instrument_frozen"                 # frozen instrument (25-motif model)
GENERATED = DATA / "generated"               # data/generated/arm_ab/{system}.jsonl
EMBEDDINGS = DATA / "embeddings"             # generated-poem embeddings (mirror atlas)
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# INSTRUMENT frozen instrument
MOTIF_CENTROIDS_NPZ = INSTRUMENT / "motif_centroids.npz"      # 25 centroids + topic_ids + topwords
HUMAN_MOTIF_DIST = INSTRUMENT / "human_doc_motif_distribution.csv"
TOPIC_INFO = INSTRUMENT / "topic_info.csv"

# Human corpus files (metadata holds bibliographic fields and the full text, which is not redistributed)
ATLAS_META = ATLAS / "metadata.csv"                     # has 원문 (whole-text) + work_id
ATLAS_EMB = ATLAS / "embeddings"                        # whole-text human embeddings

for _d in (GENERATED, EMBEDDINGS, RESULTS, FIGURES, ATLAS_EMB):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- constants
RANDOM_STATE = 42
EMBED_MODEL = "jhgan/ko-sroberta-multitask"   # same embedding space as the instrument

# Pilot prompt (single-prompt runs before the 12-prompt design; kept for the pilot scripts).
# The 12 prompts actually used in the study are in prompt_set.py.
MAX_TOKENS = 4096   # truncation guard only; does not reshape the output distribution

# ---------------------------------------------------------------- embedding spaces
# The instrument space (§2.2.1).
EMBED_SPACES = {
    "kosroberta": "jhgan/ko-sroberta-multitask",
}
PRIMARY_SPACE = "kosroberta"


def space_tag(model_name: str) -> str:
    """Reverse-lookup a short tag for a HF model id (falls back to a slug)."""
    for tag, mid in EMBED_SPACES.items():
        if mid == model_name:
            return tag
    return model_name.split("/")[-1].replace(".", "-")


def _e5_prefix(model_name: str, texts: list[str]) -> list[str]:
    """multilingual-e5 expects an instruction prefix; use the symmetric 'query: ' form."""
    if "e5" in model_name.lower():
        return [f"query: {t}" for t in texts]
    return texts


def encode_texts(embedder, texts: list[str], model_name: str, batch_size: int = 32):
    """L2-normalized embeddings, applying the model's required input convention (e.g. e5)."""
    import numpy as _np
    if not texts:
        dim = getattr(embedder, "get_sentence_embedding_dimension", lambda: 768)()
        return _np.zeros((0, dim), _np.float32)
    prepared = _e5_prefix(model_name, texts)
    return embedder.encode(prepared, normalize_embeddings=True,
                           show_progress_bar=False, batch_size=batch_size).astype(_np.float32)

# Reliability-robust motifs: T0 sea, T1 flora, T7 rain, T11 meta-poetic, T14 music, T16 cat
ROBUST_MOTIFS = [0, 1, 7, 11, 14, 16]

# Validity filters, anchored to the human corpus (N=5,166; results/params_derived.json).
KO_RATIO_MIN = 0.5               # share of Hangul among non-space characters; far below the human 1st percentile (0.84), rejects only outputs that drift into another language

# ---------------------------------------------------------------- model registry
# id -> (provider, api_model_id, [env keys]). Identifiers as used at the 2026-07-28 snapshot (SI S4).
MODELS = {
    "openai_gpt-5.6":        ("openai",   "gpt-5.6",                    ["OPENAI_API_KEY"]),
    "anthropic_opus-4-8":    ("anthropic","claude-opus-4-8",            ["ANTHROPIC_API_KEY"]),
    "google_gemini-2.5-pro": ("google",   "gemini-2.5-pro",            ["GOOGLE_API_KEY"]),
    "naver_hcx-005":         ("naver",    "HCX-005",                    ["CLOVA_STUDIO_API_KEY", "CLOVA_STUDIO_REQUEST_ID"]),
    "upstage_solar-pro3":    ("upstage",  "solar-pro3-260323",          ["UPSTAGE_API_KEY"]),
}

UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"
CLOVA_URL = "https://clovastudio.stream.ntruss.com/testapp/v3/chat-completions/HCX-005"


def load_embedder(model_name: str = EMBED_MODEL):
    """Return a SentenceTransformer for whole-text embedding."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ---------------------------------------------------------------- output validity
# Acceptance is by form only (paper §2.3): non-empty, no line break, Korean-dominant,
# no degenerate repetition. Topic never enters the accept/reject decision. Every attempt
# (valid and rejected, with the reason) is logged; the form×topic check is in SI S6.
TARGET_VALID = 25          # valid outputs per (model, prompt) cell
REP_DEGEN_THRESHOLD = 10   # a word 4-gram repeated >= this many times = degenerate loop
                           # (repetition-degeneration convention; keeps poetic anaphora, which
                           #  rarely repeats a full 4-gram 10x, while catching runaway loops)


def output_metrics(text: str) -> dict:
    """Recordable per-output metrics (logged for every attempt)."""
    import re
    from collections import Counter
    t = (text or "").strip()
    tot = len(re.findall(r"\S", t))
    ko = len(re.findall(r"[가-힣]", t))
    lat = len(re.findall(r"[A-Za-zЀ-ӿ]", t))
    w = t.split()
    rep_max = 0
    if len(w) >= 4:
        g = Counter(" ".join(w[i:i + 4]) for i in range(len(w) - 3))
        rep_max = max(g.values()) if g else 0
    return {"n_chars": len(t), "n_words": len(w),
            "ko_ratio": round(ko / tot, 3) if tot else 0.0,
            "latin_ratio": round(lat / tot, 3) if tot else 0.0,
            "has_linebreak": "\n" in t, "rep_max_4gram": rep_max, "empty": (len(t) == 0)}


def is_valid_poem(text: str) -> tuple[bool, str]:
    m = output_metrics(text)
    if m["empty"]:
        return False, "empty"
    if m["has_linebreak"]:
        return False, "linebreak"
    if m["ko_ratio"] < KO_RATIO_MIN:
        return False, "ko_ratio"
    if m["rep_max_4gram"] >= REP_DEGEN_THRESHOLD:
        return False, "degenerate"
    return True, "ok"
