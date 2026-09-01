"""
NOTE (protocol): EXAONE is generated as a fixed batch of --n candidates per prompt cell (seed_idx
order). The paper's stopping rule (generate until 25 valid per cell) is applied afterwards by
cutting each cell at its 25th valid output in seed order; downstream scripts read only that window.
The full batch is released (data/generated/arm_ab/lg_exaone.jsonl).

gen_exaone_full.py — EXAONE full generation over the frozen 12-prompt set (server, vLLM).

Runs on the server GPU. Produces data/generated/arm_ab/lg_exaone.jsonl in the SAME row
format as the API runner (scripts/gen_full.py), so the analysis is uniform.

As-deployed decoding for a local model = vendor-recommended config (EXAONE 4.0 card:
temperature=0.5, top_p=0.95, presence_penalty=1.5; a first run at temperature=0.6 without
presence_penalty degenerated into repetition and was discarded, see paper §2.3).

Usage (on the server, in the vLLM venv):
    export HF_TOKEN=$(grep -E '^HF_TOKEN=' .env | cut -d= -f2-)
    python scripts/gen_exaone_full.py --n 25 --tp 2 --out data/generated/arm_ab
"""
from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # portable sampler backend

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, MAX_TOKENS, is_valid_poem, output_metrics  # noqa: E402
import prompt_set as PS  # noqa: E402


@dataclass
class Row:
    id: str
    model: str
    provider: str
    prompt_id: str
    framing: str
    length_stated: bool
    gloss: bool
    prompt_sha256: str
    seed_idx: int
    timestamp: str
    decoding: str
    prompt: str
    response: str
    finish_reason: str
    input_tokens: int | None
    output_tokens: int | None
    duration_s: float
    is_valid: bool
    reason: str
    n_chars: int
    n_words: int
    ko_ratio: float
    latin_ratio: float
    has_linebreak: bool
    rep_max_4gram: int
    error: str | None = None


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="attempts per prompt (generous; analysis "
                    "trims to 25 valid/cell — EXAONE valid-rate ~60%, free on server)")
    ap.add_argument("--out", type=str, default="data/generated/arm_ab")
    ap.add_argument("--model", type=str, default="LGAI-EXAONE/EXAONE-4.0-32B")
    ap.add_argument("--tp", type=int, default=2)
    # EXAONE 4.0 NON-reasoning (general) recommendation: temperature < 0.6. The model card
    # also recommends presence_penalty=1.5 "if model degeneration occurs" — our first run
    # (temp=0.6, no penalty = the REASONING-mode setting) hit exactly that degeneration
    # (repetition loops), so we use the vendor's non-reasoning config + its own mitigation.
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--presence_penalty", type=float, default=1.5)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              dtype="bfloat16", generation_config="auto", trust_remote_code=True)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        presence_penalty=args.presence_penalty, max_tokens=MAX_TOKENS)
    decoding = (f"as-deployed local (EXAONE 4.0 non-reasoning: temp={args.temperature} "
                f"top_p={args.top_p} presence_penalty={args.presence_penalty}); only max_tokens")
    print(f"EXAONE {args.model} tp={args.tp} | {sp} | prompts={len(PS.PROMPT_SET)} × n={args.n}")

    tok = llm.get_tokenizer()
    recs = {r["prompt_id"]: r for r in PS.as_records()}

    tasks = []  # (prompt_tuple, seed, chat_text)
    for p in PS.PROMPT_SET:
        chat = tok.apply_chat_template([{"role": "user", "content": p[4]}],
                                       tokenize=False, add_generation_prompt=True)
        for s in range(args.n):
            tasks.append((p, s, chat))

    t0 = time.time()
    outs = llm.generate([t[2] for t in tasks], sp)
    dt_total = time.time() - t0

    rows = []
    for (p, s, _), o in zip(tasks, outs):
        pid, framing, ls, gl, ko, _en = p
        comp = o.outputs[0]
        text = (comp.text or "").strip()
        ok, reason = is_valid_poem(text)
        mm = output_metrics(text)
        rows.append(Row(
            id=f"lg_exaone__{pid}__{s:03d}", model="LGAI-EXAONE/EXAONE-4.0-32B",
            provider="local_vllm", prompt_id=pid, framing=framing, length_stated=ls, gloss=gl,
            prompt_sha256=recs[pid]["sha256"], seed_idx=s, timestamp=_ts(), decoding=decoding,
            prompt=ko, response=text,
            finish_reason=str(getattr(comp, "finish_reason", "")),
            input_tokens=len(o.prompt_token_ids) if o.prompt_token_ids is not None else None,
            output_tokens=len(comp.token_ids) if comp.token_ids is not None else None,
            duration_s=round(dt_total / max(1, len(tasks)), 3),
            is_valid=ok, reason=reason, n_chars=mm["n_chars"], n_words=mm["n_words"],
            ko_ratio=mm["ko_ratio"], latin_ratio=mm["latin_ratio"],
            has_linebreak=mm["has_linebreak"], rep_max_4gram=mm["rep_max_4gram"]))
    n_valid = sum(1 for r in rows if r.is_valid)
    print(f"  valid (line-break-free, non-degenerate, ko): {n_valid}/{len(rows)}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lg_exaone.jsonl"
    rows.sort(key=lambda r: (r.prompt_id, r.seed_idx))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"  {len(rows)} generations in {dt_total:.1f}s -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
