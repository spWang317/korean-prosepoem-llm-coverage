"""
gen_full.py — generation runner for the 12-prompt set × API systems, generate-until-N-valid per cell.

Balanced design: every (model, prompt) cell is filled to TARGET_VALID (=25) VALID outputs
(line-break-free Korean prose, non-degenerate; common.is_valid_poem). Cells that fail
validity are topped up (more attempts) up to --cap; the per-cell valid-rate (= valid /
attempts) is recorded as a finding (form-following failure rate per model). EXAONE (local
vLLM, gen_exaone_full.py) is generated as a fixed batch per cell and then cut at the 25th
valid output in seed order to match this stopping rule; the full batch is released too. "As deployed" decoding: provider default,
only max_tokens.

Use --smoke to verify model identifiers with a few calls before a full run.

Usage:
    python scripts/gen_full.py --smoke
    python scripts/gen_full.py --target 25 --out data/generated/arm_ab
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, MAX_TOKENS, TARGET_VALID, is_valid_poem, output_metrics  # noqa: E402
import prompt_set as PS  # noqa: E402

# Roster: each platform's general-purpose flagship at the 2026-07-28 snapshot (SI S4).
# Run --smoke first to verify model identifiers before a full run.
OPENAI_MODEL, ANTHROPIC_MODEL, GEMINI_MODEL, UPSTAGE_MODEL = \
    "gpt-5.6", "claude-opus-4-8", "gemini-2.5-pro", "solar-pro3-260323"
CLOVA_URL = "https://clovastudio.stream.ntruss.com/testapp/v3/chat-completions/HCX-005"
API_MODELS = ["openai_gpt-5.6", "anthropic_opus-4-8", "google_gemini-2.5-pro",
              "naver_hcx-005", "upstage_solar-pro3"]
CONCURRENCY = 6


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _openai(prompt):
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = c.chat.completions.create(model=OPENAI_MODEL, messages=[{"role": "user", "content": prompt}],
                                  max_completion_tokens=MAX_TOKENS)
    u = r.usage
    return (r.choices[0].message.content or ""), (r.choices[0].finish_reason or ""), \
        getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)


def _anthropic(prompt):
    import anthropic
    c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = c.messages.create(model=ANTHROPIC_MODEL, max_tokens=MAX_TOKENS,
                          messages=[{"role": "user", "content": prompt}])
    return r.content[0].text, (r.stop_reason or ""), r.usage.input_tokens, r.usage.output_tokens


def _gemini(prompt):
    from google import genai
    from google.genai import types as gt
    c = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    safety = [gt.SafetySetting(category=x, threshold="BLOCK_NONE") for x in
              ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
               "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]
    r = c.models.generate_content(model=GEMINI_MODEL, contents=prompt,
        config=gt.GenerateContentConfig(max_output_tokens=MAX_TOKENS, safety_settings=safety,
                                        thinking_config=gt.ThinkingConfig(thinking_budget=-1)))
    text = ""
    try:
        text = r.text or ""
    except Exception:
        pass
    if not text:
        cands = getattr(r, "candidates", None) or []
        if cands and cands[0].content and cands[0].content.parts:
            text = "".join(getattr(p, "text", "") or "" for p in cands[0].content.parts)
    finish = str(getattr(r.candidates[0], "finish_reason", "stop")) if getattr(r, "candidates", None) else "stop"
    m = getattr(r, "usage_metadata", None)
    return text, finish, (getattr(m, "prompt_token_count", None) if m else None), \
        (getattr(m, "candidates_token_count", None) if m else None)


def _upstage(prompt):
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["UPSTAGE_API_KEY"].strip(), base_url="https://api.upstage.ai/v1")
    r = c.chat.completions.create(model=UPSTAGE_MODEL, messages=[{"role": "user", "content": prompt}],
                                  max_tokens=MAX_TOKENS)
    u = r.usage
    return (r.choices[0].message.content or ""), (r.choices[0].finish_reason or ""), \
        getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)


def _clova(prompt):
    import requests
    h = {"Authorization": f"Bearer {os.environ['CLOVA_STUDIO_API_KEY'].strip()}", "Content-Type": "application/json"}
    rid = os.environ.get("CLOVA_STUDIO_REQUEST_ID", "").strip()
    if rid:
        h["X-NCP-CLOVASTUDIO-REQUEST-ID"] = rid
    rsp = requests.post(CLOVA_URL, headers=h, timeout=120,
                        json={"messages": [{"role": "user", "content": prompt}], "maxTokens": MAX_TOKENS})
    rsp.raise_for_status()
    res = rsp.json().get("result", {})
    u = res.get("usage", {})
    return (res.get("message", {}).get("content", "") or ""), res.get("stopReason", ""), \
        u.get("promptTokens"), u.get("completionTokens")


CALLS = {"openai_gpt-5.6": _openai, "anthropic_opus-4-8": _anthropic, "google_gemini-2.5-pro": _gemini,
         "naver_hcx-005": _clova, "upstage_solar-pro3": _upstage}
PROVIDER = {"openai_gpt-5.6": "openai", "anthropic_opus-4-8": "anthropic",
            "google_gemini-2.5-pro": "google", "naver_hcx-005": "naver", "upstage_solar-pro3": "upstage"}


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
    attempt: int
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


def _call_retry(fn, prompt, retries=1):
    last = None
    for _ in range(retries + 1):
        try:
            return fn(prompt), None
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(2)
    return None, last


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET_VALID, help="valid outputs per cell")
    ap.add_argument("--cap", type=int, default=60, help="max attempts per cell (runaway guard)")
    ap.add_argument("--out", type=str, default="data/generated/arm_ab")
    ap.add_argument("--models", nargs="+", default=API_MODELS)
    ap.add_argument("--prompts", nargs="+", default=[p[0] for p in PS.PROMPT_SET])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.target, args.prompts, args.cap = 1, ["P00"], 3

    pset = {p[0]: p for p in PS.PROMPT_SET}
    prompts = [pset[pid] for pid in args.prompts]
    recs = {r["prompt_id"]: r for r in PS.as_records()}
    cells = [(m, p) for m in args.models for p in prompts]
    print(f"generate-until-{args.target}-valid | {len(args.models)} models × {len(prompts)} prompts "
          f"= {len(cells)} cells (cap {args.cap} attempts/cell) | decoding: as-deployed")
    if not args.smoke:
        print("!! REAL SPEND. Ctrl-C now if unsure. Starting in 5s..."); await asyncio.sleep(5)

    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[Row] = []
    cellstat: dict[str, dict] = {}

    async def do_cell(m, p):
        pid, framing, ls, gl, ko, _ = p
        valid, attempt = 0, 0
        while valid < args.target and attempt < args.cap:
            async with sem:
                t0 = time.time()
                out, err = await asyncio.to_thread(_call_retry, CALLS[m], ko)
                dt = time.time() - t0
            attempt += 1
            if out is None:
                text, finish, it, ot = "", "error", None, None
            else:
                text, finish, it, ot = out
            text = (text or "").strip()
            ok, reason = (False, "error") if err else is_valid_poem(text)
            if ok:
                valid += 1
            mm = output_metrics(text)
            results.append(Row(
                id=f"{m}__{pid}__{attempt:03d}", model=m, provider=PROVIDER[m], prompt_id=pid,
                framing=framing, length_stated=ls, gloss=gl, prompt_sha256=recs[pid]["sha256"],
                attempt=attempt, timestamp=_ts(),
                decoding="as-deployed (provider default; only max_tokens)", prompt=ko,
                response=text, finish_reason=finish, input_tokens=it, output_tokens=ot,
                duration_s=round(dt, 2), is_valid=ok, reason=reason,
                n_chars=mm["n_chars"], n_words=mm["n_words"], ko_ratio=mm["ko_ratio"],
                latin_ratio=mm["latin_ratio"], has_linebreak=mm["has_linebreak"],
                rep_max_4gram=mm["rep_max_4gram"], error=err))
        cellstat[f"{m}/{pid}"] = {"valid": valid, "attempts": attempt}
        flag = "" if valid >= args.target else f"  << SHORT ({valid}/{args.target})"
        print(f"  {m:22s} {pid}: {valid} valid / {attempt} attempts{flag}")

    await asyncio.gather(*[do_cell(m, p) for (m, p) in cells])

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)
    for m, rows in by_model.items():
        rows.sort(key=lambda r: (r.prompt_id, r.attempt))
        with open(out_dir / f"{m}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    manifest = {
        "run_date_utc": _ts(), "target_valid_per_cell": args.target, "cap": args.cap,
        "models": args.models, "prompts": args.prompts,
        "decoding": "as-deployed (provider default; only max_tokens)",
        "snapshot_date": "2026-07-28",
        "model_api_ids": {"openai_gpt-5.6": OPENAI_MODEL, "anthropic_opus-4-8": ANTHROPIC_MODEL,
                          "google_gemini-2.5-pro": GEMINI_MODEL, "naver_hcx-005": "HCX-005",
                          "upstage_solar-pro3": UPSTAGE_MODEL},
        "prompt_shas": {k: recs[k]["sha256"] for k in args.prompts},
        "per_model_valid": {m: sum(1 for r in rows if r.is_valid) for m, rows in by_model.items()},
        "per_model_attempts": {m: len(rows) for m, rows in by_model.items()},
        "cellstat": cellstat,
        "total_calls": len(results), "total_valid": sum(1 for r in results if r.is_valid),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nDone: {len(results)} attempts, {manifest['total_valid']} valid -> {out_dir}")
    print("per-model valid/completed calls:", {m: f"{manifest['per_model_valid'][m]}/{manifest['per_model_attempts'][m]}"
                                        for m in by_model})
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
