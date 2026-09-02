"""
prompt_set.py — the 12-prompt set used in the study (paper §2.3, SI S5): full factorial of
framing (instruction | completion | persona) × length (stated | not) × gloss (none | brief
definition) = 12. A single prompt cannot show that a topic bias is prompt-robust, and a
hand-picked subset re-introduces a "why these" choice (Sclar et al. 2024; Mizrahi et al.
2024), so the whole factorial is run and the result is reported over all 12 cells.

Design constants:
  - prose, single paragraph, no internal line breaks (the object of study is line-break-free);
  - register held constant (jondaenmal, the polite register) so it does not co-vary with framing;
  - the constraint clause is byte-identical across all 12; only the framing differs;
  - persona = instruction + one role sentence.
"""
from __future__ import annotations
import hashlib
from itertools import product

# ---- Korean segments (what is actually sent) ----
_NOLINE = "줄바꿈 없이 한 단락의 연속된 산문 형태로 쓰고"
_LEN = "길이는 112자에서 1,014자 사이로 하며"
_BODY = "제목·부제·설명·인사말 없이 시 본문만 씁니다"
_GLOSS = "산문시는 행을 나누지 않고 산문처럼 이어 쓰되 시적 정서를 담은 시입니다"
_REQUEST = "다음 조건에 맞는 한국어 산문시 한 편을 써 주세요"
_REQUEST2 = "아래 조건에 맞는 한국어 산문시 한 편을 써 주세요"
_PERSONA = "당신은 한국어 산문시를 쓰는 시인입니다"
_COMP_HEAD = "다음은 한 편의 한국어 산문시입니다"
_COMP_CUE = "본문:"

# ---- English glosses (paper only; identical KO segment -> identical EN segment) ----
_NOLINE_EN = "be written as a single continuous paragraph of prose without line breaks"
_LEN_EN = "be between 112 and 1,014 characters long"
_BODY_EN = "contain only the poem body, without title, subtitle, explanation, or greeting"
_GLOSS_EN = ("A prose poem is a poem written continuously like prose without line breaks, "
             "yet carrying poetic sentiment.")
_REQUEST_EN = "Please write one Korean prose poem meeting the following conditions."
_PERSONA_EN = "You are a poet who writes Korean prose poems."
_COMP_HEAD_EN = "The following is one Korean prose poem."
_COMP_CUE_EN = "Body:"

FRAMINGS = ("instruction", "completion", "persona")
LENGTHS = (True, False)
GLOSSES = (False, True)


def _cons(ko: bool, length_stated: bool) -> str:
    if ko:
        return f"{_NOLINE}, " + (f"{_LEN}, " if length_stated else "") + f"{_BODY}."
    return "The poem must " + f"{_NOLINE_EN}, " + (f"{_LEN_EN}, " if length_stated else "") + f"and {_BODY_EN}."


def _build(framing, length_stated, gloss, ko=True):
    cons = _cons(ko, length_stated)
    if ko:
        g = f" {_GLOSS}." if gloss else ""
        if framing == "instruction":
            return f"{_REQUEST}.{g} {cons}"
        if framing == "persona":
            return f"{_PERSONA}.{g} {_REQUEST2}. {cons}"
        return f"{_COMP_HEAD}.{g} {cons} {_COMP_CUE}"
    # english gloss (paper only)
    g = f" {_GLOSS_EN}" if gloss else ""
    if framing == "instruction":
        return f"{_REQUEST_EN}{g} {cons}"
    if framing == "persona":
        return f"{_PERSONA_EN}{g} {_REQUEST_EN} {cons}"
    return f"{_COMP_HEAD_EN}{g} {cons} {_COMP_CUE_EN}"


PROMPT_SET = []
for i, (framing, ls, gl) in enumerate(product(FRAMINGS, LENGTHS, GLOSSES)):
    PROMPT_SET.append((f"P{i:02d}", framing, ls, gl,
                       _build(framing, ls, gl, ko=True),
                       _build(framing, ls, gl, ko=False)))



def as_records():
    return [{"prompt_id": pid, "framing": framing, "length_stated": ls, "gloss": gl,
             "sha256": hashlib.sha256(ko.encode()).hexdigest()[:16], "n_chars": len(ko),
             "text_ko": ko, "text_en_gloss": en}
            for pid, framing, ls, gl, ko, en in PROMPT_SET]


if __name__ == "__main__":
    import json, sys
    if "--json" in sys.argv:
        print(json.dumps(as_records(), ensure_ascii=False, indent=2)); sys.exit()
    print(f"TOTAL PROMPTS: {len(PROMPT_SET)} (full factorial 3x2x2)\n")
    for pid, framing, ls, gl, ko, en in PROMPT_SET:
        star = ""
        print(f"===== {pid}  framing={framing}  length_stated={ls}  gloss={gl}{star} =====")
        print(f"[KO · sent to model]\n{ko}\n")
        print(f"[EN gloss · paper only, NOT sent]\n{en}\n")
