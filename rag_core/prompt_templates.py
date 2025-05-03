"""
Prompt builder for FAST-NUCES / PwC / FYP multimodal RAG
"""

from textwrap import dedent
from typing import List, Dict

# ──────────────────────────────────────────────────────────────────
SYSTEM_PREAMBLE = dedent("""
You are an academic research assistant with expert knowledge of:

- FAST-NUCES Annual Report 2023-24  
- PwC slide deck “Basic Understanding of a Company’s Financials”  
- FAST School of Computing “BS Final Year Project Handbook 2023”

Think step-by-step **internally** but DO NOT reveal your reasoning.  
Return a concise, formal answer in clear Markdown:

* Use short paragraphs or bullet lists.  
* Include headings (###) when the answer has multiple parts.  
* Use tables if it improves clarity.  
* Cite each fact with its chunk ID in parentheses, e.g. (annual_p13_t0_c0).  
* If the answer is not in the context, reply “I don’t know from the provided documents.”
""").strip()

# ──────────────────────────────────────────────────────────────────
FS_ANNUAL = dedent("""
Q: How many total faculty members does FAST-NUCES report across all campuses?
A:
### Faculty Total  
FAST-NUCES employs **735** faculty members in aggregate (annual_p13_t0_c0).
""").strip()

FS_PWC = dedent("""
Q: Which statement “undoes all of the accounting principles” to show cash movement?
A:
### Statement Identified  
The **Cash-Flow Statement** is described as undoing all accounting principles to display cash flows (financials_p3_t0_c0).
""").strip()

FS_FYP = dedent("""
Q: According to the FYP Handbook, how should appendices be referenced in the narrative?
A:
### Citation Rule  
Appendices must be referenced inline with the phrase “see Appendix A” (fyp_handbook_p38_t2_c0).
""").strip()

FEW_SHOTS = [FS_ANNUAL, FS_PWC, FS_FYP]

# ──────────────────────────────────────────────────────────────────
def _label(ch: Dict) -> str:
    path = ch["id"]
    tag = "ANNUAL" if path.startswith("annual") else (
          "PWC"    if path.startswith("financials") else (
          "FYP"    if path.startswith("fyp") else "MISC"))
    return f"[{tag}:{path}]"

def build_prompt(question: str,
                 contexts: List[Dict],
                 max_tokens: int = 1800) -> str:
    ctx_lines = [f"{_label(c)} {c['content'].replace(chr(10), ' ')}"
                 for c in contexts]

    shots = "\n\n".join(FEW_SHOTS)

    def assemble(examples: str) -> str:
        return "\n".join([
            SYSTEM_PREAMBLE, "", examples, "",
            "RETRIEVED CONTEXT:", *ctx_lines, "",
            f"Q: {question}",
            "A:"  # no “step-by-step” to avoid chain-of-thought leakage
        ])

    prompt = assemble(shots)
    while len(prompt.split()) > max_tokens and FEW_SHOTS:
        FEW_SHOTS.pop()
        prompt = assemble("\n\n".join(FEW_SHOTS))

    return prompt
