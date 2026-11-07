"""Leakage check for commit-offset diagnostics.

For every completed debate, takes the ADR's own *Decision Outcome* section
(which is never part of the debate prompt), discards MADR template boilerplate
and any text that also occurs in the prompt (Context / Drivers / Options), and
tests whether a 60-character verbatim slice of what remains appears anywhere in
the stored debate trace. Because the outcome is withheld from the agents, the
only way such text can reach the trace is through repository retrieval.

Usage: python leak_check.py <db file> <table> [--csv out.csv] [--examples N] [--dump-traces DIR]
  --dump-traces DIR  writes one plain-text file per ADR (header + full debate trace) into DIR
"""
import sqlite3, re, sys, csv, argparse

TEMPLATE_ADR = re.compile(r"\[option 1\]|\[justification", re.I)  # outcome never filled in (MADR placeholder)
BOILER = re.compile(r"\[option 1\]|\[justification|\[e\.g\.|<!--|-->|TODO|k\.o\. criterion|comes out best|resolves force|optional|quality attribute satisfaction|follow-up decisions", re.I)
def norm(s): return re.sub(r"\s+", " ", s or "").strip()

def outcome_section(adr_text):
    m = re.search(r"#+\s*Decision Outcome[^\n]*\n(.*?)(?=\n#{1,2}\s|\Z)", adr_text or "", re.S | re.I)
    return norm(m.group(1)) if m else ""

def leak_slices(outcome, prompt, trace, width=60, step=15):
    hits = []
    for i in range(0, max(0, len(outcome) - width) + 1, step):
        s = outcome[i:i + width]
        if BOILER.search(s) or s in prompt:
            continue
        if s in trace:
            hits.append(s)
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("table")
    ap.add_argument("--csv"); ap.add_argument("--examples", type=int, default=0)
    ap.add_argument("--dump-traces", metavar="DIR")
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rows = con.execute(f"""select repo_name, adr_name, commit_hash, commit_offset, used_head_fallback,
        file_count, supported_side, debate_answer, context_considered_drivers, other_sections, message_history
        from {a.table} where status='debate_completed'""").fetchall()
    out = []; shown = 0
    if a.dump_traces:
        import os; os.makedirs(a.dump_traces, exist_ok=True)
    for repo, adr, ch, off, fb, fc, side, ans, prompt, adr_text, trace in rows:
        if a.dump_traces:
            fn = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{repo}__{adr}") + ".txt"
            with open(os.path.join(a.dump_traces, fn), "w", newline="") as f:
                f.write(f"RUN: {a.table}\nREPO: {repo}\nADR: {adr}\nADR COMMIT (commit_hash): {ch}\n"
                        f"RETRIEVAL SNAPSHOT: earliest ADR commit minus {off} commit(s)  used_head_fallback={fb}\n"
                        f"FILES RETRIEVED: {fc}\nSUPPORTED SIDE: {side}\nDEBATE ANSWER: {ans}\n\n"
                        + "=" * 30 + " DEBATE TRACE (prompts, retrieved repository context, agent turns) " + "=" * 30 + "\n\n" + (trace or "") + "\n")
        oc = outcome_section(adr_text); tr = norm(trace); pr = norm(prompt)
        assessable = len(oc) >= 60 and not TEMPLATE_ADR.search(oc)
        hits = leak_slices(oc, pr, tr) if assessable else []
        cites = len(re.findall(r"(decision outcome|chosen option)[^.]{0,80}(from|in) the repository", tr, re.I))
        out.append(dict(repo_name=repo, adr_name=adr, commit_hash=ch, commit_offset=off,
                        used_head_fallback=fb, file_count=fc, supported_side=side,
                        outcome_chars=len(oc), outcome_assessable=assessable, outcome_leaked=bool(hits),
                        agent_cites_repo_outcome=cites, debate_answer=norm(ans)[:300]))
        if hits and a.examples and shown < a.examples:
            shown += 1; p = tr.find(hits[0])
            print(f"\n### {repo} / {adr} (offset {off})\nADR outcome: {oc[:220]}\nIN TRACE: ...{tr[max(0,p-200):p+260]}...")
    n = len(out); m = sum(r["outcome_assessable"] for r in out); k = sum(r["outcome_leaked"] for r in out); c = sum(r["agent_cites_repo_outcome"] > 0 for r in out)
    print(f"\n{a.db}:{a.table}  completed={n}  assessable={m}  outcome_leaked={k} ({100*k/m:.0f}% of assessable)  agent_explicitly_cites_repo_outcome={c}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
if __name__ == "__main__": main()
