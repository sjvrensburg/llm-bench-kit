#!/usr/bin/env python3
"""Six-spec code eval scored by running hidden tests against the model's answer.

  python3 code-eval.py http://127.0.0.1:8110/v1 flash-next-coder --no-think
  python3 code-eval.py http://127.0.0.1:8092/v1 cybertiel-coder-35b-a3b --think --repeats 2

WHY THIS FILE EXISTS. docs/04's Flash-Next section (2026-08-28, since removed from the
tree) recorded a "code arm" of six specs that "lives in the scratchpad until it has been
used twice". The scratchpad copy was gone by 2026-10-03, which made the August numbers
(Flash-Next UD-IQ3_XXS 12/12, Ornith 10/12, thinking off) uncomparable. This is a
reconstruction from that description, so its absolute scores are NOT comparable with
August's -- only arms run through this file are comparable with each other.

Each spec carries one edge case a model can silently get wrong (exclusive vs inclusive
range end, touching intervals, split on first '=', even-count median, the sleep schedule
before each retry, unhashable-key fallback). Scoring runs the model's code against hidden
tests in a subprocess; every spec was validated against the reference first (--selftest).
temperature 0, so repeats only measure non-determinism; max_tokens is a hard ceiling and a
run that burns it with no code block scores 0 and is reported as `capped`.
"""
import argparse
import json
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

SPECS = {
"chunk_ranges": dict(
 prompt="Write a Python function chunk_ranges(n, size) returning a list of (start, end) "
        "tuples that cover range(n) in consecutive chunks of at most `size`; end is "
        "EXCLUSIVE. The last chunk may be shorter. n == 0 returns []. Raise ValueError "
        "if size < 1. Code only.",
 ref="""
def chunk_ranges(n, size):
    if size < 1: raise ValueError
    return [(i, min(i+size, n)) for i in range(0, n, size)]
""",
 test="""
assert chunk_ranges(10,4)==[(0,4),(4,8),(8,10)]
assert chunk_ranges(8,4)==[(0,4),(4,8)]
assert chunk_ranges(0,3)==[]
assert chunk_ranges(3,10)==[(0,3)]
try: chunk_ranges(5,0); raise SystemExit('no ValueError')
except ValueError: pass
"""),
"merge_intervals": dict(
 prompt="Write merge_intervals(intervals) taking a list of (start, end) integer pairs, "
        "inclusive on both ends, in any order, and returning the merged list sorted by "
        "start. Intervals that overlap or share an endpoint (e.g. (1,3) and (3,5)) must "
        "merge; (1,2) and (3,4) must not. Do not mutate the input. Code only.",
 ref="""
def merge_intervals(intervals):
    out=[]
    for s,e in sorted(intervals):
        if out and s<=out[-1][1]: out[-1]=(out[-1][0],max(out[-1][1],e))
        else: out.append((s,e))
    return out
""",
 test="""
a=[(5,6),(1,3),(3,5)]; b=list(a)
assert merge_intervals(a)==[(1,6)]; assert a==b
assert merge_intervals([(1,2),(4,5)])==[(1,2),(4,5)]
assert merge_intervals([])==[]
assert merge_intervals([(1,10),(2,3)])==[(1,10)]
assert merge_intervals([(2,3),(1,1)])==[(1,1),(2,3)]
"""),
"parse_kv": dict(
 prompt="Write parse_kv(text) that parses lines of the form key=value into a dict. Split "
        "each line on the FIRST '=' only (values may contain '='). Strip whitespace around "
        "keys and values. Skip blank lines and lines starting with '#'. A line with no "
        "'=' raises ValueError. Later duplicates win. Code only.",
 ref="""
def parse_kv(text):
    d={}
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('#'): continue
        if '=' not in line: raise ValueError(line)
        k,v=line.split('=',1); d[k.strip()]=v.strip()
    return d
""",
 test="""
assert parse_kv('a=1\\n b = x=y \\n\\n# c=3\\na=2')=={'a':'2','b':'x=y'}
assert parse_kv('')=={}
assert parse_kv('k=')=={'k':''}
try: parse_kv('a=1\\nbad'); raise SystemExit('no ValueError')
except ValueError: pass
"""),
"running_median": dict(
 prompt="Write running_median(values) returning a list where element i is the median of "
        "values[:i+1]. For an even count the median is the mean of the two middle values "
        "(a float). Code only.",
 ref="""
import statistics
def running_median(values):
    return [statistics.median(values[:i+1]) for i in range(len(values))]
""",
 test="""
assert running_median([5,1,3,2])==[5,3.0,3,2.5]
assert running_median([])==[]
assert running_median([2,2])==[2,2.0]
assert running_median([1,100,2,99])==[1,50.5,2,50.5]
"""),
"retry_backoff": dict(
 prompt="Write retry(fn, attempts, base_delay, sleep=time.sleep) that calls fn(); on "
        "exception it sleeps base_delay * 2**k BEFORE retry number k+1 (k starting at 0) "
        "and tries again, up to `attempts` calls total. Return fn's result on success. "
        "After the final failed attempt re-raise that exception WITHOUT sleeping after it. "
        "`sleep` is injectable for testing. Code only (import what you need).",
 ref="""
import time
def retry(fn, attempts, base_delay, sleep=time.sleep):
    for k in range(attempts):
        try: return fn()
        except Exception:
            if k==attempts-1: raise
            sleep(base_delay*2**k)
""",
 test="""
calls=[]; slept=[]
def f():
    calls.append(1)
    if len(calls)<3: raise RuntimeError('x')
    return 'ok'
assert retry(f,5,1,sleep=slept.append)=='ok' and slept==[1,2] and len(calls)==3
calls.clear(); slept.clear()
def g(): calls.append(1); raise KeyError('k')
try: retry(g,3,0.5,sleep=slept.append); raise SystemExit('no raise')
except KeyError: pass
assert len(calls)==3 and slept==[0.5,1.0]
"""),
"dedupe_stable": dict(
 prompt="Write dedupe_stable(items, key=None) returning a list with duplicates removed, "
        "keeping the FIRST occurrence and original order. Duplicates are judged by "
        "key(item) if key is given else the item itself. The judged values may be "
        "UNHASHABLE (e.g. lists or dicts) -- it must still work, with a fallback for "
        "those. Code only.",
 ref="""
def dedupe_stable(items, key=None):
    seen_h=set(); seen_u=[]; out=[]
    for it in items:
        k=key(it) if key else it
        try:
            if k in seen_h: continue
            seen_h.add(k)
        except TypeError:
            if k in seen_u: continue
            seen_u.append(k)
        out.append(it)
    return out
""",
 test="""
assert dedupe_stable([3,1,3,2,1])==[3,1,2]
assert dedupe_stable([[1],[2],[1]])==[[1],[2]]
assert dedupe_stable([{'a':1},{'a':1},{'b':2}])==[{'a':1},{'b':2}]
assert dedupe_stable(['a','B','b'],key=str.lower)==['a','B']
assert dedupe_stable([[1,2],[3],[1,2]],key=lambda x:x)==[[1,2],[3]]
assert dedupe_stable([1,[1],1,[1]])==[1,[1]]
assert dedupe_stable([])==[]
"""),
}


def run_tests(code, test):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n" + test)
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=20)
        return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1][:120]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def extract(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return max(blocks, key=len) if blocks else (text or "")


def ask(base, model, prompt, think, max_tokens):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    return d, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?"); ap.add_argument("model", nargs="?")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--think", action="store_true"); g.add_argument("--no-think", action="store_true")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--json"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        bad = [n for n, s in SPECS.items() if not run_tests(s["ref"], s["test"])[0]]
        print("selftest:", "ALL REFERENCES PASS" if not bad else f"FAIL {bad}")
        return 1 if bad else 0
    think = a.think
    rows = []; total = 0; n = 0
    for name, s in SPECS.items():
        res = []
        for _ in range(a.repeats):
            d, secs = ask(a.base, a.model, s["prompt"], think, a.max_tokens)
            ch = d["choices"][0]; toks = d["usage"]["completion_tokens"]
            ok, err = run_tests(extract(ch["message"].get("content")), s["test"])
            capped = ch["finish_reason"] == "length"
            res.append(dict(ok=ok, err=err, tokens=toks, secs=round(secs, 1), capped=capped))
            total += ok; n += 1
        rows.append((name, res))
        print(f"{name:16s} {sum(r['ok'] for r in res)}/{len(res)}  tok {[r['tokens'] for r in res]}  "
              f"s {[r['secs'] for r in res]}  capped {sum(r['capped'] for r in res)}  "
              f"{[r['err'] for r in res if not r['ok']]}", flush=True)
    toks = [r["tokens"] for _, rs in rows for r in rs]
    print(f"TOTAL {total}/{n}  median tok {statistics.median(toks)}  "
          f"median s {statistics.median(r['secs'] for _, rs in rows for r in rs)}")
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(dict(base=a.base, model=a.model, think=think, total=total, n=n,
                                    rows={k: v for k, v in rows})) + "\n")


if __name__ == "__main__":
    sys.exit(main())
