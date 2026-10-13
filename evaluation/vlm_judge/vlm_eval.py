"""
VLM judge for the conflict (FDR) experiments, using Gemini-2.5-Flash.

For every conflict rollout written by ``common/run_conflict.sh`` the judge
watches the first-episode video (``video/ep000.mp4``), is told which factor
value each object / action corresponds to, and answers which factor the policy
followed (``FACTOR_FOLLOWED``: factor 1, factor 2, or neither). Existing files
are never modified; results are appended to a JSONL file and the run is
resumable (runs already in the JSONL are skipped).

Input layout:  <results-root>/<experiment>/experiments/ood_<idx>_<experiment>_<i>_<j>_<run_type>_<ts>/
Output:        <out-dir>/<label>/vlm_eval_<experiment>.jsonl (+ _summary.txt)

Credentials: GEMINI_API_KEY or GOOGLE_API_KEY (environment only).

Usage:
  python vlm_eval.py --results-root OUT --label gr00t [--experiments verb_color ...]
                     [--out-dir OUT/vlm_eval] [--limit N] [--workers 10]
                     [--backend gemini|openai]
"""

import argparse, base64, json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

WORKERS = 10  # concurrent VLM requests (overridable via --workers)

MODEL = os.environ.get("VLM_MODEL", "gemini-2.5-flash")
# Native Gemini REST endpoint (default) or any OpenAI-compatible chat endpoint
# that accepts a base64 video in an image_url part (``--backend openai``).
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OPENAI_URL = os.environ.get(
    "VLM_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
BACKEND = "gemini"


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment.")
    return key

VERBS   = ["lift", "grasp", "push", "pull", "rotate", "slide"]
COLORS  = ["red", "yellow", "blue", "orange", "green", "black"]
SHAPES  = ["cube", "sphere", "cup", "car", "pyramid", "star"]
SIZES   = ["smallest", "second smallest", "middle-sized", "second largest", "largest", "biggest"]
SPATIALS= ["on the left", "on the right", "in the middle", "in front", "at the back", "in the center"]

ALL_EXPERIMENTS = [
    "verb_color", "verb_object", "verb_size", "verb_spatial",
    "color_object", "size_object", "color_size", "color_spatial",
    "spatial_size", "spatial_object",
]

# Prompt builders
def prompt_verb_experiment(instruction, pair_i, pair_j, factor2):
    verb_i = VERBS[pair_i]; verb_j = VERBS[pair_j]
    if factor2 == "color":
        factor2_desc = f"color '{COLORS[pair_j]}' (which the robot was trained to interact with using '{verb_j}')"
    elif factor2 == "shape":
        factor2_desc = f"shape '{SHAPES[pair_j]}' (which the robot was trained to interact with using '{verb_j}')"
    elif factor2 == "size":
        factor2_desc = f"size '{SIZES[pair_j]}' (which the robot was trained to interact with using '{verb_j}')"
    elif factor2 == "spatial":
        factor2_desc = f"spatial position '{SPATIALS[pair_j]}' (which the robot was trained to interact with using '{verb_j}')"
    else:
        factor2_desc = f"factor {pair_j} (trained verb: '{verb_j}')"
    return f"""Watch this robot manipulation video carefully.

**Instruction given to the robot:** "{instruction}"

This is a CONFLICT experiment about verb bias. The robot was trained with specific verb-attribute pairings:
- Verb "{verb_i}" is paired with one attribute set (instruction verb)
- Verb "{verb_j}" is paired with {factor2_desc}

The robot must choose which action to perform. Observe carefully.

Answer in this EXACT format:
ROBOT_ACTION: [describe what physical action the robot arm performed in 1 sentence]
ACTION_TYPE: ["{verb_i}" if robot performed the instructed action, "{verb_j}" if robot performed the competing action, "other" if neither]
FACTOR_FOLLOWED: ["verb" if robot followed the instructed verb '{verb_i}', "{factor2}" if robot followed the {factor2} factor (doing '{verb_j}'), "neither" if unclear]
CONFIDENCE: [high / medium / low]
REASONING: [1-2 sentences explaining your judgment]"""


def prompt_two_object_experiment(instruction, pair_i, pair_j, factor1, factor2):
    def describe_factor(f, k):
        if f == "color":   return f"color='{COLORS[k]}'"
        if f == "shape":   return f"shape='{SHAPES[k]}'"
        if f == "size":    return f"size='{SIZES[k]}'"
        if f == "spatial": return f"position='{SPATIALS[k]}'"
        return f"{f}={k}"
    obj_a_desc = describe_factor(factor1, pair_i)
    obj_b_desc = describe_factor(factor2, pair_j)
    return f"""Watch this robot manipulation video carefully.

**Instruction given to the robot:** "{instruction}"

This is a CONFLICT experiment about factor bias. The scene contains two objects:
- Object A: has {obj_a_desc}  - matches the '{factor1}' factor in the instruction
- Object B: has {obj_b_desc}  - matches the '{factor2}' factor in the instruction

No object has BOTH attributes simultaneously. The robot must choose one.

Answer in this EXACT format:
ROBOT_ACTION: [describe what the robot arm did in 1 sentence]
OBJECT_TOUCHED: [A (matches {factor1}), B (matches {factor2}), or neither]
FACTOR_FOLLOWED: ["{factor1}" if robot went for Object A, "{factor2}" if robot went for Object B, "neither"]
CONFIDENCE: [high / medium / low]
REASONING: [1-2 sentences explaining your judgment]"""


VERB_EXPS = {"verb_color": "color", "verb_object": "shape",
             "verb_size": "size", "verb_spatial": "spatial"}
TWO_OBJ_EXPS = {
    "color_object": ("color", "shape"), "size_object": ("size", "shape"),
    "color_size": ("color", "size"), "color_spatial": ("color", "spatial"),
    "spatial_size": ("spatial", "size"), "spatial_object": ("spatial", "shape"),
}

def get_prompt(experiment, instruction, pair_i, pair_j, run_type):
    if experiment in VERB_EXPS:
        return prompt_verb_experiment(instruction, pair_i, pair_j, VERB_EXPS[experiment])
    if experiment in TWO_OBJ_EXPS:
        f1, f2 = TWO_OBJ_EXPS[experiment]
        return prompt_two_object_experiment(instruction, pair_i, pair_j, f1, f2)
    return f'Watch this robot video. Instruction: "{instruction}". What did the robot do?'

# API call
def _post(prompt, video_b64):
    """Send one (prompt, video) request; return (status_code, text_or_error)."""
    key = _api_key()
    if BACKEND == "gemini":
        payload = {"contents": [{"parts": [
                      {"text": prompt},
                      {"inline_data": {"mime_type": "video/mp4", "data": video_b64}}]}],
                   "generationConfig": {"maxOutputTokens": 2000}}
        resp = requests.post(GEMINI_URL.format(model=MODEL),
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=payload, timeout=120)
        if resp.status_code != 200:
            return resp.status_code, resp.text
        parts = resp.json()["candidates"][0].get("content", {}).get("parts", [])
        return 200, "".join(p.get("text", "") for p in parts)
    payload = {"model": MODEL, "max_tokens": 2000,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{video_b64}"}}]}]}
    resp = requests.post(OPENAI_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload, timeout=120)
    if resp.status_code != 200:
        return resp.status_code, resp.text
    return 200, resp.json()["choices"][0]["message"]["content"] or ""


def call_vlm(video_path, prompt):
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    for attempt in range(4):
        try:
            status, content = _post(prompt, video_b64)
            if status == 200:
                return parse_response(content)
            elif status == 429:
                print("  rate-limited 30s", flush=True); time.sleep(30)
            else:
                print(f"  API {status}: {content[:120]}", flush=True); time.sleep(5)
        except Exception as e:
            print(f"  exc: {e}", flush=True); time.sleep(5)
    return {"raw": "FAILED", "factor_followed": "error"}


def parse_response(text):
    result = {"raw": text}
    for field in ["ROBOT_ACTION", "OBJECT_TOUCHED", "ACTION_TYPE",
                  "FACTOR_FOLLOWED", "CONFIDENCE", "REASONING"]:
        m = re.search(rf"{field}:\s*(.+?)(?=\n[A-Z_]+:|$)", text, re.DOTALL)
        result[field.lower()] = m.group(1).strip() if m else ""
    return result

# Run discovery (one dir per index, latest timestamp)
def load_runs(root: Path, experiment: str):
    expdir = root / experiment / "experiments"
    best = {}
    for d in sorted(expdir.glob(f"ood_*_{experiment}_*")):
        if not d.is_dir():
            continue
        parts = d.name.split("_")
        try:
            exp_len = len(experiment.split("_"))
            idx = int(parts[1])
            pair_i = int(parts[2 + exp_len])
            pair_j = int(parts[3 + exp_len])
            run_type = parts[4 + exp_len]
            run_name = "_".join(parts[:5 + exp_len])
        except (ValueError, IndexError):
            continue
        sr = d / "success_rate.txt"
        existing = "?/1"
        if sr.exists():
            m = re.search(r"_success=(\d+/\d+)", sr.read_text())
            if m: existing = m.group(1)
        best[idx] = {"idx": idx, "pair_i": pair_i, "pair_j": pair_j,
                     "run_type": run_type, "existing_success": existing,
                     "run_name": run_name, "dir": str(d)}  # sorted() => last wins = latest ts
    return [best[k] for k in sorted(best)]

# Per-experiment evaluation
def evaluate_experiment(model: str, experiment: str, root: Path, out_base: Path,
                        limit: int = 0):
    out_dir = out_base / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"vlm_eval_{experiment}.jsonl"
    print(f"\n{'='*60}\n[{model}] Evaluating: {experiment}\nOutput: {out_file}", flush=True)

    runs = load_runs(root, experiment)
    print(f"found {len(runs)} runs", flush=True)

    done_names = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            try:
                done_names.add(json.loads(line)["run_name"])
            except Exception:
                pass
    if done_names:
        print(f"resume: {len(done_names)} already done, skipping them", flush=True)

    factor_counts = {}
    # Include results already written to the jsonl in the summary.
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            try:
                f = json.loads(line).get("vlm_factor_followed", "error")
                factor_counts[f] = factor_counts.get(f, 0) + 1
            except Exception:
                pass

    # Pending videos (skip finished or missing ones), up to --limit.
    pending = []
    for run in runs:
        if run["run_name"] in done_names:
            continue
        run_dir = Path(run["dir"])
        video_path = run_dir / "video" / "ep000.mp4"
        if not run_dir.exists() or not video_path.exists():
            print(f"  [{run['idx']}] MISSING dir/video", flush=True); continue
        instruction = "unknown"
        sr_file = run_dir / "success_rate.txt"
        if sr_file.exists():
            m = re.search(r"instruction='(.+?)'", sr_file.read_text())
            if m: instruction = m.group(1)
        run["_video"] = str(video_path); run["_instruction"] = instruction
        pending.append(run)
        if limit and len(pending) >= limit:
            break
    print(f"pending={len(pending)}  workers={WORKERS}", flush=True)

    out_f = open(out_file, "a")
    lock = threading.Lock()
    state = {"n": 0}

    def work(run):
        prompt = get_prompt(experiment, run["_instruction"], run["pair_i"],
                            run["pair_j"], run["run_type"])
        vlm = call_vlm(Path(run["_video"]), prompt)
        factor = vlm.get("factor_followed", "error").lower().strip('"\'').strip()
        record = {"model": model, "experiment": experiment, "run_name": run["run_name"],
            "idx": run["idx"], "pair_i": run["pair_i"], "pair_j": run["pair_j"],
            "run_type": run["run_type"], "instruction": run["_instruction"],
            "existing_success": run["existing_success"],
            "vlm_factor_followed": factor,
            "vlm_object_touched": vlm.get("object_touched", ""),
            "vlm_action_type": vlm.get("action_type", ""),
            "vlm_robot_action": vlm.get("robot_action", ""),
            "vlm_confidence": vlm.get("confidence", ""),
            "vlm_reasoning": vlm.get("reasoning", ""),
            "vlm_raw": vlm.get("raw", "")}
        with lock:
            out_f.write(json.dumps(record) + "\n"); out_f.flush()
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
            state["n"] += 1
            print(f"  [{model}/{experiment} {state['n']}/{len(pending)}] "
                  f"idx{run['idx']} {factor.upper()} [{vlm.get('confidence','')}]", flush=True)

    if pending:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(work, pending))
    out_f.close()

    total = sum(factor_counts.values())
    with open(out_dir / f"vlm_eval_{experiment}_summary.txt", "w") as f:
        f.write(f"# VLM Eval: {model} {experiment}\nmodel: {MODEL}\ntotal_evaluated: {total}\n\n")
        for fac, cnt in sorted(factor_counts.items(), key=lambda x: -x[1]):
            f.write(f"{fac}: {cnt}/{total} ({100*cnt/max(1,total):.1f}%)\n")
    print(f"  [{model}/{experiment}] +{state['n']} new, totals={factor_counts}", flush=True)
    return factor_counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gemini VLM judge for conflict rollouts")
    ap.add_argument("--results-root", required=True, type=Path,
                    help="directory holding <experiment>/experiments/ood_* run dirs")
    ap.add_argument("--label", default="gr00t",
                    help="policy label; output goes to <out-dir>/<label>/")
    ap.add_argument("--experiments", nargs="*", default=ALL_EXPERIMENTS,
                    choices=ALL_EXPERIMENTS)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <results-root>/vlm_eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--backend", choices=["gemini", "openai"], default=BACKEND)
    a = ap.parse_args()
    WORKERS = a.workers
    BACKEND = a.backend
    _api_key()
    out_base = a.out_dir or (a.results_root / "vlm_eval")
    model = a.label
    results = {}
    for e in a.experiments:
        results[e] = evaluate_experiment(model, e, a.results_root, out_base, limit=a.limit)
    print("\n" + "="*60 + f"\n[{model}] ALL DONE")
    for e, c in results.items():
        t = sum(c.values()); top = max(c, key=c.get) if c else "NA"
        print(f"  {e}: dominant={top} ({100*c.get(top,0)/max(1,t):.1f}%)")
