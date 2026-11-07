"""????????? ReAct-?????: ??? ??????????? (bash, finish), ???????? ????????, ??????? ??????."""
import glob
import hashlib
import json
import os
import subprocess
import threading
try:
    import openjiuwen  # noqa: F401
except ImportError:
    openjiuwen = None
SYSTEM = (
    "You solve one small task in the current directory using the bash tool. "
    "Be terse: no explanations, no long plans. Use as few tool calls as possible. "
    "When you know the final answer call finish(answer) with ONLY the answer text "
    "(for coding tasks: the word DONE after the tests pass)."
)
TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a bash command in the task directory.",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
    {"type": "function", "function": {
        "name": "finish", "description": "Submit the final answer.",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}},
]
MAX_OBS = 1500
_traj_lock = threading.Lock()
def _run(cmd, cwd, timeout=25):
    try:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CHERE_INVOKING": "1",
        }
        p = subprocess.run(["bash", "-lc", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
        out = (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        out = "[timeout]"
    if len(out) <= MAX_OBS:
        return out
    return out[:MAX_OBS // 2] + "\n...[cut]...\n" + out[-MAX_OBS // 2:]
def _hashes(files):
    return {f: hashlib.sha1(open(f, "rb").read()).hexdigest() for f in files}
def _save_trajectory(task_id, model, spent, ans, msgs):
    if os.environ.get("SAVE_TRAJ") != "1":
        return
    try:
        with _traj_lock:
            os.makedirs("trajectories", exist_ok=True)
            p = os.path.join("trajectories", f"{task_id}_{model}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"task_id": task_id, "model": model, "spent_tokens": round(spent, 1),
                           "answer": ans, "messages": msgs}, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
def run_agent(llm, task, model, max_steps=8, max_tokens=700, task_cap=None):
    """?????????? (answer|None, spent_tokens)."""
    wd = task["_work"]
    is_coding = task.get("task_type") == "coding"
    tests = sorted(glob.glob(os.path.join(wd, "test*.py")))
    test_hash = _hashes(tests)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": task["instruction"]}]
    spent = 0.0
    for _ in range(max_steps):
        if task_cap and spent > task_cap:
            break
        msg, cost = llm.chat(model, msgs, tools=TOOLS, max_tokens=max_tokens)
        spent += cost
        keep = {"role": "assistant", "content": msg.get("content") or ""}
        calls = msg.get("tool_calls") or []
        if calls:
            keep["tool_calls"] = calls
        msgs.append(keep)
        if not calls:
            msgs.append({"role": "user", "content": "Use a tool: bash, or finish(answer)."})
            continue
        for c in calls:
            fn = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if fn == "finish":
                ans = str(args.get("answer", "")).strip()
                if is_coding and tests:
                    if _hashes(tests) != test_hash:
                        obs = "Tests were modified. Revert them and fix mylib.py instead."
                    else:
                        tfile = os.path.basename(tests[0])
                        obs = _run(f"(python3 {tfile} || python {tfile}) 2>&1 | tail -15", wd)
                        if obs.strip().endswith("OK"):
                            _save_trajectory(task["task_id"], model, spent, "DONE", msgs)
                            return "DONE", spent
                    msgs.append({"role": "tool", "tool_call_id": c["id"], "content": obs})
                    continue
                _save_trajectory(task["task_id"], model, spent, ans or None, msgs)
                return (ans or None), spent
            obs = _run(args.get("cmd", ""), wd) if fn == "bash" else "unknown tool"
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": obs or "(no output)"})
    _save_trajectory(task["task_id"], model, spent, None, msgs)
    return None, spent
