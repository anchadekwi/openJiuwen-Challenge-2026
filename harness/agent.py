"""Точный и экономный ReAct-агент с полной предзагрузкой контекста и верификацией решений."""
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading

from harness import fastpath
from harness.store import normalize_answer

try:
    import openjiuwen  # noqa: F401
except ImportError:
    openjiuwen = None

SYSTEM = (
    "You are an expert autonomous agent solving a task in the current directory.\n"
    "RULES:\n"
    "1. Accuracy is the #1 priority. Carefully read all details, edge cases, and filters in the instruction.\n"
    "2. For math and data_analysis tasks: verify your calculation using the bash tool (python3 -c ...) unless the pre-computed data already gives an indisputable answer.\n"
    "3. For coding tasks: fix all bugs in the code (considering edge cases too), run the test suite via bash to confirm all tests pass, do NOT modify test files, and then call finish('DONE').\n"
    "4. If the task is genuinely impossible/unsolvable (e.g., required file is missing, requested entity/column does not exist in the data, or tests contradict each other), immediately call finish('UNSOLVABLE') to save budget.\n"
    "5. When you have the verified final answer, call finish(answer) with ONLY the exact final value (no extra words or units)."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the task directory.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the verified final answer (or 'DONE' for coding, or 'UNSOLVABLE' if impossible).",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]

MAX_OBS = 2000
_traj_lock = threading.Lock()


def _build_env():
    """Формирует окружение для bash, гарантируя доступность текущего интерпретатора Python."""
    py_dir = os.path.dirname(os.path.abspath(sys.executable))
    cur_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    full_path = f"{py_dir}{os.pathsep}{cur_path}" if py_dir not in cur_path else cur_path
    return {
        "PATH": full_path,
        "HOME": os.environ.get("HOME", "/tmp"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "CHERE_INVOKING": "1",
    }


def _patch_python_cmd(cmd):
    """Заменяет вызовы python3 / python в bash-команде на точный путь к sys.executable."""
    py_exe = sys.executable.replace("\\", "/")
    return re.sub(r"\bpython3?\b", f'"{py_exe}"', cmd)


def _run(cmd, cwd, timeout=25):
    try:
        patched = _patch_python_cmd(cmd)
        p = subprocess.run(
            ["bash", "-c", patched],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_build_env(),
        )
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        out = "[timeout]"
    except Exception as e:
        out = f"[error: {e}]"
    if len(out) <= MAX_OBS:
        return out
    return out[: MAX_OBS // 2] + "\n...[cut]...\n" + out[-MAX_OBS // 2 :]


def _hashes(files):
    return {f: hashlib.sha1(open(f, "rb").read()).hexdigest() for f in files if os.path.isfile(f)}


def snapshot_workspace(wd):
    """Сохраняет исходное состояние файлов workspace в память для чистого отката между попытками."""
    snap = {}
    for root, _, files in os.walk(wd):
        for fn in files:
            if fn == "answer.txt":
                continue
            p = os.path.join(root, fn)
            try:
                if os.path.getsize(p) <= 300_000:
                    snap[p] = open(p, "rb").read()
            except OSError:
                pass
    return snap


def restore_workspace(snap):
    for p, data in snap.items():
        try:
            with open(p, "wb") as f:
                f.write(data)
        except OSError:
            pass


def _verify_coding_tests(wd, tests, orig_test_hash):
    if _hashes(tests) != orig_test_hash:
        return False, "ERROR: Test files were modified! Revert them and fix the implementation instead."
    for tf in tests:
        try:
            p = subprocess.run(
                [sys.executable, os.path.basename(tf)],
                cwd=wd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=12,
                env=_build_env(),
            )
            out = (p.stdout + p.stderr).strip()
            # Поддерживаем как unittest (OK в конце), так и pytest / простые assert-скрипты (returncode == 0)
            if p.returncode != 0 or "Ran 0 tests" in out or "FAILED" in out or "Traceback" in out:
                return False, out[-1000:]
        except Exception as e:
            return False, repr(e)
    return True, "OK"


def _build_initial_context(task, wd):
    """Формирует богатый стартовый контекст, чтобы LLM видела все файлы и подсказки с 1-го шага."""
    ttype = task.get("task_type", "")
    instr = task.get("instruction", "")
    ctx = [f"TASK INSTRUCTION:\n{instr}"]

    # Список файлов в рабочей директории
    try:
        all_files = []
        for r, _, fs in os.walk(wd):
            for f in sorted(fs):
                if f == "answer.txt" or f.endswith(".pyc"):
                    continue
                rel = os.path.relpath(os.path.join(r, f), wd).replace("\\", "/")
                all_files.append(rel)
        if all_files:
            if len(all_files) <= 15:
                ctx.append(f"\nWORKSPACE FILES ({len(all_files)}): {', '.join(all_files)}")
            else:
                ctx.append(f"\nWORKSPACE FILES ({len(all_files)} total, first 15): {', '.join(all_files[:15])} ...")
    except Exception:
        pass

    # Если детерминированный парсер смог посчитать кандидата — даём его модели для быстрой проверки
    if ttype in ("math", "data_analysis", "search"):
        cand = fastpath.solve(task, wd)
        if cand is not None:
            ctx.append(
                f"\n[PRE-COMPUTED CANDIDATE ANSWER]: {cand}\n"
                "(Verify if this matches ALL conditions in the instruction. If it is 100% accurate, you can call finish immediately.)"
            )

    try:
        if ttype == "coding":
            py_files = sorted(glob.glob(os.path.join(wd, "*.py")))
            tests = sorted(glob.glob(os.path.join(wd, "test*.py")) + glob.glob(os.path.join(wd, "*_test.py")))
            for pf in py_files[:6]:
                if os.path.getsize(pf) < 6000:
                    name = os.path.basename(pf)
                    ctx.append(f"\n--- FILE: {name} ---\n{open(pf, encoding='utf-8-sig').read().strip()}")
            if tests:
                _, tout = _verify_coding_tests(wd, tests, _hashes(tests))
                ctx.append(f"\n--- INITIAL TEST OUTPUT ---\n{tout}")

        elif ttype == "data_analysis":
            csv_files = sorted(glob.glob(os.path.join(wd, "*.csv")) + glob.glob(os.path.join(wd, "*.tsv")))
            for cf in csv_files[:3]:
                with open(cf, newline="", encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    cols = list(rows[0].keys())
                    meta = [f"\n--- CSV FILE: {os.path.basename(cf)} ({len(rows)} rows) ---", f"Columns: {', '.join(cols)}"]
                    for c in cols:
                        uniq = sorted({str(r.get(c, "")) for r in rows})
                        if 1 <= len(uniq) <= 15 and not all(u.isdigit() for u in uniq):
                            meta.append(f"Unique '{c}': {uniq}")
                    sample = "\n".join(",".join(str(r.get(c, "")) for c in cols) for r in rows[:5])
                    meta.append(f"First 5 rows:\n{sample}")
                    ctx.append("\n".join(meta))

        elif ttype == "search":
            kb = os.path.join(wd, "kb")
            if os.path.isdir(kb):
                md_files = sorted(f for f in os.listdir(kb) if f.endswith(".md"))
                total_kb_size = sum(os.path.getsize(os.path.join(kb, f)) for f in md_files)
                # Если вся база kb/ компактная (< 8 КБ, как в задачах из 37 коротких файлов) —
                # подаём сводный индекс всей базы сразу! Это даёт 100% точность на любых многоходовых вопросах!
                if total_kb_size <= 8000:
                    compact_entries = []
                    for fn in md_files:
                        txt = open(os.path.join(kb, fn), encoding="utf-8-sig").read().strip()
                        one_line = " | ".join(line.strip() for line in txt.splitlines() if line.strip())
                        compact_entries.append(f"{fn}: {one_line}")
                    ctx.append("\n--- FULL KB INDEX ---\n" + "\n".join(compact_entries))
                else:
                    words = {w.lower() for w in re.findall(r"\b[A-Z][a-z]+\b", instr)}
                    matched = [fn for fn in md_files if any(w in fn.lower() for w in words)]
                    snippets = []
                    linked = set()
                    for fn in matched[:8]:
                        txt = open(os.path.join(kb, fn), encoding="utf-8-sig").read().strip()
                        snippets.append(f"[{fn}]\n{txt}")
                        for _, val in re.findall(r"^([A-Za-z_ ]+):\s*(.+)$", txt, re.M):
                            linked.add(val.strip().lower().replace(" ", "_"))
                    for fn in md_files:
                        if fn not in matched and any(ln in fn.lower() for ln in linked if len(ln) > 2):
                            txt = open(os.path.join(kb, fn), encoding="utf-8-sig").read().strip()
                            snippets.append(f"[{fn}]\n{txt}")
                    if snippets:
                        ctx.append("\n--- RELEVANT KB FILES ---\n" + "\n\n".join(snippets[:12]))
    except Exception:
        pass

    return "\n".join(ctx)


def _read_disk_answer(wd, task_dir):
    for d in (wd, task_dir):
        p = os.path.join(d, "answer.txt")
        if os.path.isfile(p):
            try:
                val = open(p, encoding="utf-8-sig").read().strip()
                if val:
                    return val
            except OSError:
                pass
    return None


def _save_trajectory(task_id, model, spent, ans, msgs):
    if os.environ.get("SAVE_TRAJ") != "1":
        return
    traj_dir = os.environ.get("TRAJ_DIR", "trajectories")
    try:
        with _traj_lock:
            os.makedirs(traj_dir, exist_ok=True)
            p = os.path.join(traj_dir, f"{task_id}_{model}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task_id": task_id,
                        "model": model,
                        "spent_tokens": round(spent, 1),
                        "answer": ans,
                        "messages": msgs,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
    except OSError:
        pass


def run_agent(llm, task, model, max_steps=6, max_tokens=800, task_cap=None, orig_test_hash=None):
    """Возвращает (answer|None|'UNSOLVABLE', spent_tokens)."""
    wd = task["_work"]
    is_coding = task.get("task_type") == "coding"
    tests = sorted(glob.glob(os.path.join(wd, "test*.py")) + glob.glob(os.path.join(wd, "*_test.py")))
    if orig_test_hash is None:
        orig_test_hash = _hashes(tests)

    for d in {wd, task["_dir"]}:
        ap = os.path.join(d, "answer.txt")
        if os.path.exists(ap):
            try:
                os.remove(ap)
            except OSError:
                pass

    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _build_initial_context(task, wd)},
    ]
    spent = 0.0

    for _ in range(max_steps):
        if task_cap is not None and spent >= task_cap:
            break
        msg, cost = llm.chat(model, msgs, tools=TOOLS, max_tokens=max_tokens)
        spent += cost

        keep = {"role": "assistant", "content": msg.get("content") or ""}
        calls = msg.get("tool_calls") or []
        if calls:
            keep["tool_calls"] = calls
        msgs.append(keep)

        if not calls:
            disk_ans = _read_disk_answer(wd, task["_dir"])
            if disk_ans and not is_coding:
                clean = normalize_answer(task, disk_ans)
                _save_trajectory(task["task_id"], model, spent, clean, msgs)
                return clean, spent
            if is_coding and tests:
                ok, _ = _verify_coding_tests(wd, tests, orig_test_hash)
                if ok:
                    _save_trajectory(task["task_id"], model, spent, "DONE", msgs)
                    return "DONE", spent
            msgs.append({"role": "user", "content": "Call a tool: bash(cmd) or finish(answer)."})
            continue

        # Сначала выполняем все вызовы bash в этом шаге, и только затем обрабатываем finish
        finish_call = None
        for c in calls:
            fn = c["function"]["name"]
            if fn == "finish":
                finish_call = c
                continue
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            obs = _run(args.get("cmd", ""), wd) if fn == "bash" else "unknown tool"
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": obs or "(no output)"})

        if finish_call is not None:
            try:
                args = json.loads(finish_call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            ans = str(args.get("answer", "")).strip()

            if ans.upper() == "UNSOLVABLE":
                _save_trajectory(task["task_id"], model, spent, "UNSOLVABLE", msgs)
                return "UNSOLVABLE", spent

            if is_coding and tests:
                ok, obs = _verify_coding_tests(wd, tests, orig_test_hash)
                if ok:
                    _save_trajectory(task["task_id"], model, spent, "DONE", msgs)
                    return "DONE", spent
                msgs.append({"role": "tool", "tool_call_id": finish_call["id"], "content": "Tests failed:\n" + obs})
                continue

            disk_ans = _read_disk_answer(wd, task["_dir"])
            if (not ans or ans.upper() == "DONE") and disk_ans and disk_ans.upper() != "DONE":
                ans = disk_ans

            clean = normalize_answer(task, ans)
            _save_trajectory(task["task_id"], model, spent, clean or None, msgs)
            return (clean or None), spent

    # Fallback после исчерпания шагов
    if is_coding and tests:
        ok, _ = _verify_coding_tests(wd, tests, orig_test_hash)
        if ok:
            _save_trajectory(task["task_id"], model, spent, "DONE", msgs)
            return "DONE", spent
    else:
        disk_ans = _read_disk_answer(wd, task["_dir"])
        if disk_ans and disk_ans.upper() != "DONE":
            clean = normalize_answer(task, disk_ans)
            _save_trajectory(task["task_id"], model, spent, clean, msgs)
            return clean, spent

    _save_trajectory(task["task_id"], model, spent, None, msgs)
    return None, spent