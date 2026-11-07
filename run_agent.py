#!/usr/bin/env python3
"""Точка входа: python /app/run_agent.py --tasks-dir /data/tasks

Золотая середина:
1. FASTPATH=1 по умолчанию решает за 0 токенов ТОЛЬКО 100% совпадающие шаблонные задачи.
2. Все нестандартные, усложнённые и новые задачи автоматически передаются LLM-агенту.
3. WORKERS=2 по умолчанию гарантирует отсутствие ошибки HTTP 429 (потолок тарифа 3 на прокси).
"""
import argparse
import glob
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Автозагрузка локального .env
_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_file):
    for _line in open(_env_file, encoding="utf-8-sig"):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

from harness import fastpath, store  # noqa: E402
from harness.agent import _hashes, restore_workspace, run_agent, snapshot_workspace  # noqa: E402
from harness.llm import LLM, Budget, BudgetExhausted  # noqa: E402
from harness.planner import load_tasks, plan  # noqa: E402

CHEAP = os.environ.get("MODEL_CHEAP", "qwen3.6-35b-a3b-noreason")
STRONG = os.environ.get("MODEL_STRONG", "qwen3.8-27b")
WORKERS = int(os.environ.get("WORKERS", "2"))
DEADLINE = time.time() + float(os.environ.get("TIME_LIMIT_S", "3300"))
FASTPATH = os.environ.get("FASTPATH", "1") == "1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.environ.get("TASKS", "/data/tasks"))
    ap.add_argument("--out", "--out-file", "--output", dest="out", default=None)
    a, extra = ap.parse_known_args()

    out = a.out or os.environ.get("OUT")
    if not out and extra:
        out = extra[0]
    out_file = os.path.join(out, "solutions.json") if out and not out.endswith((".json", ".txt")) else out

    tasks = load_tasks(a.tasks_dir)
    log(f"tasks: {len(tasks)}")

    fast_answers = {}
    if FASTPATH:
        for t in tasks:
            ans = fastpath.solve(t, t["_work"])
            if ans is not None:
                fast_answers[t["task_id"]] = ans

    queue, skipped = plan(tasks, lambda t: t["task_id"] in fast_answers)
    log(f"queue={len(queue)} fast={len(fast_answers)} skipped_as_unsolvable={[t['task_id'] for t in skipped]}")

    total = os.environ.get("TOKEN_BUDGET")
    budget = Budget(float(total) if total else None)
    done = set()
    done_lock = threading.Lock()

    rest = []
    for t in queue:
        tid = t["task_id"]
        if tid in fast_answers:
            ans = fast_answers[tid]
            clean = store.write_answer(t, ans)
            store.record(t, clean, "fastpath", 0, out_file)
            done.add(tid)
            log("fast", tid, clean)
        else:
            rest.append(t)

    def work(t):
        if time.time() > DEADLINE or budget.exhausted:
            return
        wd = t["_work"]
        snap = snapshot_workspace(wd)
        tests = sorted(glob.glob(os.path.join(wd, "test*.py")) + glob.glob(os.path.join(wd, "*_test.py")))
        orig_test_hash = _hashes(tests)

        cap = max(4.0 * t["_cost"], 2500.0)
        spent_total = 0.0
        try:
            for model, steps, m_tok in ((CHEAP, 6, 800), (STRONG, 5, 2048)):
                rem_cap = cap - spent_total
                if rem_cap <= 100 or time.time() > DEADLINE or budget.exhausted:
                    break
                restore_workspace(snap)
                ans, spent = run_agent(
                    llm,
                    t,
                    model,
                    max_steps=steps,
                    max_tokens=m_tok,
                    task_cap=rem_cap,
                    orig_test_hash=orig_test_hash,
                )
                spent_total += spent
                if ans == "UNSOLVABLE":
                    restore_workspace(snap)
                    log("unsolvable_detected_by_llm", t["task_id"], model, f"{spent_total:.0f}")
                    return
                if ans:
                    clean = store.write_answer(t, ans)
                    store.record(t, clean, model, spent_total, out_file)
                    with done_lock:
                        done.add(t["task_id"])
                    log("solved", t["task_id"], model, f"ans={clean}", f"tok={spent_total:.0f}")
                    return
                log("retry", t["task_id"], model, f"{spent_total:.0f}")
            restore_workspace(snap)
        except BudgetExhausted as e:
            restore_workspace(snap)
            log("budget exhausted:", e)
        except Exception as e:
            restore_workspace(snap)
            log("error", t["task_id"], repr(e))

    if rest:
        llm = LLM(budget)
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(work, rest))

    log(f"done. solved={len(done)}/{len(tasks)}; spent={budget.spent:.1f}")


if __name__ == "__main__":
    main()