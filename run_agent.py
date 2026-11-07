#!/usr/bin/env python3
"""Точка входа харнесса (openJiuwen, BUDGET из окружения, до 5 потоков).

Запуск организаторов:
  python /app/run_agent.py --tasks-dir /data/tasks

Переменные: LLM_BASE_URL, LLM_API_KEY, BUDGET, OUT, TASKS.
answer.txt пишется рядом с task.json сразу после решения.
solutions.json перезаписывается после каждого засчитанного ответа.
Поле expected из task.json не читается.
"""
import argparse
import concurrent.futures as cf
import os
import signal
import sys
import time

# Прокси в сети hack-net. Без этого openJiuwen режет внутренний адрес.
os.environ.setdefault("SSRF_PROTECT_ENABLED", "false")

from harness import describe_react_agent, get_openjiuwen_meta
from harness.agent import run_agent
from harness.fastpath import try_fastpath_detailed
from harness.llm import Budget, BudgetExhausted, LLM, _api_base
from harness.planner import load_tasks, plan
from harness.store import dump_everywhere, remove_answer, write_answer

ALLOWED_MODELS = (
    "qwen3.8-27b",
    "qwen3.8-27b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.6-35b-a3b-noreason",
)
# По умолчанию варианты без рассуждений: reasoning тарифицируется как выход.
CHEAP = os.environ.get("MODEL_CHEAP", "qwen3.6-35b-a3b-noreason")
STRONG = os.environ.get("MODEL_STRONG", "qwen3.8-27b-noreason")
RETRY_MIN_REWARD = 2


def _pick_model(name, fallback):
    return name if name in ALLOWED_MODELS else fallback


CHEAP = _pick_model(CHEAP, "qwen3.6-35b-a3b-noreason")
STRONG = _pick_model(STRONG, "qwen3.8-27b-noreason")


def _load_dotenv(path=".env"):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def resolve_out(tasks_dir):
    raw = (os.environ.get("OUT") or "").strip()
    if not raw:
        return os.path.join(tasks_dir, "solutions.json")
    if os.path.isdir(raw) or raw.endswith(("/", "\\")) or not os.path.splitext(raw)[1]:
        return os.path.join(raw, "solutions.json")
    return raw


def solve_one(llm, t):
    cap = 2.5 * t["_est_cost"]
    left = llm.budget.left()
    if left is not None and left < 0.4 * t["_est_cost"]:
        return t["task_id"], "SKIPPED_BUDGET", None, None, 0.0

    status, ans, spent = run_agent(llm, t, CHEAP, max_steps=5, max_tokens=550, task_cap=cap)
    if status in ("SOLVED", "DONE_SOLVED") and ans is not None:
        return t["task_id"], "SOLVED", ans, CHEAP, spent
    if status == "UNSOLVABLE":
        return t["task_id"], "UNSOLVABLE", None, CHEAP, spent

    left = llm.budget.left()
    if t.get("reward", 1) >= RETRY_MIN_REWARD and (left is None or left > 2.0 * t["_est_cost"]):
        status2, ans2, s2 = run_agent(llm, t, STRONG, max_steps=6, max_tokens=750, task_cap=cap * 1.3)
        spent += s2
        if status2 in ("SOLVED", "DONE_SOLVED") and ans2 is not None:
            return t["task_id"], "SOLVED", ans2, STRONG, spent
        if status2 == "UNSOLVABLE":
            return t["task_id"], "UNSOLVABLE", None, STRONG, spent

    return t["task_id"], "FAILED", None, None, spent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.environ.get("TASKS"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "5")))
    env_budget = os.environ.get("BUDGET") or os.environ.get("TOKEN_BUDGET") or "100000"
    ap.add_argument("--budget", type=float, default=float(env_budget))
    a = ap.parse_args()
    if not a.tasks_dir:
        ap.error("нужен --tasks-dir или переменная TASKS")

    if a.out:
        os.environ["OUT"] = a.out
    out_path = resolve_out(a.tasks_dir)

    oj_meta = get_openjiuwen_meta()
    log(
        f"framework={oj_meta['framework']} version={oj_meta.get('version')} "
        f"llm={oj_meta.get('llm')} budget={a.budget:.0f} workers={min(5, a.workers)} "
        f"cheap={CHEAP} strong={STRONG}"
    )

    tasks = load_tasks(a.tasks_dir)
    log(f"tasks loaded: {len(tasks)}")
    queue, skipped = plan(tasks)
    log(f"planned queue={len(queue)} skipped_by_planner={skipped}")

    results = {}
    state = {"results": results, "spent": 0.0, "out": out_path, "tasks_dir": a.tasks_dir}

    def persist():
        written = dump_everywhere(state["out"], state["tasks_dir"], state["results"], state["spent"])
        if written:
            log(f"solutions -> {written[0]}")

    def on_term(signum, _frame):
        log(f"signal {signum}, flushing solutions")
        persist()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_term)
    try:
        signal.signal(signal.SIGINT, on_term)
    except Exception:
        pass

    llm_queue = []
    for t in queue:
        status, val = try_fastpath_detailed(t)
        if status == "SOLVED" and val is not None:
            write_answer(t, val)
            results[t["task_id"]] = {"answer": val, "how": "fastpath", "tokens": 0, "reward": t.get("reward")}
            log(f"fast {t['task_id']} -> {val}")
            persist()
        elif status == "UNSOLVABLE":
            remove_answer(t)
            skipped.append(t["task_id"])
            log(f"unsolvable (fast) {t['task_id']} ({val})")
        else:
            llm_queue.append(t)

    if not llm_queue:
        log(f"done (all handled without LLM). solved={len(results)} unsolvable={len(skipped)}")
        return

    budget = Budget(a.budget)
    llm = LLM(budget)
    try:
        describe_react_agent(CHEAP, _api_base(os.environ.get("LLM_BASE_URL", "")), os.environ.get("LLM_API_KEY", ""))
    except Exception as e:
        log(f"ReActAgent configure skipped: {e!r}")
    workers = max(1, min(5, a.workers))

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(solve_one, llm, t): t for t in llm_queue}
        for fut in cf.as_completed(futs):
            t = futs[fut]
            try:
                tid, status, ans, how, spent = fut.result()
            except BudgetExhausted:
                log("budget exhausted, stopping remaining tasks")
                state["spent"] = budget.spent
                persist()
                break
            except Exception as e:
                log(f"error {t['task_id']} {e!r}")
                continue

            state["spent"] = budget.spent
            if status == "SOLVED" and ans is not None:
                write_answer(t, ans)
                results[tid] = {
                    "answer": ans,
                    "how": how,
                    "tokens": round(spent, 1),
                    "reward": t.get("reward"),
                }
                persist()
                log(f"ok   {tid} [{how}] -> {ans} ({spent:.0f} tok, total={budget.spent:.0f})")
            elif status == "UNSOLVABLE":
                remove_answer(t)
                skipped.append(tid)
                log(f"unsolvable (llm) {tid} ({spent:.0f} tok)")
            else:
                remove_answer(t)
                log(f"miss {tid} ({spent:.0f} tok)")

    log(f"done. solved={len(results)} unsolvable={len(skipped)} spent={budget.spent:.0f}/{a.budget:.0f}")


if __name__ == "__main__":
    sys.exit(main())