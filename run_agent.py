#!/usr/bin/env python3
"""Точка входа: python /app/run_agent.py --tasks-dir /data/tasks

Конвейер: загрузка задач -> оценка стоимости/невыполнимости -> порядок по reward/cost ->
(1) нулевые по токенам решатели -> (2) LLM-агент, до 5 сессий параллельно, noreason-модель,
эскалация на рассуждающую модель при неудаче. Ответ сохраняется сразу после каждой задачи.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from harness import fastpath, store
from harness.agent import run_agent
from harness.llm import LLM, Budget, BudgetExhausted
from harness.planner import load_tasks, plan

CHEAP = os.environ.get("MODEL_CHEAP", "qwen3.6-35b-a3b-noreason")
STRONG = os.environ.get("MODEL_STRONG", "qwen3.8-27b")  # с рассуждениями, только для повторной попытки
WORKERS = int(os.environ.get("WORKERS", "5"))
DEADLINE = time.time() + float(os.environ.get("TIME_LIMIT_S", "3300"))  # 55 мин из 60
FASTPATH = os.environ.get("FASTPATH", "1") == "1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.environ.get("TASKS"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.environ.get("OUT")
    out_file = os.path.join(out, "solutions.json") if out and not out.endswith(".json") else out

    tasks = load_tasks(a.tasks_dir)
    log(f"tasks: {len(tasks)}")

    def can_fast(t):
        return FASTPATH and fastpath.solve(t, t["_work"]) is not None

    queue, skipped = plan(tasks, can_fast)
    log(f"queue={len(queue)} skipped_as_unsolvable={[t['task_id'] for t in skipped]}")

    total = os.environ.get("TOKEN_BUDGET")  # если организаторы сообщат бюджет — харнесс сам не вылетит за него
    budget = Budget(float(total) if total else None)
    llm = None
    done = set()

    # 1) бесплатные решатели
    rest = []
    for t in queue:
        ans = fastpath.solve(t, t["_work"]) if FASTPATH else None
        if ans is not None:
            store.write_answer(t, ans)
            store.record(t, ans, "fastpath", 0, out_file)
            done.add(t["task_id"])
            log("fast", t["task_id"], ans)
        else:
            rest.append(t)

    # 2) LLM-агент
    def work(t):
        nonlocal llm
        if time.time() > DEADLINE:
            return
        cap = 3 * t["_cost"]
        spent_total = 0.0
        try:
            for model, steps in ((CHEAP, 8), (STRONG, 6)):
                ans, spent = run_agent(llm, t, model, max_steps=steps, task_cap=cap)
                spent_total += spent
                if ans:
                    store.write_answer(t, ans)
                    store.record(t, ans, model, spent_total, out_file)
                    log("solved", t["task_id"], model, f"{spent_total:.0f}")
                    return
                log("retry", t["task_id"], f"{spent_total:.0f}")
                if time.time() > DEADLINE:
                    return
        except BudgetExhausted:
            log("budget exhausted")
        except Exception as e:  # сбой одной задачи не должен ронять весь прогон
            log("error", t["task_id"], repr(e))

    if rest:
        llm = LLM(budget)
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(work, rest))
    log(f"done. solved={len(done)} fast + llm; spent~{budget.spent:.0f}")


if __name__ == "__main__":
    main()
