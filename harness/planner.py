"""Планировщик: оценка стоимости, детектор невыполнимых задач, порядок решения.

Оценка стоимости — классическая (таблица приоров по типу + размер workspace).
ВАЖНО: числа в PRIOR — стартовые, их надо откалибровать по траекториям
на тренировочном наборе (см. README, раздел «Калибровка»).
"""
import json
import os
import re

# оценка расхода в «выходных токенах» (input/10 + output) на одну задачу
PRIOR = {"math": 1500, "search": 3500, "data_analysis": 3500, "coding": 6000}
DEFAULT_PRIOR = 5000
FILE_RE = re.compile(r"\b([\w./-]+\.(?:csv|json|py|md|txt|tsv|xlsx))\b")


def load_tasks(root):
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        p = os.path.join(d, "task.json")
        if os.path.isfile(p):
            t = json.load(open(p, encoding="utf-8-sig"))
            t["_dir"] = d
            t["_work"] = os.path.join(d, "workspace") if os.path.isdir(os.path.join(d, "workspace")) else d
            out.append(t)
    return out


def _ws_bytes(wd):
    n = 0
    for r, _, fs in os.walk(wd):
        for f in fs:
            n += os.path.getsize(os.path.join(r, f))
    return n


def estimate_cost(t):
    base = PRIOR.get(t.get("task_type"), DEFAULT_PRIOR)
    # агент читает файлы -> часть workspace попадёт в контекст (÷4 символов/токен, ÷10 вес input)
    extra = min(_ws_bytes(t["_work"]), 30000) / 4 / 10 * 3
    return base + extra


def unsolvable_score(t):
    """0..1 — подозрение, что задача невыполнима. Только дешёвые признаки без LLM."""
    instr = t.get("instruction", "")
    wd = t["_work"]
    s = 0.0
    if len(instr.strip()) < 10:
        s += 1
    if re.search(r"https?://|\binternet\b|\bdownload\b|\bonline\b", instr, re.I):
        s += 0.6
    for f in set(FILE_RE.findall(instr)):
        if f == "answer.txt":
            continue
        if not os.path.exists(os.path.join(wd, f)):
            s += 0.6
    if re.search(r"\bkb/", instr):
        kb = os.path.join(wd, "kb")
        if not (os.path.isdir(kb) and os.listdir(kb)):
            s += 0.8
    if t.get("task_type") == "coding" and not any(f.startswith("test") for f in os.listdir(wd)):
        s += 0.8
    return min(s, 1.0)


def plan(tasks, can_fast):
    """Возвращает (queue, skipped). queue — по убыванию reward/стоимость."""
    scored = []
    skipped = []
    for t in tasks:
        u = unsolvable_score(t)
        cost = 0.0 if can_fast(t) else estimate_cost(t)
        t["_u"], t["_cost"] = u, cost
        if u >= 0.5:
            skipped.append(t)
            continue
        # ожидаемая ценность на токен; поправка на вероятность выполнимости
        scored.append(((1 - u) * t.get("reward", 0) / max(cost, 1.0), t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored], skipped
