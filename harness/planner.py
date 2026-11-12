"""Загрузка задач, детектор заведомо невыполнимых задач и сортировка по reward/cost."""
import glob
import json
import os
import re

# Реалистичные оценки расхода с учётом предзагрузки контекста (в токенах бюджета 100К)
PRIOR_COST = {
    "math": 250,
    "data_analysis": 350,
    "search": 400,
    "coding": 850,
}

IMPOSSIBLE_PATTERNS = [
    # Ссылка сама по себе не признак невыполнимости ("photos: https://..." — просто подпись).
    # Невыполнимо только когда условие требует ИДТИ по ссылке / в интернет.
    r"\b(?:fetch|download|open|visit|browse|scrape|look\s+up|go\s+to|check)\b[^.?!]{0,60}(?:https?|ftp)://\S+",
    r"\b(?:fetch|download|open|visit|browse|scrape|look\s+up|go\s+to|check)\b[^.?!]{0,60}\b(?:api\.|www\.)\S+\.\w{2,}",
    # "current stock" / "today's stock" — обычные слова про склад, не биржа.
    r"\b(?:current|today'?s|live)\s+(?:weather|price|exchange rate|news|bitcoin|stock\s+(?:price|quote|market))\b",
    r"\b(?:cannot be determined|unsolvable|impossible to know)\b",
]


def load_tasks(tasks_dir):
    tasks = []
    for p in sorted(glob.glob(os.path.join(tasks_dir, "*", "task.json"))):
        with open(p, encoding="utf-8-sig") as f:
            t = json.load(f)
        d = os.path.dirname(p)
        t["_dir"] = d
        ws = os.path.join(d, "workspace")
        t["_work"] = ws if os.path.isdir(ws) else d
        # Старый answer.txt с прошлого запуска не должен всплыть как «решение».
        for base in {d, t["_work"]}:
            ans_file = os.path.join(base, "answer.txt")
            if os.path.isfile(ans_file):
                try:
                    os.remove(ans_file)
                except OSError:
                    pass
        tasks.append(t)
    return tasks


def _existing_names(work_dir):
    names = set()
    for root, dirs, files in os.walk(work_dir):
        for n in files + dirs:
            names.add(n)
            names.add(os.path.relpath(os.path.join(root, n), work_dir).replace("\\", "/"))
    return names


def suspicion_score(task):
    """Возвращает score >= 1.0, если задача гарантированно невыполнимая."""
    instr = task.get("instruction") or ""
    ttype = task.get("task_type") or ""
    wd = task["_work"]
    if not instr.strip() or task.get("reward", 1) <= 0:
        return 1.0
    for pat in IMPOSSIBLE_PATTERNS:
        if re.search(pat, instr, re.I):
            return 1.0

    existing = _existing_names(wd)

    # Проверяем упомянутые файлы в инструкции (кроме answer.txt)
    mentioned = set(re.findall(
        r"\b([A-Za-z0-9_./-]+\.(?:csv|tsv|json|md|txt|py|db|sqlite|log|yaml|yml))\b",
        instr,
    ))
    mentioned.discard("answer.txt")
    for m in mentioned:
        base = os.path.basename(m)
        if m not in existing and base not in existing:
            return 1.0

    # Проверка типовых требований к файлам
    if ttype == "coding":
        py_files = [f for f in existing if f.endswith(".py")]
        if not py_files:
            return 1.0
    elif ttype == "search":
        has_docs = any(f.endswith((".md", ".txt", ".json", ".csv")) for f in existing)
        if not has_docs:
            return 1.0
    elif ttype == "data_analysis":
        has_data = any(f.endswith((".csv", ".tsv", ".json", ".db", ".sqlite", ".txt")) for f in existing)
        if not has_data:
            return 1.0

    return 0.0


def plan(tasks):
    """Делит задачи на (очередь по убыванию reward/cost, список пропущенных)."""
    keep, skipped = [], []
    for t in tasks:
        if suspicion_score(t) >= 0.8:
            skipped.append(t["task_id"])
            continue
        cost = PRIOR_COST.get(t.get("task_type"), 500)
        t["_est_cost"] = cost
        t["_value_density"] = float(t.get("reward", 1)) / cost
        keep.append(t)
    keep.sort(key=lambda x: (-x["_value_density"], x["_est_cost"], x["task_id"]))
    return keep, skipped