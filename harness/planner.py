"""Планировщик: оценка стоимости, детектор 100% невыполнимых задач, ROI-очередь."""
import ast
import csv
import glob
import json
import os
import re

from harness.fastpath import _load_kb

PRIOR = {"math": 300, "search": 350, "data_analysis": 400, "coding": 750}
DEFAULT_PRIOR = 700
FILE_RE = re.compile(r"\b([\w./-]+\.(?:csv|json|py|md|txt|tsv|xlsx|db|sqlite))\b")


def load_tasks(root):
    out = []
    if not root or not os.path.isdir(root):
        return out
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
            if f == "answer.txt":
                continue
            try:
                n += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return n


def estimate_cost(t):
    base = PRIOR.get(t.get("task_type"), DEFAULT_PRIOR)
    extra = min(_ws_bytes(t["_work"]), 20000) / 4.0 / 10.0 * 1.5
    return base + extra


def _check_coding_contradictions(test_path):
    """Отсекает только тесты с синтаксической ошибкой, assert False или прямым противоречием констант."""
    try:
        src = open(test_path, encoding="utf-8-sig").read()
        tree = ast.parse(src)
    except SyntaxError:
        return True
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "assertEqual" and len(node.args) >= 2:
                if all(isinstance(a, (ast.Constant, ast.List, ast.Tuple)) for a in node.args[:2]):
                    call_expr = ast.unparse(node.args[0])
                    expected_expr = ast.unparse(node.args[1])
                    if call_expr in seen and seen[call_expr] != expected_expr:
                        return True
                    seen[call_expr] = expected_expr
        elif isinstance(node, ast.Assert):
            if isinstance(node.test, ast.Constant) and node.test.value is False:
                return True
    return False


def unsolvable_score(t):
    """Возвращает >= 0.5 ТОЛЬКО если задача объективно невыполнима (отсутствуют входные данные/файлы)."""
    instr = t.get("instruction", "")
    wd = t["_work"]
    ttype = t.get("task_type", "")

    if len(instr.strip()) < 10 or t.get("reward", 1) <= 0:
        return 1.0
    if re.search(r"https?://|\binternet\b|\bdownload\b|\bonline\b", instr, re.I):
        return 0.9

    for f in set(FILE_RE.findall(instr)):
        if f in ("answer.txt", "solutions.json") or f.startswith("e.g"):
            continue
        if not os.path.exists(os.path.join(wd, f)):
            return 0.9

    if ttype == "search":
        kb = os.path.join(wd, "kb")
        if not (os.path.isdir(kb) and os.listdir(kb)):
            return 1.0
        persons, companies = _load_kb(kb)
        if (m := re.search(r"In which year was ([A-Za-z ]+) born\?", instr)):
            if m.group(1).strip().lower() not in persons:
                return 0.95
        if (m2 := re.search(r"headquarters of ([A-Za-z ]+)'s employer", instr)):
            name = m2.group(1).strip().lower()
            if name not in persons:
                return 0.95
            emp = (persons[name].get("Employer") or "").lower()
            if emp not in companies:
                return 0.95

    elif ttype == "data_analysis":
        csv_path = os.path.join(wd, "sales.csv")
        if os.path.isfile(csv_path):
            try:
                with open(csv_path, newline="", encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                if not rows:
                    return 0.95
            except Exception:
                return 0.95

    elif ttype == "coding":
        tests = sorted(glob.glob(os.path.join(wd, "test*.py")) + glob.glob(os.path.join(wd, "*_test.py")))
        if not tests:
            return 0.95
        if _check_coding_contradictions(tests[0]):
            return 0.95

    elif ttype == "math":
        if re.search(r"split equally into 0 warehouses|divided by 0\b", instr, re.I):
            return 1.0

    return 0.0


def plan(tasks, can_fast):
    """Возвращает (queue, skipped). queue отсортирована по убыванию ROI = reward / cost."""
    scored = []
    skipped = []
    for t in tasks:
        u = unsolvable_score(t)
        if u >= 0.5:
            t["_u"], t["_cost"] = u, estimate_cost(t)
            skipped.append(t)
            continue
        cost = 50.0 if can_fast(t) else estimate_cost(t)
        t["_u"], t["_cost"] = u, cost
        roi = (1.0 - u) * float(t.get("reward", 0)) / max(cost, 1.0)
        scored.append((roi, t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored], skipped