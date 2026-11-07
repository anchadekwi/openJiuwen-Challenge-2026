"""Нулевой по токенам путь: детерминированные решатели для шаблонных задач.

Каждый решатель возвращает строку-ответ либо None (тогда задача идёт к LLM-агенту).
Решатели намеренно строгие: любое непонятное предложение/файл -> None.
"""
import csv
import os
import re


def _math(instr):
    m = re.search(r"starts with (\d+) boxes", instr)
    if not m:
        return None
    n = int(m.group(1))
    body = instr[m.end():].split("How many boxes")[0]
    for s in [x.strip() for x in body.split(".") if x.strip()]:
        s = re.sub(r"^Then ", "", s)
        if (g := re.search(r"split equally into (\d+) warehouses and it keeps one share", s)):
            n //= int(g.group(1))
        elif "doubled" in s:
            n *= 2
        elif (g := re.search(r"ships (\d+) boxes per day for (\d+) days", s)):
            n -= int(g.group(1)) * int(g.group(2))
        elif (g := re.search(r"receives (\d+) deliveries of (\d+) boxes each", s)):
            n += int(g.group(1)) * int(g.group(2))
        else:
            return None
    return str(n)


def _sales(instr, wd):
    m = re.search(r"units of product '(\w+)' were sold in months (\d+) through (\d+) inclusive", instr)
    p = os.path.join(wd, "sales.csv")
    if not m or not os.path.exists(p):
        return None
    prod, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"product", "month", "units"} <= set(rows[0]):
        return None
    return str(sum(int(r["units"]) for r in rows if r["product"] == prod and lo <= int(r["month"]) <= hi))


def _kb_field(kb, prefix, name, field):
    p = os.path.join(kb, f"{prefix}_{name.lower().replace(' ', '_')}.md")
    if not os.path.exists(p):
        return None
    m = re.search(rf"^{field}:\s*(.+)$", open(p, encoding="utf-8").read(), re.M)
    return m.group(1).strip() if m else None


def _search(instr, wd):
    kb = os.path.join(wd, "kb")
    if not os.path.isdir(kb):
        return None
    if (m := re.search(r"In which year was (.+?) born\?", instr)):
        return _kb_field(kb, "person", m.group(1), "Born")
    if (m := re.search(r"headquarters of (.+?)'s employer", instr)):
        emp = _kb_field(kb, "person", m.group(1), "Employer")
        return _kb_field(kb, "company", emp, "Headquarters") if emp else None
    return None


def solve(task, workdir):
    instr = task.get("instruction", "")
    try:
        t = task.get("task_type", "")
        if t == "math":
            return _math(instr)
        if t == "data_analysis":
            return _sales(instr, workdir)
        if t == "search":
            return _search(instr, workdir)
    except Exception:
        return None
    return None
