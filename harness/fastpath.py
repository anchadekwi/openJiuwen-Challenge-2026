"""Строгий fastpath: 0 токенов только при полном совпадении шаблона.

Любое непонятное предложение -> UNKNOWN (дальше решает LLM).
UNSOLVABLE — только когда шаблон распознан, а сущности/колонки в данных нет.
"""
import csv
import glob
import os
import re


def try_fastpath_detailed(task):
    t = task.get("task_type")
    try:
        if t == "math":
            return _math(task["instruction"])
        if t == "data_analysis":
            return _data(task["instruction"], task["_work"])
        if t == "search":
            return _search(task["instruction"], task["_work"])
    except Exception:
        pass
    return "UNKNOWN", None


def try_fastpath(task):
    status, val = try_fastpath_detailed(task)
    return val if status == "SOLVED" else None


def _sentences(body):
    parts = []
    for chunk in re.split(r"(?<=[.?!])\s+", body.strip()):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def _norm_sentence(s):
    return re.sub(r"\s+", " ", s).strip().rstrip(".?!").strip()


# Каждое предложение операции должно совпасть целиком.
_RE_SPLIT = re.compile(
    r"^(?:then\s+)?(?:the\s+stock|it|the\s+\w+)\s+is\s+split\s+equally\s+into\s+(\d+)\s+\w+\s+"
    r"and\s+(?:it|the\s+\w+)\s+keeps\s+one\s+share(?:\s*\(rounded\s+down\))?$",
    re.I,
)
_RE_DOUBLE = re.compile(
    r"^(?:then\s+)?(?:the\s+stock|it|the\s+\w+)\s+is\s+doubled"
    r"(?:\s+in\s+size|\s+by\s+merging\s+with\s+a\s+sister\s+warehouse)?$",
    re.I,
)
_RE_TRIPLE = re.compile(
    r"^(?:then\s+)?(?:the\s+stock|it|the\s+\w+)\s+is\s+tripled(?:\s+in\s+size)?$",
    re.I,
)
_RE_SHIP = re.compile(
    r"^(?:then\s+)?(?:it|he|she|the\s+\w+)\s+ships\s+(-?\d+)\s+\w+\s+per\s+day\s+for\s+(-?\d+)\s+days?$",
    re.I,
)
_RE_RECV = re.compile(
    r"^(?:then\s+)?(?:it|he|she|the\s+\w+)\s+receives\s+(-?\d+)\s+deliver(?:y|ies)\s+of\s+(-?\d+)\s+\w+(?:\s+each)?$",
    re.I,
)
_RE_ADD = re.compile(
    r"^(?:then\s+)?(?:it|he|she|the\s+\w+)\s+adds\s+(-?\d+)\s+more(?:\s+\w+)?$",
    re.I,
)
_RE_LOSE = re.compile(
    r"^(?:then\s+)?(?:it|he|she|the\s+\w+)\s+loses\s+(-?\d+)(?:\s+\w+)?$",
    re.I,
)
_RE_START = re.compile(
    r"^(?:a|an|the)\s+.+?\s+(?:starts\s+with|initially\s+has|has)\s+(-?\d+)\s+boxes?$",
    re.I,
)


def _apply_op(sentence, val):
    s = _norm_sentence(sentence)
    m = _RE_SPLIT.fullmatch(s)
    if m:
        divisor = int(m.group(1))
        if divisor <= 0:
            return None
        return val // divisor
    if _RE_DOUBLE.fullmatch(s):
        return val * 2
    if _RE_TRIPLE.fullmatch(s):
        return val * 3
    m = _RE_SHIP.fullmatch(s)
    if m:
        return val - int(m.group(1)) * int(m.group(2))
    m = _RE_RECV.fullmatch(s)
    if m:
        return val + int(m.group(1)) * int(m.group(2))
    m = _RE_ADD.fullmatch(s)
    if m:
        return val + int(m.group(1))
    m = _RE_LOSE.fullmatch(s)
    if m:
        return val - int(m.group(1))
    return None


def _math(instr):
    body = re.split(
        r"\b(?:Compute it|Work only|Use bash|When you are done)\b",
        instr,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    raw = _sentences(body)
    if len(raw) < 2 or "?" not in raw[-1]:
        return "UNKNOWN", None

    question = _norm_sentence(raw[-1])
    if not re.search(r"\bhow many\b", question, re.I) or not re.search(r"\bbox", question, re.I):
        return "UNKNOWN", None

    start = _RE_START.fullmatch(_norm_sentence(raw[0]))
    if not start:
        return "UNKNOWN", None
    val = int(start.group(1))

    for sentence in raw[1:-1]:
        nxt = _apply_op(sentence, val)
        if nxt is None:
            return "UNKNOWN", None
        val = nxt
    return "SOLVED", str(val)


def _read_csv(csv_path):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    norm = []
    for r in rows:
        norm.append({(k or "").strip().lower(): (v or "").strip() for k, v in r.items() if k})
    return norm


def _data(instr, wd):
    csv_path = os.path.join(wd, "sales.csv")
    if not os.path.isfile(csv_path):
        return "UNKNOWN", None

    rows = _read_csv(csv_path)
    if not rows:
        return "UNSOLVABLE", "sales.csv is empty"

    cols = set(rows[0].keys())
    known_prods = {r.get("product", "").strip().lower() for r in rows}
    known_regs = {r.get("region", "").strip().lower() for r in rows}

    # Число строк в условии бывает любым: "(20 rows)", "(300 rows)", ...
    # Поэтому вопрос ищется по всему тексту, а не после буквального "(20 rows).".
    m = re.search(
        r"How\s+many\s+units\s+of\s+product\s+['\"]([^'\"]+)['\"]\s+"
        r"were\s+sold\s+in\s+months\s+(\d+)\s+through\s+(\d+)\s+inclusive\?",
        instr,
        re.I,
    )
    if not m:
        return "UNKNOWN", None

    prod = m.group(1).strip().lower()
    if prod not in known_prods:
        return "UNSOLVABLE", f"product '{m.group(1)}' does not exist in sales.csv"

    rm = re.search(r"\bregion\s+['\"]([^'\"]+)['\"]", instr, re.I)
    if rm and rm.group(1).strip().lower() not in known_regs:
        return "UNSOLVABLE", f"region '{rm.group(1)}' does not exist in sales.csv"

    if "units" not in cols or "month" not in cols or "product" not in cols:
        return "UNSOLVABLE", "required columns missing in sales.csv"

    m1, m2 = int(m.group(2)), int(m.group(3))
    if m1 > m2:
        return "UNKNOWN", None

    total = 0
    for r in rows:
        if r.get("product", "").strip().lower() != prod:
            continue
        month = int(r["month"])
        if m1 <= month <= m2:
            total += int(r["units"])
    return "SOLVED", str(total)


def _load_kb(wd):
    kb_dir = os.path.join(wd, "kb")
    if not os.path.isdir(kb_dir):
        return None
    entities = {}
    for p in glob.glob(os.path.join(kb_dir, "*.md")):
        name = None
        kv = {}
        with open(p, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# "):
                    name = line[2:].strip()
                elif ":" in line and not line.startswith("#"):
                    k, v = line.split(":", 1)
                    kv[k.strip().lower()] = v.strip()
        if name:
            entities[name.lower()] = kv
    return entities


def _search(instr, wd):
    kb = _load_kb(wd)
    if kb is None:
        return "UNKNOWN", None
    if not kb:
        return "UNSOLVABLE", "kb directory is empty"

    m1 = re.search(
        r"In which city is the headquarters of ([^?']+?)'s employer\?",
        instr,
        re.I,
    )
    if m1:
        person = re.sub(r"\s+", " ", m1.group(1)).strip().lower()
        if person not in kb:
            return "UNSOLVABLE", f"person '{person}' not in kb"
        emp = kb[person].get("employer")
        if not emp or emp.lower() not in kb:
            return "UNSOLVABLE", f"employer '{emp}' not in kb"
        hq = kb[emp.lower()].get("headquarters")
        if not hq:
            return "UNSOLVABLE", f"headquarters missing for '{emp}'"
        return "SOLVED", hq

    m2 = re.search(r"In which year was ([^?']+?) (born|founded)\?", instr, re.I)
    if m2:
        ent = re.sub(r"\s+", " ", m2.group(1)).strip().lower()
        field = m2.group(2).lower()
        if ent not in kb:
            return "UNSOLVABLE", f"entity '{ent}' not in kb"
        val = kb[ent].get(field)
        if not val:
            return "UNSOLVABLE", f"field '{field}' missing for '{ent}'"
        return "SOLVED", val

    return "UNKNOWN", None