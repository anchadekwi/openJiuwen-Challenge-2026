"""Строгий fastpath: 0 токенов только при полном совпадении шаблона.

Любое непонятное предложение -> UNKNOWN (дальше решает LLM).
UNSOLVABLE — только когда шаблон распознан, а сущности/колонки в данных нет.
"""
import csv
import glob
import io
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
_RE_GROW = re.compile(
    r"^(?:then\s+)?(?:the\s+stock|it|the\s+\w+)\s+(?:grows|increases)\s+by\s+(\d+)\s+percent$",
    re.I,
)
_RE_SHRINK = re.compile(
    r"^(?:then\s+)?(?:the\s+stock|it|the\s+\w+)\s+(?:shrinks|decreases)\s+by\s+(\d+)\s+percent$",
    re.I,
)
_RE_PCT_LOST = re.compile(
    r"^(?:then\s+)?(\d+)\s+percent\s+of\s+the\s+boxes\s+are\s+damaged\s+and\s+thrown\s+away$",
    re.I,
)
_RE_START = re.compile(
    r"^(?:a|an|the)\s+.+?\s+(?:starts\s+with|initially\s+has|has)\s+(-?\d+)\s+boxes?$",
    re.I,
)


_RE_FINAL_Q = re.compile(
    r"^how\s+many\s+boxes\s+(?:remain|are\s+left|are\s+there"
    r"|are\s+(?:in|at)\s+(?:the\s+)?[\w' -]{1,30}?"
    r"|(?:does|do)\s+(?:it|the\s+\w+)\s+(?:hold|have|contain))"
    r"(?:\s+(?:in\s+the\s+end|at\s+the\s+end|at\s+the\s+end\s+of\s+the\s+process|now))?$",
    re.I,
)


def _apply_op(sentence, val):
    s = _norm_sentence(sentence)
    m = _RE_SPLIT.fullmatch(s)
    if m:
        divisor = int(m.group(1))
        if divisor == 0:
            return "ZERO"
        if divisor < 0:
            return None
        return val // divisor
    if _RE_DOUBLE.fullmatch(s):
        return val * 2
    if _RE_TRIPLE.fullmatch(s):
        return val * 3
    # Проценты: точная целочисленная арифметика. Нецелый результат -> правило округления
    # неизвестно, не гадаем (None -> задача уйдёт к LLM).
    for rx, sign in ((_RE_GROW, 1), (_RE_SHRINK, -1), (_RE_PCT_LOST, -1)):
        m = rx.fullmatch(s)
        if m:
            num = val * (100 + sign * int(m.group(1)))
            return num // 100 if num % 100 == 0 else None
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
    # Только вопрос про итоговое количество. "Сколько отгружено / потеряно / было ДО последней
    # операции / после первых двух" -> не наш шаблон, пусть решает LLM.
    if not _RE_FINAL_Q.fullmatch(question):
        return "UNKNOWN", None

    start = _RE_START.fullmatch(_norm_sentence(raw[0]))
    if not start:
        return "UNKNOWN", None
    val = int(start.group(1))

    for sentence in raw[1:-1]:
        nxt = _apply_op(sentence, val)
        if nxt == "ZERO":
            return "UNSOLVABLE", "division by zero"
        if nxt is None:
            return "UNKNOWN", None
        val = nxt
    return "SOLVED", str(val)


def _read_csv(csv_path):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    header = text.splitlines()[0] if text.strip() else ""
    delim = max([",", ";", "\t"], key=header.count)
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    norm = []
    for r in rows:
        norm.append({(k or "").strip().lower(): (v or "").strip() for k, v in r.items() if k})
    return norm


_RE_REGION_FILTER = re.compile(r"^Count\s+only\s+rows\s+from\s+region\s+['\"]([^'\"]+)['\"]$", re.I)
_RE_DATA_Q = re.compile(
    r"^How\s+many\s+units\s+of\s+product\s+['\"]([^'\"]+)['\"]\s+"
    r"were\s+sold\s+in\s+months\s+(\d+)\s+through\s+(\d+)\s+inclusive$",
    re.I,
)


def _data(instr, wd):
    csv_path = os.path.join(wd, "sales.csv")
    if not os.path.isfile(csv_path):
        return "UNKNOWN", None

    # Разбираем условие по предложениям: каждое должно быть опознано, иначе -> LLM.
    question = None
    region = None
    for s in _sentences(instr):
        n = _norm_sentence(s)
        mq = _RE_DATA_Q.fullmatch(n)
        mr = _RE_REGION_FILTER.fullmatch(n)
        if mq:
            question = mq
        elif mr:
            region = mr.group(1).strip().lower()
        elif re.match(r"^The\s+file\s+sales\.csv\s+contains\b.*\bcolumns\b", n, re.I):
            pass
        elif re.match(r"^Work\s+only\s+inside\s+the\s+task\s+directory$", n, re.I):
            pass
        elif re.match(r"^When\s+you\s+are\s+done,\s+write\s+the\s+answer\s+to\s+answer\.txt$", n, re.I):
            pass
        else:
            return "UNKNOWN", None
    if question is None:
        return "UNKNOWN", None

    rows = _read_csv(csv_path)
    if not rows:
        return "UNSOLVABLE", "sales.csv is empty"

    cols = set(rows[0].keys())
    known_prods = {r.get("product", "").strip().lower() for r in rows}
    known_regs = {r.get("region", "").strip().lower() for r in rows}

    prod = question.group(1).strip().lower()
    if "product" in cols and prod not in known_prods:
        return "UNSOLVABLE", f"product '{question.group(1)}' does not exist in sales.csv"
    if region is not None:
        if "region" not in cols:
            return "UNSOLVABLE", "region column missing in sales.csv"
        if region not in known_regs:
            return "UNSOLVABLE", f"region '{region}' does not exist in sales.csv"

    if "units" not in cols or "month" not in cols or "product" not in cols:
        return "UNSOLVABLE", "required columns missing in sales.csv"

    m1, m2 = int(question.group(2)), int(question.group(3))
    if m1 > m2:
        return "UNKNOWN", None

    total = 0
    for r in rows:
        if r.get("product", "").strip().lower() != prod:
            continue
        if region is not None and r.get("region", "").strip().lower() != region:
            continue
        month = int(r["month"])
        if m1 <= month <= m2:
            total += int(r["units"])
    return "SOLVED", str(total)


_KB_KEYS = {"employer", "born", "role", "headquarters", "founded", "industry", "aka"}
_AMBIG = "__ambiguous__"


def _load_kb(wd):
    """name(lower) -> {"kv": {...}, "odd": bool} | _AMBIG. Второй результат — весь текст kb."""
    kb_dir = os.path.join(wd, "kb")
    if not os.path.isdir(kb_dir):
        return None, ""
    entities = {}
    alltext = []
    for p in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
        name = None
        kv = {}
        odd = False  # нестандартные ключи, повтор ключей или проза -> доверять нельзя
        with open(p, encoding="utf-8-sig") as f:
            for line in f:
                alltext.append(line)
                line = line.strip()
                if not line:
                    continue
                if line.startswith("# "):
                    name = line[2:].strip()
                    continue
                m = re.fullmatch(r"([A-Za-z_]+):\s*(.+)", line)
                if m and m.group(1).lower() in _KB_KEYS:
                    k = m.group(1).lower()
                    if k in kv:
                        odd = True
                    kv[k] = m.group(2).strip()
                else:
                    odd = True
        if name:
            key = name.lower()
            if key in entities:
                entities[key] = _AMBIG  # два документа с одним именем (два "John Smith")
            else:
                entities[key] = {"kv": kv, "odd": odd}
    return entities, "".join(alltext).lower()


_RE_SEARCH_OK = (
    re.compile(r"^Use\s+only\s+the\s+documents\s+in\s+kb/?$", re.I),
    re.compile(r"^When\s+you\s+are\s+done,\s+write\s+only\s+the\s+answer\s+to\s+answer\.txt$", re.I),
)


def _only_standard_extras(instr, question_text):
    """Кроме вопроса допустимы только два стандартных предложения; любое другое
    ("Use the John Smith born in 1975", "Use the most recent information") -> не наш шаблон."""
    rest = instr.replace(question_text, " ")
    for s in _sentences(rest):
        n = _norm_sentence(s)
        if n and not any(r.fullmatch(n) for r in _RE_SEARCH_OK):
            return False
    return True


def _year(value):
    years = re.findall(r"\b(1\d{3}|20\d{2})\b", value or "")
    return years[0] if len(years) == 1 else None


def _entity(kb, alltext, name):
    """('ok', entity) | ('missing', None) | ('unknown', None)"""
    e = kb.get(name)
    if e == _AMBIG:
        return "unknown", None
    if e is None:
        # имени нет заголовком, но оно может встречаться в тексте (проза, алиасы) -> LLM
        return ("unknown", None) if name in alltext else ("missing", None)
    if e["odd"] or not e["kv"]:
        return "unknown", None
    return "ok", e


def _search(instr, wd):
    kb, alltext = _load_kb(wd)
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
        if not _only_standard_extras(instr, m1.group(0)):
            return "UNKNOWN", None
        person = re.sub(r"\s+", " ", m1.group(1)).strip().lower()
        st, pe = _entity(kb, alltext, person)
        if st == "unknown":
            return "UNKNOWN", None
        if st == "missing":
            return "UNSOLVABLE", f"person '{person}' not in kb"
        emp = pe["kv"].get("employer")
        if not emp:
            return "UNSOLVABLE", f"employer missing for '{person}'"
        el = emp.lower()
        if el not in kb:
            # "Orion" vs "Orion Labs", алиасы (aka) -> пусть решает LLM
            fuzzy = any(
                el in n or n in el
                or (isinstance(v, dict) and el == v["kv"].get("aka", "").lower())
                for n, v in kb.items()
            )
            return ("UNKNOWN", None) if fuzzy else ("UNSOLVABLE", f"employer '{emp}' not in kb")
        st, ee = _entity(kb, alltext, el)
        if st != "ok":
            return "UNKNOWN", None
        hq = ee["kv"].get("headquarters")
        if not hq:
            return "UNSOLVABLE", f"headquarters missing for '{emp}'"
        return "SOLVED", hq

    m2 = re.search(r"In which year was ([^?']+?) (born|founded)\?", instr, re.I)
    if m2:
        if not _only_standard_extras(instr, m2.group(0)):
            return "UNKNOWN", None
        ent = re.sub(r"\s+", " ", m2.group(1)).strip().lower()
        field = m2.group(2).lower()
        if ent.startswith("the ") or " of " in ent:  # "the employer of X" — многошаговый вопрос
            return "UNKNOWN", None
        st, e = _entity(kb, alltext, ent)
        if st == "unknown":
            return "UNKNOWN", None
        if st == "missing":
            return "UNSOLVABLE", f"entity '{ent}' not in kb"
        val = e["kv"].get(field)
        if not val:
            return "UNSOLVABLE", f"field '{field}' missing for '{ent}'"
        y = _year(val)
        return ("SOLVED", y) if y else ("UNKNOWN", None)

    return "UNKNOWN", None
