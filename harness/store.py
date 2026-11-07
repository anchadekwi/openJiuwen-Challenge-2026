"""Сохранение и нормализация ответов под строгие чекеры жюри (exact_int, exact_str, unittest)."""
import json
import os
import re
import threading

_lock = threading.Lock()
_results = {}


def normalize_answer(task, raw_answer):
    """Очищает ответ LLM от лишнего текста, кавычек, markdown и точек в конце."""
    if raw_answer is None:
        return ""
    ans = str(raw_answer).strip()
    ttype = task.get("task_type", "")
    instr = task.get("instruction", "")

    if ttype == "coding":
        return "DONE"

    # Если многострочный вывод — берём последнюю непустую строку
    lines = [line.strip() for line in ans.splitlines() if line.strip()]
    if lines:
        ans = lines[-1]

    # Убираем маркдаун-оформление и внешние кавычки
    ans = re.sub(r"^\*\*(.+)\*\*$", r"\1", ans)
    ans = re.sub(r"^`(.+)`$", r"\1", ans)
    ans = ans.strip().strip("'\"")

    # Убираем типичные префиксы ("Answer: 42", "Final answer is 42")
    ans = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*", "", ans, flags=re.I).strip()
    ans = ans.strip("'\"")

    # Проверяем, ожидается ли строго числовой ответ
    expects_number = (
        ttype == "math"
        or task.get("check") == "exact_int"
        or bool(re.search(r"\b(?:How many|In which year|total|sum|count|average|maximum|minimum)\b", instr, re.I))
    )
    if expects_number:
        # Убираем разделители тысяч (например, "1,735" -> "1735")
        ans_no_commas = re.sub(r"(?<=\d),(?=\d{3}\b)", "", ans)
        m_num = re.search(r"[-+]?\d+(?:\.\d+)?", ans_no_commas)
        if m_num:
            num_str = m_num.group(0)
            # Если число вида "158.0" или "158.00", а задача целочисленная — приводим к int
            if re.fullmatch(r"[-+]?\d+\.0+", num_str):
                num_str = num_str.split(".")[0]
            return num_str

    # Для строковых ответов (search и др.) убираем случайную точку в конце предложения
    if ans.endswith(".") and not ans.endswith(".."):
        ans = ans[:-1].strip()

    return ans


def write_answer(task, answer):
    clean = normalize_answer(task, answer)
    # Пишем и в корень задачи, и в workspace/, чтобы гарантировать наличие файла при любой проверке
    for d in {task["_dir"], task["_work"]}:
        with open(os.path.join(d, "answer.txt"), "w", encoding="utf-8") as f:
            f.write(clean + "\n")
    return clean


def record(task, answer, how, tokens, out_path):
    clean = normalize_answer(task, answer)
    with _lock:
        _results[task["task_id"]] = {"answer": clean, "how": how, "tokens": round(tokens, 1)}
        if not out_path:
            return
        try:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            tmp = out_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_results, f, ensure_ascii=False, indent=1)
            os.replace(tmp, out_path)
        except OSError:
            pass