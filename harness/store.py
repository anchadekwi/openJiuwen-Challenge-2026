"""Сохранение ответов. Файл решений обновляется атомарно после КАЖДОЙ решённой задачи."""
import json
import os
import threading

_lock = threading.Lock()
_results = {}


def write_answer(task, answer):
    answer = answer.strip()
    # В регламенте «answer.txt в каталоге задачи», в инструкциях — «в current directory»
    # (= workspace). Пишем в оба места, чтобы не потерять решение.
    for d in {task["_dir"], task["_work"]}:
        with open(os.path.join(d, "answer.txt"), "w", encoding="utf-8") as f:
            f.write(answer + "\n")


def record(task, answer, how, tokens, out_path):
    with _lock:
        _results[task["task_id"]] = {"answer": answer, "how": how, "tokens": round(tokens, 1)}
        if not out_path:
            return
        try:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            tmp = out_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_results, f, ensure_ascii=False, indent=1)
            os.replace(tmp, out_path)
        except OSError:
            pass  # OUT может быть недоступен для записи (контейнер --read-only) — answer.txt уже записан
