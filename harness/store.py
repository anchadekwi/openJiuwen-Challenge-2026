"""Атомарная запись ответов и файла решений.

answer.txt лежит строго рядом с task.json.
Для невыполненной и нерешённой задачи файла нет.
solutions.json перезаписывается после каждого засчитанного ответа.
"""
import json
import os
import threading

_lock = threading.Lock()


def answer_path(task):
    return os.path.join(task["_dir"], "answer.txt")


def write_answer(task, answer):
    """Только рядом с task.json. В workspace/ файл не кладём."""
    ans_str = str(answer).strip()
    target = answer_path(task)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(ans_str + "\n")
    os.replace(tmp, target)


def remove_answer(task):
    """Невыполнимые задачи сдаются отсутствием answer.txt."""
    seen = set()
    for base in (task.get("_dir"), task.get("_work")):
        if not base:
            continue
        p = os.path.join(base, "answer.txt")
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def dump_Aggregate(out_path, results, spent):
    if not out_path:
        return
    with _lock:
        parent = os.path.dirname(os.path.abspath(out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "spent": round(float(spent or 0), 1),
            "solved": len(results),
            "answers": results,
        }
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_path)


def dump_everywhere(primary, tasks_dir, results, spent):
    """Пишет solutions после каждого ответа.

    Контейнер read-only, кроме /tmp и смонтированного каталога задач.
    Если OUT не примонтирован, копия в каталоге задач всё равно переживёт контейнер.
    """
    paths = []
    for p in (primary, os.path.join(tasks_dir, "solutions.json") if tasks_dir else None, "/tmp/solutions.json"):
        if p and p not in paths:
            paths.append(p)
    ok = []
    for p in paths:
        try:
            dump_Aggregate(p, results, spent)
            ok.append(p)
        except OSError:
            continue
    return ok