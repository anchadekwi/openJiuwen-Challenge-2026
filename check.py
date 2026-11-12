import json, os, sys
root = sys.argv[1] if len(sys.argv) > 1 else "tasks_run"
ok = bad = 0
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d, "task.json")
    if not os.path.isfile(p):
        continue
    t = json.load(open(p, encoding="utf-8"))
    ans_p = os.path.join(root, d, "answer.txt")
    got = open(ans_p, encoding="utf-8").read().strip() if os.path.isfile(ans_p) else None
    exp = t.get("expected")
    exp = None if exp is None else str(exp).strip()
    if got == exp:
        ok += 1
    else:
        bad += 1
        print(f"FAIL {d}: expected={exp!r} got={got!r}  | {t.get('trap','')[:80]}")
print(f"OK {ok} / FAIL {bad}")