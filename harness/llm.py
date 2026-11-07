"""?????? ?????? ? LLM-?????? (OpenAI-??????????? /chat/completions) + ???? ???????.
?????? ????????? ?? ???????? ??????????: output + reasoning ????????? 1:1,
input ? 10:1, ??? ?? ???????????.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
try:
    import openjiuwen  # noqa: F401
except ImportError:
    openjiuwen = None
try:
    import requests
except ImportError:
    requests = None
class BudgetExhausted(Exception):
    pass
class Budget:
    def __init__(self, total=None):
        self.total = total  # None = ????? ?????????? ????????, ????? ??????
        self.spent = 0.0
        self.lock = threading.Lock()
    def add(self, prompt_tokens, completion_tokens):
        with self.lock:
            self.spent += prompt_tokens / 10.0 + completion_tokens
            return self.spent
    def left(self):
        return None if self.total is None else self.total - self.spent
    def check(self, reserve=0):
        left = self.left()
        if left is not None and left < reserve:
            raise BudgetExhausted(f"left={left:.0f}")
class LLM:
    def __init__(self, budget):
        base = os.environ["LLM_BASE_URL"].strip().rstrip("/")
        self.url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        self.key = os.environ.get("LLM_API_KEY", "").strip()
        self.budget = budget
        self.sess = requests.Session() if requests else None
    def _post_json(self, body):
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        if self.sess is not None:
            r = self.sess.post(self.url, json=body, timeout=180, headers=headers)
            return r.status_code, r.text
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace")
    def chat(self, model, messages, tools=None, max_tokens=700):
        """?????????? (message_dict, spent_tokens_for_this_call)."""
        self.budget.check(reserve=max_tokens)
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
        last = None
        for attempt in range(3):
            try:
                status, text = self._post_json(body)
                if status == 200:
                    data = json.loads(text)
                    u = data.get("usage") or {}
                    pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
                    if pt is None or ct is None:
                        pt = sum(len(str(m.get("content") or "")) for m in messages) // 3
                        ct = len(str(data["choices"][0]["message"])) // 3
                    self.budget.add(pt, ct)
                    return data["choices"][0]["message"], pt / 10.0 + ct
                last = f"HTTP {status}: {text[:200]}"
            except Exception as e:
                last = repr(e)
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(last)
