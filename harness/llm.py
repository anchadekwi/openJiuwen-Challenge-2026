"""Клиент к LLM-прокси на базе openJiuwen с защитой от Rate Limit (потолок 3 запроса) и точным учётом бюджета."""
import json
import os
import threading
import time
import urllib.error
import urllib.request

try:
    import openjiuwen  # noqa: F401
    OPENJIUWEN_AVAILABLE = True
except ImportError:
    openjiuwen = None
    OPENJIUWEN_AVAILABLE = False

try:
    import requests
except ImportError:
    requests = None


class BudgetExhausted(Exception):
    pass


class Budget:
    def __init__(self, total=None):
        self.total = total
        self.spent = 0.0
        self.exhausted = False
        self.lock = threading.Lock()

    def add(self, prompt_tokens, completion_tokens, extra_reasoning_tokens=0):
        with self.lock:
            cost = (prompt_tokens / 10.0) + completion_tokens + extra_reasoning_tokens
            self.spent += cost
            return cost

    def mark_exhausted(self):
        with self.lock:
            self.exhausted = True

    def left(self):
        if self.exhausted:
            return 0.0
        return None if self.total is None else max(0.0, self.total - self.spent)

    def check(self, reserve=0):
        if self.exhausted:
            raise BudgetExhausted("proxy limit reached")
        left = self.left()
        if left is not None and left < reserve:
            raise BudgetExhausted(f"left={left:.0f}")


def _extract_tokens(data, messages):
    """Извлекает (prompt_tokens, completion_tokens, extra_reasoning_tokens) по правилам регламента."""
    u = data.get("usage") or {}
    pt = u.get("prompt_tokens")
    ct = u.get("completion_tokens")
    tt = u.get("total_tokens")

    msg = (data.get("choices") or [{}])[0].get("message") or {}
    if pt is None or ct is None:
        pt = sum(len(str(m.get("content") or "")) for m in messages) // 3
        ct = len(str(msg.get("content") or "") + str(msg.get("tool_calls") or "")) // 3

    rt = u.get("reasoning_tokens") or 0
    details = u.get("completion_tokens_details") or {}
    if not rt and isinstance(details, dict):
        rt = details.get("reasoning_tokens") or 0

    reasoning_text = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if not rt and reasoning_text:
        est_rt = len(str(reasoning_text)) // 3
        if tt is not None and tt > pt + ct:
            rt = tt - (pt + ct)
        elif ct < est_rt:
            rt = est_rt

    extra_rt = 0
    if rt > 0:
        if tt is not None:
            if tt >= pt + ct + rt - 2:
                extra_rt = rt
        else:
            extra_rt = rt

    return int(pt), int(ct), int(extra_rt)


class LLM:
    def __init__(self, budget):
        base = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8099/v1").strip().rstrip("/")
        self.url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        self.key = os.environ.get("LLM_API_KEY", "").strip()
        self.budget = budget
        self.sess = requests.Session() if requests else None
        # Жёсткий семафор на максимум 2 одновременных HTTP-запроса, чтобы никогда не пробить потолок тарифа (3)
        max_conc = int(os.environ.get("MAX_CONCURRENT_LLM", "2"))
        self.sem = threading.Semaphore(max_conc)

    def _post_json(self, body):
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        with self.sem:
            if self.sess is not None:
                r = self.sess.post(self.url, json=body, timeout=120, headers=headers)
                return r.status_code, r.text
            req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return resp.status, resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode("utf-8", errors="replace")

    def chat(self, model, messages, tools=None, max_tokens=800):
        """Возвращает (message_dict, spent_tokens_for_this_call)."""
        self.budget.check(reserve=min(max_tokens, 100))
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools

        last = None
        for attempt in range(6):
            self.budget.check(reserve=0)
            try:
                status, text = self._post_json(body)
                if status == 200:
                    data = json.loads(text)
                    pt, ct, extra_rt = _extract_tokens(data, messages)
                    cost = self.budget.add(pt, ct, extra_rt)
                    return data["choices"][0]["message"], cost

                txt_low = (text or "").lower()

                # HTTP 429 из-за одновременных запросов или RPM — это НЕ конец бюджета, ждём и повторяем!
                if status == 429 and (
                    "одновременных" in txt_low
                    or "потолок" in txt_low
                    or "rate" in txt_low
                    or "concurrent" in txt_low
                    or "too many" in txt_low
                ):
                    last = f"HTTP 429 (rate/concurrency limit): {text[:150]}"
                    time.sleep(2.0 + 1.5 * attempt)
                    continue

                # Реальное исчерпание бюджета токенов (HTTP 402 или явное сообщение о бюджете/квоте)
                if status in (402, 403) and any(
                    w in txt_low for w in ("budget", "quota", "token limit", "budget_exceeded", "исчерпан")
                ):
                    self.budget.mark_exhausted()
                    raise BudgetExhausted(f"HTTP {status}: {text[:120]}")

                last = f"HTTP {status}: {text[:200]}"
            except BudgetExhausted:
                raise
            except Exception as e:
                last = repr(e)
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(last)