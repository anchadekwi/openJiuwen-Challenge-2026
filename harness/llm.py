"""Клиент LLM через openJiuwen Model.invoke + учёт токенов по регламенту.

Бюджет в выходных токенах: 10 входных = 1 выходной.
Токены рассуждений считаются выходными, в каком бы поле прокси их ни вернул.
Скидка за кеш провайдера не применяется: вход берётся целиком.
Локальные модели запрещены — только прокси организаторов.
"""
import asyncio
import json
import os
import threading
import time
import urllib.error
import urllib.request

try:
    import openjiuwen  # noqa: F401
    from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig
    OPENJIUWEN_LLM = True
except ImportError:
    openjiuwen = None
    Model = ModelClientConfig = ModelRequestConfig = None
    OPENJIUWEN_LLM = False

try:
    import requests
except ImportError:
    requests = None


class BudgetExhausted(Exception):
    pass


def billable(prompt_tokens, completion_tokens, reasoning_tokens=0):
    """Возвращает (input, output) уже в сырых токенах.

    output = completion, если reasoning уже внутри него (reasoning <= completion),
    иначе completion + reasoning (прокси положил рассуждения отдельным полем).
    Кеш не вычитается.
    """
    prompt = max(0, int(prompt_tokens or 0))
    completion = max(0, int(completion_tokens or 0))
    reasoning = max(0, int(reasoning_tokens or 0))
    if reasoning > completion:
        output = completion + reasoning
    else:
        output = completion
    return prompt, output


class Budget:
    def __init__(self, total=None):
        self.total = float(total) if total is not None else 100000.0
        self.spent = 0.0
        self.lock = threading.Lock()

    def add(self, prompt_tokens, completion_tokens, reasoning_tokens=0):
        prompt, output = billable(prompt_tokens, completion_tokens, reasoning_tokens)
        with self.lock:
            cost = (prompt / 10.0) + output
            self.spent += cost
            return cost

    def left(self):
        with self.lock:
            return None if self.total is None else self.total - self.spent

    def check(self, reserve=0):
        left = self.left()
        if left is not None and left < reserve:
            raise BudgetExhausted(f"left={left:.0f}")


def _api_base(raw):
    base = (raw or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("Переменная окружения LLM_BASE_URL не задана!")
    for suffix in ("/chat/completions", "/v1/chat/completions"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def _reasoning_from_usage(usage):
    if not isinstance(usage, dict):
        return 0
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    values = [
        usage.get("reasoning_tokens") or 0,
        details.get("reasoning_tokens") or 0,
        usage.get("reasoning_output_tokens") or 0,
    ]
    return max(int(v or 0) for v in values)


class LLM:
    """Один вызов = openJiuwen Model.invoke. Если SDK недоступен, прямой HTTP к тому же прокси."""

    def __init__(self, budget):
        # Прокси организаторов в docker-сети hack-net. Защита SSRF openJiuwen
        # режет внутренние адреса и без этого флага не пускает к LLM_BASE_URL.
        os.environ.setdefault("SSRF_PROTECT_ENABLED", "false")
        self.api_base = _api_base(os.environ.get("LLM_BASE_URL", ""))
        self.url = self.api_base + "/chat/completions"
        self.key = os.environ.get("LLM_API_KEY", "").strip()
        self.budget = budget
        self.sess = requests.Session() if requests else None
        self._fallback_logged = False
        self._models = {}
        self._model_lock = threading.Lock()

    def _model(self, model_name, max_tokens):
        key = (model_name, int(max_tokens))
        with self._model_lock:
            cached = self._models.get(key)
            if cached is not None:
                return cached
            client = ModelClientConfig(
                client_id="contest-proxy",
                client_provider="OpenAI",
                api_key=self.key or "x",
                api_base=self.api_base,
                timeout=120,
                verify_ssl=False,
            )
            req = ModelRequestConfig(
                model_name=model_name,
                temperature=0,
                max_tokens=max_tokens,
            )
            inst = Model(model_client_config=client, model_config=req)
            self._models[key] = inst
            return inst

    def _normalize(self, resp):
        content = getattr(resp, "content", "") or ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            content = "".join(parts)
        calls = []
        for tc in getattr(resp, "tool_calls", None) or []:
            name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else "")
            arguments = getattr(tc, "arguments", None) if not isinstance(tc, dict) else tc.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, ensure_ascii=False)
            tid = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
            ttype = getattr(tc, "type", None) if not isinstance(tc, dict) else tc.get("type")
            calls.append({
                "id": tid or "call",
                "type": ttype or "function",
                "function": {"name": name, "arguments": arguments},
            })
        msg = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = calls
        usage = getattr(resp, "usage_metadata", None)
        if usage is None:
            pt, ct, rt = None, None, 0
        else:
            pt = getattr(usage, "input_tokens", None)
            ct = getattr(usage, "output_tokens", None)
            rt = getattr(usage, "reasoning_tokens", 0) or 0
            # cache_tokens есть, но по регламенту скидка не делается
        return msg, pt, ct, int(rt or 0)

    def _chat_openjiuwen(self, model, messages, tools, max_tokens):
        inst = self._model(model, max_tokens)

        async def _call():
            return await inst.invoke(
                messages,
                tools=tools,
                temperature=0,
                max_tokens=max_tokens,
                model=model,
            )

        resp = asyncio.run(_call())
        return self._normalize(resp)

    def _post_json(self, body):
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        if self.sess is not None:
            r = self.sess.post(self.url, json=body, timeout=120, headers=headers)
            return r.status_code, r.text
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace")

    def _chat_http(self, model, messages, tools, max_tokens):
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if tools:
            body["tools"] = tools
        last = None
        for attempt in range(3):
            try:
                status, text = self._post_json(body)
                if status == 200:
                    data = json.loads(text)
                    usage = data.get("usage") or {}
                    pt = usage.get("prompt_tokens")
                    ct = usage.get("completion_tokens")
                    rt = _reasoning_from_usage(usage)
                    if pt is None:
                        pt = sum(len(str(m.get("content") or "")) for m in messages) // 4
                    if ct is None:
                        ct = len(str(data["choices"][0]["message"])) // 4
                    return data["choices"][0]["message"], pt, ct, rt
                if status in (402, 429) and "budget" in text.lower():
                    raise BudgetExhausted(text[:200])
                last = f"HTTP {status}: {text[:200]}"
            except BudgetExhausted:
                raise
            except Exception as e:
                last = repr(e)
            time.sleep(1.2 * (attempt + 1))
        raise RuntimeError(last)

    def chat(self, model, messages, tools=None, max_tokens=600):
        """Возвращает (message_dict, spent_tokens_for_this_call)."""
        self.budget.check(reserve=min(max_tokens, 250))
        pt = ct = rt = None
        msg = None
        if OPENJIUWEN_LLM:
            try:
                msg, pt, ct, rt = self._chat_openjiuwen(model, messages, tools, max_tokens)
            except BudgetExhausted:
                raise
            except Exception as e:
                if not self._fallback_logged:
                    self._fallback_logged = True
                    print(f"openJiuwen Model.invoke fallback to HTTP: {e!r}", flush=True)
        if msg is None:
            msg, pt, ct, rt = self._chat_http(model, messages, tools, max_tokens)
        if pt is None:
            pt = sum(len(str(m.get("content") or "")) for m in messages) // 4
        if ct is None:
            ct = len(str(msg)) // 4
        cost = self.budget.add(pt, ct, rt or 0)
        return msg, cost