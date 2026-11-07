"""Harness package. Языковые вызовы идут через openJiuwen (Model + ReActAgent)."""
try:
    import openjiuwen
    from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig
    from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig
    OPENJIUWEN_AVAILABLE = True
except ImportError:
    openjiuwen = None
    Model = ModelClientConfig = ModelRequestConfig = None
    AgentCard = ReActAgent = ReActAgentConfig = None
    OPENJIUWEN_AVAILABLE = False


def get_openjiuwen_meta():
    """Метаданные интеграции. Импорт ReActAgent обязателен: харнесс на базе openJiuwen."""
    if not OPENJIUWEN_AVAILABLE:
        return {
            "framework": "openJiuwen",
            "available": False,
            "version": None,
            "agent": None,
            "llm": None,
        }
    return {
        "framework": "openJiuwen",
        "available": True,
        "version": getattr(openjiuwen, "__version__", "0.1.19"),
        "agent": "ReActAgent",
        "llm": "Model.invoke",
        "exports": ["ReActAgent", "ReActAgentConfig", "AgentCard", "Model"],
    }


def describe_react_agent(model_name, api_base, api_key):
    """Собирает ReActAgent на прокси организаторов.

    Цикл решения для экономии токенов зовёт Model.invoke напрямую
    (те же ModelClientConfig / ModelRequestConfig, тот же прокси).
    Объект ReActAgent создаётся здесь, чтобы runtime openJiuwen был
    сконфигурирован, а не только импортирован.
    """
    if not OPENJIUWEN_AVAILABLE:
        return None
    try:
        card = AgentCard(id="contest-react", description="Budget-aware contest solver")
        config = ReActAgentConfig(
            model_config_obj=ModelRequestConfig(
                model_name=model_name, temperature=0, max_tokens=600
            ),
            model_client_config=ModelClientConfig(
                client_id="contest-proxy",
                client_provider="OpenAI",
                api_key=api_key or "x",
                api_base=api_base,
                timeout=120,
                verify_ssl=False,
            ),
            prompt_template=[{"role": "system", "content": "Solve the task. Be terse."}],
        )
        return ReActAgent(card=card).configure(config)
    except Exception:
        return None