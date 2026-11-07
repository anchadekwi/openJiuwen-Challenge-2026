# Харнесс для соревнования (бюджет токенов, 100 задач)

## Запуск
```
docker build -t hack:latest .
docker run --rm -e LLM_BASE_URL -e LLM_API_KEY -v "$TASKS:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp hack:latest \
  python /app/run_agent.py --tasks-dir /data/tasks
```
Переменные: `LLM_BASE_URL`, `LLM_API_KEY` (обязательно); `OUT` (файл/каталог для `solutions.json`,
best-effort), `TOKEN_BUDGET` (если известен), `MODEL_CHEAP`, `MODEL_STRONG`, `WORKERS`, `TIME_LIMIT_S`,
`FASTPATH=0|1`.

## Как устроено
1. `planner.py` — оценка стоимости задачи (таблица приоров по типу + размер workspace, классический метод,
   без LLM) и детектор невыполнимых задач (дешёвые признаки: ссылки на несуществующие файлы, пустой `kb/`,
   нет тестов у coding, URL/интернет в условии). Подозрительные (score ≥ 0.5) не решаются и не тратят токены.
2. Порядок решения — по убыванию `reward × P(выполнима) / оценка_стоимости`.
3. `fastpath.py` — детерминированные решатели шаблонных задач (math / sales.csv / kb), 0 токенов.
   Строгие: при любом непонятном фрагменте возвращают None, и задача идёт к агенту. Отключается `FASTPATH=0`.
4. `agent.py` — ReAct с двумя инструментами (`bash`, `finish`); контекст короткий, вывод инструментов
   обрезается, reasoning не пересылается обратно, лимит шагов и токенов на задачу (3× оценки).
   Для coding `finish` принимается только если тесты проходят и файлы тестов не изменены (проверка локальная, бесплатная).
5. Модель: `qwen3.6-35b-a3b-noreason` по умолчанию; при неудаче — одна повторная попытка на `qwen3.8-27b`.
6. `store.py` — `answer.txt` пишется сразу после решения (в каталог задачи и в `workspace/`), `solutions.json` — атомарно.
7. Учёт расхода: `input/10 + output` по полю `usage` ответа прокси.

## Проверка без прокси
`python tools/mock_proxy.py &` и `LLM_BASE_URL=http://127.0.0.1:8099/v1 LLM_API_KEY=x python run_agent.py --tasks-dir <dir>`.

## Калибровка (TODO)
Приоры в `planner.PRIOR` стартовые. Нужно прогнать тренировочный набор, собрать фактический расход по типам
и положить траектории в `trajectories/`.
