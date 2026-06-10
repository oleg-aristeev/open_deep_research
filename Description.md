**1. Назначение проекта**

Open Deep Research — Python/LangGraph агент для автоматизированного deep research: принимает исследовательский вопрос, уточняет его при необходимости, ищет источники через search/API/MCP tools, сжимает найденные материалы и генерирует итоговый markdown-отчёт. Это подтверждается README: проект описан как configurable open-source deep research agent для разных model providers, search tools и MCP servers [README.md](open_deep_research/README.md:5).

Основной пользователь: разработчик или оператор LangGraph Studio/Open Agent Platform, который хочет запустить настраиваемого исследовательского агента. Сценарии: локальный запуск в LangGraph Studio, hosted deployment на LangGraph Platform/OAP, пакетная оценка на LangSmith Deep Research Bench [README.md](open_deep_research/README.md:43), [README.md](open_deep_research/README.md:85).

Результат: финальный markdown-отчёт в поле `final_report` и AI-сообщении, плюс служебные `raw_notes`/`notes` для оценки groundedness [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:655), [state.py](open_deep_research/src/open_deep_research/state.py:65).

**2. Общая архитектурная модель**

Главная архитектура — один LangGraph-граф `Deep Researcher`, опубликованный через `langgraph.json` как `./src/open_deep_research/deep_researcher.py:deep_researcher` [langgraph.json](open_deep_research/langgraph.json:3). Граф состоит из основного workflow и двух подграфов: supervisor-subgraph и researcher-subgraph [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:353), [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:589).

```text
[User messages / LangGraph Studio / API]
        ↓
[clarify_with_user]
        ↓
[write_research_brief]
        ↓
[research_supervisor subgraph]
        ↓
[parallel researcher_subgraph instances]
        ↓
[compress_research]
        ↓
[final_report_generation]
        ↓
[Markdown final_report + messages + raw_notes]
```

Вход: `messages` из `MessagesState`, runtime-конфигурация LangGraph/RunnableConfig и переменные окружения. Выход: отчёт, сообщения, исследовательские заметки. Внешние зависимости: LangGraph/LangChain, LLM providers, Tavily/OpenAI/Anthropic web search, MCP servers, LangSmith для evaluation, Supabase для OAP auth.

**3. Структура репозитория**

| Путь | Назначение | Почему важно |
| ---- | ---------- | ------------ |
| [README.md](open_deep_research/README.md:21) | Quickstart, конфигурация, evaluation, deployment | Лучший старт для запуска и понимания публичного сценария |
| [pyproject.toml](open_deep_research/pyproject.toml:1) | Python package, зависимости, ruff config | Показывает стек, Python `>=3.10`, dev tools |
| [uv.lock](open_deep_research/uv.lock) | Lockfile uv | Фиксирует версии зависимостей |
| [langgraph.json](open_deep_research/langgraph.json:1) | LangGraph deployment config | Главный runtime entry point и auth hook |
| [.env.example](open_deep_research/.env.example:1) | Шаблон ключей | Показывает обязательные/опциональные внешние сервисы |
| [src/open_deep_research/deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:1) | Основной граф | Центральный файл проекта |
| [src/open_deep_research/configuration.py](open_deep_research/src/open_deep_research/configuration.py:38) | Runtime-настройки | Управляет моделями, search, MCP, concurrency |
| [src/open_deep_research/state.py](open_deep_research/src/open_deep_research/state.py:15) | State и structured outputs | Контракты между nodes/tools |
| [src/open_deep_research/utils.py](open_deep_research/src/open_deep_research/utils.py:43) | Search, MCP, tokens, API keys | Интеграционный слой |
| [src/open_deep_research/prompts.py](open_deep_research/src/open_deep_research/prompts.py:3) | Prompt templates | Определяет поведение агента |
| [src/security/auth.py](open_deep_research/src/security/auth.py:21) | LangGraph auth middleware | Supabase JWT и owner-based access |
| [tests/](open_deep_research/tests) | LangSmith evaluation scripts | Не unit tests, а benchmark/eval слой |
| [tests/expt_results/](open_deep_research/tests/expt_results) | JSONL результаты benchmark | Примеры готовых отчётов для Deep Research Bench |
| [examples/](open_deep_research/examples) | Примеры отчётов | Полезны для понимания ожидаемого output |
| [src/legacy/](open_deep_research/src/legacy) | Старые реализации | Workflow и multi-agent варианты, не текущий главный граф |
| [.github/workflows/](open_deep_research/.github/workflows) | GitHub workflows | Claude review/assistant, но не тестовый CI |

**4. Ключевые компоненты**

`Configuration`: Pydantic-модель runtime-настроек. Отвечает за clarification, concurrency, search API, модели для summarization/research/compression/final report, MCP config [configuration.py](open_deep_research/src/open_deep_research/configuration.py:38). Значения читаются из env или `config["configurable"]` [configuration.py](open_deep_research/src/open_deep_research/configuration.py:236).

`AgentState`, `SupervisorState`, `ResearcherState`: состояние основного графа, supervisor и researcher agents [state.py](open_deep_research/src/open_deep_research/state.py:65). Structured outputs/tools: `ClarifyWithUser`, `ResearchQuestion`, `ConductResearch`, `ResearchComplete`, `Summary` [state.py](open_deep_research/src/open_deep_research/state.py:15).

`clarify_with_user`: решает, нужно ли задавать уточняющий вопрос; может завершить граф вопросом пользователю или продолжить research brief [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:60).

`write_research_brief`: превращает диалог в структурированный research brief и инициализирует supervisor-сообщения [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:118).

`supervisor` и `supervisor_tools`: supervisor планирует research, вызывает `ConductResearch`, `think_tool` или `ResearchComplete`; `supervisor_tools` запускает researcher subgraphs параллельно через `asyncio.gather` [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:178), [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:288).

`researcher` и `researcher_tools`: отдельный агент получает topic, собирает tools через `get_all_tools`, вызывает search/MCP/think tools и завершает в compression [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:365), [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:435).

`compress_research`: очищает и сохраняет найденные данные без потери источников; возвращает `compressed_research` и `raw_notes` [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:511).

`final_report_generation`: собирает `notes`, применяет final report prompt, обрабатывает token-limit retry и возвращает итоговый отчёт [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:607).

`utils.py`: Tavily search + summarization [utils.py](open_deep_research/src/open_deep_research/utils.py:43), MCP loading/auth [utils.py](open_deep_research/src/open_deep_research/utils.py:449), tool assembly [utils.py](open_deep_research/src/open_deep_research/utils.py:569), API-key routing [utils.py](open_deep_research/src/open_deep_research/utils.py:892).

**5. Жизненный цикл выполнения**

Основной сценарий LangGraph:

1. Запуск через `uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking` [README.md](open_deep_research/README.md:45).
2. LangGraph читает `langgraph.json`: graph `Deep Researcher`, `.env`, dependency `"."`, auth handler [langgraph.json](open_deep_research/langgraph.json:3).
3. Пользователь отправляет `messages`; graph стартует с `clarify_with_user` [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:714).
4. Если clarification выключен или не нужен, создаётся `research_brief` [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:149).
5. Supervisor делегирует one-or-many research topics через `ConductResearch` [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:201).
6. Researcher agents запускают tools: Tavily, OpenAI/Anthropic native search, MCP или `none` + MCP [utils.py](open_deep_research/src/open_deep_research/utils.py:531).
7. Исследования сжимаются, возвращаются supervisor как tool messages, затем `notes` переходят в final report generation [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:254).
8. Final report model пишет markdown-ответ [prompts.py](open_deep_research/src/open_deep_research/prompts.py:228).

Второй сценарий: evaluation. `tests/run_evaluate.py` компилирует `deep_researcher_builder` с `MemorySaver`, задаёт config для Deep Research Bench и запускает LangSmith `aevaluate` [run_evaluate.py](open_deep_research/tests/run_evaluate.py:32), [run_evaluate.py](open_deep_research/tests/run_evaluate.py:63).

Legacy-сценарии: `src/legacy/graph.py` — plan-and-execute с human interrupt [graph.py](open_deep_research/src/legacy/graph.py:142); `src/legacy/multi_agent.py` — supervisor/research_team через `Send` [multi_agent.py](open_deep_research/src/legacy/multi_agent.py:303).

**6. Конфигурация, зависимости и окружение**

Установка из README:

```bash
uv venv
source .venv/bin/activate
uv sync
cp .env.example .env
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

Зависимости объявлены в [pyproject.toml](open_deep_research/pyproject.toml:11): LangGraph, LangChain providers, Tavily, MCP, Supabase, LangSmith, Google/AWS/Anthropic/OpenAI integrations и др. Dev extras: `mypy`, `ruff` [pyproject.toml](open_deep_research/pyproject.toml:49). Build backend: setuptools [pyproject.toml](open_deep_research/pyproject.toml:52).

Env keys из [.env.example](open_deep_research/.env.example:1): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING`, `SUPABASE_KEY`, `SUPABASE_URL`, `GET_API_KEYS_FROM_CONFIG`.

Важная логика ключей: если `GET_API_KEYS_FROM_CONFIG=true`, ключи берутся из `configurable.apiKeys`; иначе из env [utils.py](open_deep_research/src/open_deep_research/utils.py:892). Это нужно для OAP/production deployments.

Команды из репозитория:

| Команда | Назначение | Источник |
| ---- | ---- | ---- |
| `uv sync` | install dependencies | [README.md](open_deep_research/README.md:31) |
| `uvx ... langgraph dev --allow-blocking` | local LangGraph server | [README.md](open_deep_research/README.md:45) |
| `python tests/run_evaluate.py` | LangSmith Deep Research Bench evaluation | [README.md](open_deep_research/README.md:93) |
| `python tests/extract_langsmith_data.py ...` | export JSONL for benchmark | [README.md](open_deep_research/README.md:100) |
| `ruff check` | linting command documented in repo notes | [CLAUDE.md](open_deep_research/CLAUDE.md:54) |
| `mypy` | type checking command documented in repo notes | [CLAUDE.md](open_deep_research/CLAUDE.md:54) |

Предположение: для полного lint/typecheck в рабочем дереве обычно запускали бы `ruff check .` и `mypy src tests`, но таких script aliases в `pyproject.toml` нет.

**7. Тесты и качество кода**

В текущем `tests/` нет обычных unit tests. Это evaluation harness вокруг LangSmith:

- `run_evaluate.py` запускает graph на dataset `"Deep Research Bench"` и наборе LLM-as-judge evaluators [run_evaluate.py](open_deep_research/tests/run_evaluate.py:13).
- `evaluators.py` проверяет overall quality, relevance, structure, correctness, groundedness, completeness; groundedness сравнивает `final_report` с `raw_notes` [evaluators.py](open_deep_research/tests/evaluators.py:134).
- `supervisor_parallel_evaluation.py` проверяет число tool calls supervisor относительно ожидаемой parallelism-разметки [supervisor_parallel_evaluation.py](open_deep_research/tests/supervisor_parallel_evaluation.py:10).
- `extract_langsmith_data.py` выгружает successful runs в JSONL [extract_langsmith_data.py](open_deep_research/tests/extract_langsmith_data.py:13).

Legacy pytest есть в [src/legacy/tests/test_report_quality.py](open_deep_research/src/legacy/tests/test_report_quality.py:139): он запускает legacy `graph` или `multi_agent` и оценивает отчёт LLM-оценщиком. Это интеграционный тест, требующий API keys и внешних сервисов.

Качество: ruff настроен в `pyproject.toml` [pyproject.toml](open_deep_research/pyproject.toml:67), mypy только в dev dependency [pyproject.toml](open_deep_research/pyproject.toml:49). GitHub Actions не запускают pytest/ruff/mypy: workflows вызывают Claude Code review/assistant [claude-code-review.yml](open_deep_research/.github/workflows/claude-code-review.yml:34), [claude.yml](open_deep_research/.github/workflows/claude.yml:33). Тесты я не запускал: они дорогие/сетевые и требуют LLM/Search/LangSmith ключей.

**8. Как расширять проект**

Добавить новый search provider: расширить `SearchAPI` в [configuration.py](open_deep_research/src/open_deep_research/configuration.py:11), добавить UI option в `Configuration.search_api` [configuration.py](open_deep_research/src/open_deep_research/configuration.py:78), реализовать ветку в `get_search_tool` [utils.py](open_deep_research/src/open_deep_research/utils.py:531), при необходимости добавить API-key routing в `get_api_key_for_model`/отдельный helper [utils.py](open_deep_research/src/open_deep_research/utils.py:892).

Добавить новый graph step: добавить поле state в [state.py](open_deep_research/src/open_deep_research/state.py:65), node-функцию в [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:701), затем `add_node`/`add_edge` в builder [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:708).

Изменить поведение агента: сначала менять prompts в [prompts.py](open_deep_research/src/open_deep_research/prompts.py:79), затем при необходимости tool schemas в [state.py](open_deep_research/src/open_deep_research/state.py:15).

Добавить MCP-интеграцию без кода: передать `mcp_config.url`, `tools`, `auth_required`, `mcp_prompt` через LangGraph config/UI [configuration.py](open_deep_research/src/open_deep_research/configuration.py:213). Код уже фильтрует tools и оборачивает auth errors [utils.py](open_deep_research/src/open_deep_research/utils.py:506).

Добавлять проверки лучше в `tests/`: для regression-поведения можно добавить unit tests вокруг pure helpers, а для agent-quality продолжать LangSmith evaluators.

**9. Архитектурная оценка**

Сильные стороны: компактный центральный граф, явные LangGraph nodes/edges, конфигурация через Pydantic и OAP UI metadata, разделение prompts/state/utils, поддержка parallel sub-researchers, retries и token-limit fallback [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:701), [configuration.py](open_deep_research/src/open_deep_research/configuration.py:45).

Ограничения и риски:

- В `supervisor_tools` есть `if is_token_limit_exceeded(...) or True`, из-за чего любое исключение в delegated research завершает research phase без явной ошибки [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:332). Это хрупкое место.
- README говорит о wide range search tools, но текущий основной `SearchAPI` поддерживает только `anthropic`, `openai`, `tavily`, `none` [configuration.py](open_deep_research/src/open_deep_research/configuration.py:11). Более широкий список живёт в legacy config [configuration.py](open_deep_research/src/legacy/configuration.py:20).
- MCP loading молча возвращает `[]` при ошибке подключения [utils.py](open_deep_research/src/open_deep_research/utils.py:498), что может выглядеть как “нет tools”, а не как integration failure.
- `MODEL_TOKEN_LIMITS` сам помечен как потенциально устаревший [utils.py](open_deep_research/src/open_deep_research/utils.py:787).
- CI не исполняет тесты/линтеры; это снижает защиту от regressions.
- `.github/dependabot.yml` содержит два top-level `updates`, YAML-дубликат ключа может привести к тому, что первая секция будет затёрта второй [dependabot.yml](open_deep_research/.github/dependabot.yml:7).
- `pyproject.toml` объявляет package data `py.typed`, но в `src/open_deep_research` такого файла не видно; это стоит проверить при packaging/type checking [pyproject.toml](open_deep_research/pyproject.toml:64).

**10. Карта чтения кода**

1. Сначала прочитать [README.md](open_deep_research/README.md:21), потому что он даёт запуск, config и evaluation.
2. Затем [langgraph.json](open_deep_research/langgraph.json:3), потому что там настоящий runtime entry point.
3. Затем [deep_researcher.py](open_deep_research/src/open_deep_research/deep_researcher.py:60), потому что это основной граф.
4. Затем [state.py](open_deep_research/src/open_deep_research/state.py:15) и [configuration.py](open_deep_research/src/open_deep_research/configuration.py:38), потому что они задают контракты.
5. Затем [utils.py](open_deep_research/src/open_deep_research/utils.py:43), потому что там integrations.
6. Затем [prompts.py](open_deep_research/src/open_deep_research/prompts.py:79), потому что prompt logic сильно определяет поведение.
7. Затем [tests/run_evaluate.py](open_deep_research/tests/run_evaluate.py:32) и [tests/evaluators.py](open_deep_research/tests/evaluators.py:25), потому что они показывают ожидаемое качество.
8. Затем [src/security/auth.py](open_deep_research/src/security/auth.py:21), если нужен deployment/OAP.
9. В конце [src/legacy/graph.py](open_deep_research/src/legacy/graph.py:487) и [src/legacy/multi_agent.py](open_deep_research/src/legacy/multi_agent.py:474), если нужно понять эволюцию проекта.

Самые важные файлы: `README.md`, `langgraph.json`, `src/open_deep_research/deep_researcher.py`, `configuration.py`, `state.py`, `utils.py`, `prompts.py`, `tests/run_evaluate.py`, `tests/evaluators.py`, `src/security/auth.py`.

**11. Итоговая ментальная модель**

Проект можно понимать как LangGraph-оркестратор исследовательского процесса: из пользовательского вопроса он делает research brief, supervisor решает, какие подзадачи исследовать, researcher agents собирают источники через tools, compression сохраняет факты и ссылки, final writer собирает markdown-отчёт.

Главная логика сосредоточена в `src/open_deep_research/deep_researcher.py`; интеграции в `utils.py`; поведение LLM в `prompts.py`; настройки в `configuration.py`. Для расширения чаще всего нужно менять config enum/UI metadata, tool assembly, prompts и добавить evaluation. Главная архитектурная идея — не “один агент пишет всё”, а staged graph: clarification → planning → delegated research → compression → final synthesis.