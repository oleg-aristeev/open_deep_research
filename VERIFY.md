# Verified Deep Research — MVP

Реализация MVP из `deep_research_mvp_plan.md`: верификационный контур поверх
форка `langchain-ai/open_deep_research` (ODR). Стандартный deep-research-поток
дополнен фазой verify: **claimify → skeptic → охота за контрсвидетельствами →
stance → агрегированная уверенность → verified-отчёт с линтером атрибуции**.

## Поток

```
clarify → write_brief → supervisor ⇄ [researcher×N*] → compress
                                                        │ enable_verification=true
                                                        ▼
   notetake → claimify → rank → map: verify_one(c)×K → collect
                                                        ▼
                       compose → lint → (repair ≤2) → finalize → END

* tavily_search патчен: сырые выдачи и полные тексты страниц уходят в
  snapshot/trace store ДО лоссивной compression (# PATCH(verify) в utils.py).
  Цитаты в отчёте указывают на снапшоты с char-offsets, а не на пересказ.
```

При `enable_verification=false` граф работает как upstream ODR (baseline).

## Структура

| Путь | Что это |
|---|---|
| `src/verify/schema.py` | Pydantic: Claim, Evidence, Source, Protocol(Φ), Sigma, Budget |
| `src/verify/findings.py` | note-taker: снапшоты → findings с verbatim-цитатами (substring-проверка кодом) |
| `src/verify/claimify.py` | извлечение + деконтекстуализация + типизация клеймов; gate τ_extract |
| `src/verify/skeptic.py` | протокол гипотез v0: NEGATION / OUTDATED / SCOPE / NUMERIC / CIRCULAR |
| `src/verify/hunter.py` | целевой pro/contra retrieval c exclude-origins и снапшотами |
| `src/verify/stance.py` | StanceClassifier: LLM (default) \| MiniCheck на vLLM (интерфейс §4.5) |
| `src/verify/sources.py` | tier-реестр (configs/domains/tiers.yaml) + origin-кластеризация (3 эвристики) |
| `src/verify/aggregate.py` | σ v1: log-odds + tanh-сатурация кластеров + cap при одном origin + флаги |
| `src/verify/compose.py` | composer (факты только из verified-клеймов) + детерминированные рендеры |
| `src/verify/linter.py` | инвариант I1 (`[clm_*]` / `(интерпретация)`) + NLI-проверка дрейфа |
| `src/verify/graph.py` | LangGraph-subgraph фазы verify (Send-параллелизм, repair-петля) |
| `src/store/` | снапшоты (content-addressed, offsets), трейсы всех вызовов, DDL Postgres |
| `src/explore_backend.py` | хедж: Protocol + адаптеры ODR (default) / GPT Researcher |
| `configs/mvp.yaml` | бюджеты, пороги, модели, константы агрегатора |
| `evals/trapset/v0.jsonl` | trap set v0 (15 вопросов-сидов; цель 30→50, нужна ревизия командой) |
| `evals/run_trapset.py` | CER / verification accuracy / citation precision / false-disputed / дисперсия |
| `evals/annotate/` | streamlit-форма разметки + гайдлайн (κ ≥ 0.6) |
| `deploy/docker-compose.yml` | postgres+pgvector, minio, langfuse, (vllm закомментирован) |

Патчи в код ODR минимальны и помечены `# PATCH(verify)`:
`configuration.py` (+`enable_verification`), `utils.py` (tee в `tavily_search`),
`deep_researcher.py` (условное ребро на verify-subgraph), `pyproject.toml`.
Состояние графа не патчилось: сырые результаты идут через run store, ключуемый
`configurable.verify_run_id` (fallback — `thread_id`).

## Запуск

```bash
uv sync
cp .env.example .env   # OPENAI_API_KEY + TAVILY_API_KEY минимум

# LangGraph Studio (verified-отчёт по умолчанию)
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking

# Один вопрос из trap set, end-to-end
uv run python evals/run_trapset.py --ids trap_006 --variants verified

# Сравнение с baseline («голый» ODR)
uv run python evals/run_trapset.py --variants verified,odr_baseline --limit 5

# Оффлайн-тесты verify-слоя (без сети/LLM)
uv run pytest tests/test_verify_offline.py -q
```

Артефакты прогона: `.verify_store/runs/<run_id>/` — `report.md`,
`claim_graph.json`, `findings.json`, `traces.jsonl`, `raw_hits.jsonl`;
снапшоты — `.verify_store/snapshots/`. Результаты evals — `evals/results/`.

## Статус относительно плана (§7)

Сделано (код): схемы, store, note-taker, claimify+gate, skeptic-lite,
hunter, stance-LLM (+MiniCheck-клиент), origin-кластеризация, агрегатор v1 с
флагами, composer+linter+repair, verify-subgraph, патчи ODR, trap set v0 (15),
eval-харнесс, форма разметки, docker-compose, оффлайн-тесты.

Осталось по плану: добить trap set до 30→50 и отревьюить золото; первый
честный прогон + разметка (κ); подбор констант агрегатора и порогов на trap
set; ablation (без skeptic / без кластеризации); MiniCheck на vLLM по триггеру
цены (>40% стоимости на stance); калибровка p_raw — Phase 2 (логирование уже
есть в traces).
