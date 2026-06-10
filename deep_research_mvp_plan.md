# MVP: Verified Deep Research поверх форка open_deep_research
## Build Plan — v0.1 (детализация §8 основного design doc)

---

## 0. Цель MVP и проверяемая гипотеза

**Гипотеза продукта:** добавление верификационного контура (claimify → skeptic → охота за контрсвидетельствами → stance → агрегированная уверенность) к стандартному deep-research-пайплайну даёт *измеримо* более надёжные отчёты — конкретно: Counter-Evidence Recall ≥ 0.6 и citation precision ≥ 0.85 на trap set при стоимости ≤ $5/отчёт, и эксперт в слепом сравнении предпочитает наш отчёт обычному deep research для принятия решения.

**MVP должен уметь:** принять вопрос → исследовать веб → выдать Markdown-отчёт, где топ-10–15 утверждений имеют вердикт, оценку уверенности, источники «за», результаты поиска «против» и флаги (`disputed / stale / single_origin / not_enough_evidence`) + machine-readable `claim_graph.json` + полный протокол Φ «как искали».

**Не-цели MVP (фиксируем письменно, чтобы не расползтись):** UI, своя база знаний с TTL, debate-эскалация, subjective logic, обучаемая репутация источников, мультиязычный retrieval, re-verification, calibration-сервис (логируем p_raw, калибруем после), production-инфраструктура (k8s, rate limiting).

**Команда и срок:** 2 инженера (далее A — инфраструктура/explore, B — epistemic layer), 6 недель.

---

## 1. Выбор базы для форка

### 1.1. Кандидаты

| Критерий | **langchain-ai/open_deep_research** | **assafelovic/gpt-researcher** | smolagents open-deep-research | STORM/Co-STORM |
|---|---|---|---|---|
| Лицензия | MIT | Apache 2.0 | Apache 2.0 | MIT |
| Оркестрация | **LangGraph state graph** (supervisor → параллельные researcher-субагенты → compress → final report) | собственная абстракция (ResearchConductor, skill-менеджеры) + отдельный multi-agent режим | скрипт-уровень, GAIA-ориентирован | свой пайплайн, заточен на wiki-статьи |
| Встраивание новой фазы | **естественно**: verify = ещё один subgraph между compress и report | придётся резать их конвейер или вешаться на выход | нужно дописывать всё | чужая цель |
| Checkpointing / HITL | нативно в LangGraph (persistence, interrupts) | нет из коробки | нет | нет |
| Structured outputs | везде (Pydantic-схемы) | частично | минимально | частично |
| Конфигурация моделей по ролям | да (summarization / research / compression / report — отдельные модели) | да (three-tier) | нет | частично |
| Поиск/скрейпинг из коробки | Tavily/Exa/ArXiv/MCP и др. | **самый широкий набор** retrievers + scrapers, deep/detailed режимы, фронтенд, MCP-сервер | базовый | свой |
| Evals из коробки | **скрипт прогона на Deep Research Bench** (формат JSONL под их лидерборд) | нет | GAIA | свой |
| Зрелость/активность | активен, но молодой; нет Docker/CI-прода | очень зрелый, активные релизы и в 2026 | пример-код | исследовательский |

### 1.2. Решение

**Форкаем `langchain-ai/open_deep_research` (далее ODR) как оркестрационный скелет.** Решающие аргументы: (1) verify-фаза вставляется как LangGraph-subgraph без вскрытия чужих абстракций; (2) персистентность и interrupts LangGraph бесплатно дают checkpointing и будущие HITL-точки; (3) Pydantic-структурированные выходы по всему пайплайну — это ровно та дисциплина, которая нужна для claims/evidence; (4) готовый прогон на Deep Research Bench закрывает часть report-level evals; (5) MIT.

**GPT Researcher используем двумя способами, не форкая:** (а) как **донор компонентов** — его retrievers/scrapers подключаем как библиотеку (`pip install gpt-researcher`) там, где у ODR не хватает покрытия; (б) как **baseline в evals** — «обычный deep research без верификации» в слепых сравнениях.

**Хедж:** explore-фазу прячем за интерфейс `ExploreBackend` (protocol: `q → list[Finding]`). Если supervisor ODR окажется нестабильным/дорогим, меняем бэкенд на GPT Researcher за ~2 дня, не трогая verify-контур. Это страховка стоимостью один абстрактный класс.

Отвергнутые: smolagents-вариант — слишком тонкий, всё равно пишем сами; STORM — ценен идеями (multi-perspective questions заимствуем в промпт skeptic), но его пайплайн заточен под энциклопедические статьи.

### 1.3. Гигиена форка

- Форк с закреплением на конкретном коммите (тег `odr-base`), upstream-remote сохраняем; **наш код живёт в отдельных пакетах** `src/verify/`, `src/store/`, `src/evals/` — патчи в файлы ODR минимизируем и помечаем `# PATCH(verify):`, чтобы переживать upstream-merge.
- Точки, которые придётся патчить в ODR (полный список — §3.3): схема state, обвязка инструментов researcher'а (перехват сырых результатов поиска до compression), узел финального отчёта.

---

## 2. Целевая структура репозитория

```
verified-deep-research/                  # форк open_deep_research
├── src/
│   ├── open_deep_research/              # код ODR (минимальные патчи, помечены PATCH(verify))
│   │   ├── deep_researcher.py           #   + расширение state, перехват raw results
│   │   └── ...
│   ├── verify/                          # ───── НАШ КОД ─────
│   │   ├── schema.py                    # Pydantic: Claim, Evidence, Source, Protocol, Sigma
│   │   ├── claimify.py                  # извлечение + деконтекстуализация + типизация
│   │   ├── skeptic.py                   # протокол гипотез (5 шаблонов v0)
│   │   ├── hunter.py                    # целевой pro/contra retrieval
│   │   ├── stance.py                    # StanceClassifier: LLM | MiniCheck (интерфейс)
│   │   ├── sources.py                   # tier-список, origin-кластеризация (эвристики)
│   │   ├── aggregate.py                 # σ: формула v1 + флаги
│   │   ├── graph.py                     # LangGraph subgraph фазы verify
│   │   ├── compose.py                   # отчёт: MD + claim_graph.json
│   │   └── linter.py                    # инвариант I1 + дрейф смысла (NLI)
│   ├── store/
│   │   ├── db.py                        # Postgres (схема §5), pgvector
│   │   ├── snapshots.py                 # S3/minio: html/text снапшоты, sha, locator-API
│   │   └── traces.py                    # все LLM/поисковые вызовы (+ Langfuse)
│   └── explore_backend.py               # Protocol + адаптеры: ODR (default), GPTResearcher
├── evals/
│   ├── trapset/v0.jsonl                 # 30→50 вопросов с золотом (§6)
│   ├── run_trapset.py                   # прогон + расчёт CER/precision/accuracy/дисперсии
│   ├── annotate/                        # streamlit-форма ручной разметки + гайдлайн
│   └── baselines.py                     # GPT Researcher и «голый» ODR на тех же вопросах
├── deploy/docker-compose.yml            # postgres+pgvector, minio, langfuse, (vllm)
└── configs/{mvp.yaml, domains/*.yaml}   # бюджеты, tier-список, τ-пороги
```

---

## 3. Встраивание verify-фазы в граф ODR

### 3.1. Поток ODR до и после форка

```
ODR (upstream):   clarify → write_brief → supervisor ⇄ [researcher×N] → compress → final_report

Наш форк:         clarify → write_brief → supervisor ⇄ [researcher×N*] → compress
                        └────────────────────────────────────┐
                                                              ▼
                  claimify → rank_claims → map: verify_claim(c) ×K → aggregate_flags
                                                              ▼
                                    compose_verified_report → lint → (repair ≤2) → END
* researcher патчится: сырые результаты поиска/страницы уходят в trace+snapshot store ДО compression
```

Ключевой патч (без него MVP не имеет смысла): **compression в ODR — лоссивная операция**, после неё теряются точные цитаты и URL-привязки. Мы перехватываем сырые выдачи и тексты страниц в evidence-хранилище со снапшотами и offsets, а compressed-заметки используем только для синтеза и как вход claimify. Цитата в отчёте всегда указывает на снапшот, не на пересказ.

### 3.2. Verify-subgraph (LangGraph)

```python
# src/verify/graph.py — структура узлов (упрощено)
class VerifyState(TypedDict):
    brief: Brief
    findings: list[Finding]           # с locator'ами на снапшоты
    claims: list[Claim]
    budget: Budget

def claimify_node(s):       s["claims"] = claimify(s["findings"], s["brief"]); return s
def rank_node(s):           s["claims"] = rank_by_decision_relevance(s["claims"], s["brief"])[:s["budget"].max_claims]; return s

# верификация одного клейма — отдельный подграф, запускается map'ом (Send) параллельно ×K
def verify_claim_subgraph(c: Claim, budget) -> Claim:
    protocol = skeptic.build(c)                       # 5 гипотез v0 → запросы
    for hyp in protocol.hypotheses:
        ev = hunter.gather(hyp, k=budget.per_hyp,     # поиск + fetch полных страниц + снапшот
                           exclude=c.origin_domains)
        for e in ev:
            e.stance, e.score = stance.classify(c, e) # MiniCheck/LLM, см. §4.5
        c.attach(ev, hyp)
    c.evidence = sources.cluster_origins(c.evidence)  # эвристики §4.6
    c.sigma, c.flags = aggregate(c)                   # §4.7
    c.protocol = protocol.log()                       # Φ — в trace store
    return c

graph.add_node("claimify", claimify_node)
graph.add_node("rank", rank_node)
graph.add_conditional_edges("rank", fan_out_claims)   # Send("verify_one", c) для каждого клейма
graph.add_node("verify_one", verify_one_node)
graph.add_node("collect", collect_node)
graph.add_node("compose", compose_node)               # MD + JSON, предложения с [clm_xxxx]
graph.add_node("lint", lint_node)
graph.add_conditional_edges("lint", lambda s: "repair" if s["violations"] and s["repairs"] < 2 else END)
```

Параллелизм: клеймы верифицируются конкурентно (Send API), внутри клейма гипотезы — последовательно (дёшево и позволяет раннюю остановку: если ¬A сразу подтвердился качественным источником, режем бюджет остальных гипотез вдвое).

### 3.3. Полный список патчей в код ODR

1. `state`: + `raw_results: list[RawHit]`, `snapshot_refs` (расширение TypedDict, аддитивно).
2. Обвязка search-инструмента researcher'а: tee сырого ответа в `store.traces` + постановка URL в очередь снапшотера.
3. `final_report`-узел отключаем, вместо него вход в verify-subgraph (одно ребро).
4. Конфиг: + блок `verify:` (бюджеты, пороги, tier-файл) в их конфиг-схему.
5. Прочее не трогаем — clarify/brief/supervisor/researcher используем как есть.

Оценка объёма патчей: ≤ 300 строк диффа в чужих файлах. Если выходит сильно больше — сигнал, что мы делаем что-то не так (или пора на `ExploreBackend`-хедж).

---

## 4. Спецификации новых компонентов

### 4.1. Snapshot & trace store (фундамент, делается первым)

Каждая затронутая страница: `(url, fetched_at, content_sha, чистый текст, raw html в S3/minio)`. Каждый LLM/поисковый вызов: `(run_id, node, prompt_ref, params, result, cost, latency)` → Postgres + зеркало в Langfuse. Без этого невозможны ни evals, ни цитаты с offsets, ни разбор инцидентов. Объём: ~2 дня (A).

### 4.2. Findings с локаторами

`Finding = {text, verbatim_quote, snapshot_id, char_start, char_end, url, published_at?, subquestion_id}`. Note-taker — один вызов средней модели на страницу со structured output; правило: `verbatim_quote` обязан строго подстрочно находиться в снапшоте (проверяется кодом, при провале — повторный вызов). Объём: 2–3 дня (A).

### 4.3. Claimify

Один вызов frontier-модели на пачку findings: извлечь атомарные клеймы, деконтекстуализировать (разрешить кореференции, добавить субъект/время/единицы/методику из контекста), типизировать (`factual|statistical|causal|trend|predictive|normative|definitional`), привязать `derived_from`. Промпт собирается по рецептам Claimify/FActScore; обязательные негативные инструкции: не атомизировать до потери смысла, мнения → `normative`, при неоднозначности — поле `ambiguity_note`, а не угадывание.

Контроль качества встроенный: для каждого клейма MiniCheck(claim ↔ исходный finding) ≥ τ_extract, иначе клейм помечается `extraction_unstable` и не идёт в верификацию (идёт в отчёт как «непроверенное наблюдение»). Объём: 4–5 дней (B), включая 30 золотых примеров-тестов.

### 4.4. Skeptic-lite (5 шаблонов v0)

Вход — клейм, выход — `Protocol{hypotheses:[{kind, statement, queries[2..3], expected_signal}]}` одним LLM-вызовом. Шаблоны v0: **(1)** прямое опровержение ¬A; **(2)** устаревание (запросы с датным фильтром позже самого свежего pro-свидетельства); **(3)** условия/scope («A только при B?»); **(4)** числовая ловушка — только для `statistical` (альтернативные оценки, база сравнения, единицы); **(5)** circular reporting (запрос самого раннего упоминания, паттерны "according to"). Скелет промпта:

```text
Ты — скептик-аналитик (методология ACH). Дан проверяемый клейм: {claim_json}.
Сгенерируй проверочные гипотезы строго по типам: NEGATION, OUTDATED, SCOPE,
{NUMERIC если type=statistical}, CIRCULAR. Для каждой: формулировка гипотезы,
2-3 поисковых запроса НА НАХОЖДЕНИЕ свидетельств этой гипотезы (не подтверждения
клейма), и какой результат считать сигналом. Запросы должны быть такими, какие
задал бы критик клейма, а не его сторонник. Ответ — JSON по схеме Protocol.
```

Объём: 3 дня (B), включая фикс-сьют из 15 клеймов с ожидаемыми гипотезами.

### 4.5. StanceClassifier (интерфейс с двумя реализациями)

`classify(claim, evidence_text) → (supports|refutes|qualifies|mentions, score)`. **Реализация 1 (недели 3–4):** дешёвая LLM с рубрикой и structured output — быстро запускается, дорого масштабируется. **Реализация 2 (неделя 4–5, по триггеру цены):** Bespoke-MiniCheck-7B на vLLM (одна A10/L4, ~$1/ч аренды) для бинарного grounding + LLM только для различения refutes/qualifies и пограничных скоров (|score−0.5|<0.15). Решение о включении №2 — по факту: если stance-затраты > 40% стоимости отчёта. Объём: 3 дня (B) + 2 дня деплой vLLM (A).

### 4.6. Sources v0: tier + origin-кластеризация

Tier — статический YAML ~200 доменов по нашим трём доменам (A: регуляторы/отчётность/Cochrane/первичные данные; B: peer-review, качественные СМИ; C: обычные СМИ/экспертные блоги; D: UGC/неизвестное; источник не в списке → D + лог для пополнения). Origin-кластеризация — три эвристики по убыванию приоритета: (1) extraction "according to X / sources told Y / пресс-релиз" из текста; (2) near-dup текста (MinHash/эмбеддинги, cos > 0.92); (3) одинаковая дата+сущности+числа. Объём: 3–4 дня (A).

### 4.7. Aggregate v1 (реальный код, не псевдо)

```python
TIER_W = {"A": 1.0, "B": 0.75, "C": 0.45, "D": 0.15}
TYPE_W = {"primary": 1.0, "secondary": 0.7, "opinion": 0.35, "dataset": 1.0}

def sigma(claim, tau_days) -> Sigma:
    clusters = group_by(claim.evidence, "origin_cluster")
    contribs = []
    for ev_list in clusters.values():
        x = sum(e.signed_score * TIER_W[e.tier] * TYPE_W[e.etype]
                * exp(-e.age_days / tau_days) for e in ev_list)
        contribs.append(tanh(x / LAMBDA))                 # сатурация внутри кластера
    L = PRIOR[claim.type] + BETA * sum(contribs)
    p = sigmoid(L)
    if len(clusters) == 1: p = min(p, 0.90)               # cap при одном первоисточнике
    pos = [c for c in contribs if c > 0.15]; neg = [c for c in contribs if c < -0.15]
    agreement = "low" if (pos and neg) else ("high" if len(pos) + len(neg) >= 3 else "medium")
    u = ignorance(n_clusters=len(clusters),               # «мало искали» ≠ «спорно»
                  hyp_covered=claim.protocol.coverage)    # доля отработанных гипотез
    ci = bootstrap_ci(claim.evidence, sigma_fn=...)       # перевыборка evidence, 200 итер.
    return Sigma(p_true=p, ci=ci, agreement=agreement, ignorance_u=u,
                 grade=grade_from(clusters), verbal=icd203(p))
```

Флаги (пороговые правила): `disputed` ⇔ есть neg-кластер tier≥B при pos-кластерах; `single_origin` ⇔ 1 кластер; `stale` ⇔ max(age pro-свидетельств) > τ_domain и гипотеза OUTDATED ничего свежего не нашла; `not_enough_evidence` ⇔ u > 0.5. Константы (LAMBDA, BETA, PRIOR, τ) — в `configs/mvp.yaml`, подбираются на trap set в неделю 6. Объём: 3 дня (B).

### 4.8. Composer + linter

Composer: frontier-модель пишет отчёт **только из verified-клеймов** (текст клеймов + sigma + флаги подаются как единственный источник фактов; findings в промпт не подаются — это главная защита от утечки непроверенного). Каждое утверждающее предложение заканчивается `[clm_xxxx]`; интерпретации — маркер `(интерпретация)`. Рендер: MD-отчёт (синтез + таблица клеймов + раздел «Спорное» + раздел «Чего мы не знаем» + Φ-сводка) и `claim_graph.json`.

Linter: (1) каждое предложение без маркера и без `[clm_]` — нарушение; (2) каждый `[clm_]` существует и `status != unverified`; (3) MiniCheck(предложение ↔ текст клейма) ≥ τ_drift — ловим дрейф смысла при синтезе. Нарушения → repair-вызов composer'а (≤2 итераций), остаток — в видимый блок «не прошло линт» (честность важнее красоты). Объём: 4 дня (A: рендер/линт-механика, B: промпты).

---

## 5. Модель данных (Postgres, DDL-скелет)

```sql
create table runs      (id text primary key, question text, brief jsonb, config jsonb,
                        started_at timestamptz, cost_usd numeric, status text);
create table sources   (id text primary key, domain text unique, tier char(1), tier_basis text[],
                        notes text);
create table snapshots (id text primary key, url text, fetched_at timestamptz,
                        content_sha char(64), storage_key text, text_len int);
create table findings  (id text primary key, run_id text references runs,
                        subquestion text, text text, quote text,
                        snapshot_id text references snapshots, char_start int, char_end int);
create table claims    (id text primary key, run_id text references runs,
                        text text, type text, scope jsonb, decision_relevance real,
                        status text, sigma jsonb, flags text[],
                        derived_from text[], embedding vector(1024));
create table evidence  (id text primary key, claim_id text references claims,
                        hypothesis_kind text, stance text, stance_score real,
                        quote text, snapshot_id text references snapshots,
                        char_start int, char_end int, source_id text references sources,
                        etype text, published_at date, origin_cluster text);
create table protocols (claim_id text primary key references claims,
                        hypotheses jsonb, searches jsonb, coverage real);
create table traces    (id bigserial primary key, run_id text, node text, kind text,
                        payload jsonb, cost_usd numeric, latency_ms int, ts timestamptz);
```

---

## 6. Eval-харнесс (строится в неделю 1, раньше пайплайна)

### 6.1. Trap set: формат и состав

`evals/trapset/v0.jsonl`, 30 вопросов в неделю 1 → 50 к неделе 5. Состав по основному doc §8.4 (устаревшие / мифы / спорные / числовые ловушки / контрольные бесспорные), по трём доменам (AI-рынок, нутрициология, финотчётность).

```jsonc
{ "id": "trap_017", "domain": "nutrition", "category": "contested",
  "question": "Снижает ли умеренное потребление кофе сердечно-сосудистые риски?",
  "gold": {
    "report_verdict": "disputed_with_lean_support",
    "key_claims": [
      { "text_pattern": "умеренное потребление (3-5 чашек) ассоциировано со снижением ССЗ-рисков",
        "verdict": "supported", 
        "must_find_counter": ["обсервационный дизайн / healthy user bias",
                               "не относится к людям с гипертонией/аритмией"] }
    ],
    "must_not": ["категоричное 'кофе полезен всем' без qualifier'ов"],
    "freshest_anchor": null },
  "author": "B", "reviewed_by": "A", "created": "2026-06-15" }
```

### 6.2. Метрики, реализуемые в `run_trapset.py`

CER = |найденные must_find_counter| / |все must_find_counter| (матчинг — NLI claim↔counter + ручная сверка спорных); verification accuracy по key_claims; citation precision — MiniCheck по всем (предложение, цитата)-парам + ручная подвыборка 10%; false-disputed rate на контрольной категории; дисперсия вердиктов по 3 прогонам; cost/latency. Отчёт прогона — одна HTML-страница со сравнением: наш / «голый» ODR / GPT Researcher.

### 6.3. Разметка

Streamlit-форма: разметчику показывается клейм + цитата + подсвеченный снапшот; вопросы — «поддерживает ли цитата клейм?», «вердикт верен?», «scope верен?». Гайдлайн на 2 страницы; первые 30 клеймов размечают оба инженера независимо → κ; κ < 0.6 ⇒ чиним гайдлайн до продолжения.

---

## 7. План по неделям (2 инженера)

| Нед. | A (инфра / explore) | B (epistemic) | Выход недели / чек |
|---|---|---|---|
| 1 | Форк ODR, запуск as-is; docker-compose (pg, minio, langfuse); trace store | Trap set v0 (30 во­просов); гайдлайн разметки; спайк E2: MiniCheck vs LLM-stance на 50 парах | «Голый» ODR прогнан на trap set → **baseline-цифры**; выбор stance-реализации №1 |
| 2 | Snapshot store; патч researcher'а (перехват raw); findings с offsets; DDL | Claimify + деконтекстуализация + 30 золотых тестов; схемы Pydantic | Из прогона ODR извлекаются клеймы с валидными локаторами (демо на 3 вопросах) |
| 3 | Evidence hunter (поиск+fetch+снапшот, exclude-origins); бюджеты в конфиге | Skeptic-lite (5 шаблонов) + фикс-сьют; StanceClassifier-LLM | Один клейм проходит полный verify-цикл руками (notebook) |
| 4 | Verify-subgraph в LangGraph (Send-параллелизм, retries, ранняя остановка); vLLM+MiniCheck если триггер цены | Aggregate v1 + флаги + юнит-тесты на синтетических evidence-наборах; origin-эвристики (с A) | End-to-end прогон одного вопроса: отчёт с таблицей клеймов |
| 5 | Composer-рендер + linter + repair-петля; trap set добивается до 50 | Прогон trap set №1; разметка; разбор топ-5 классов ошибок; фиксы промптов | Первые честные CER / precision / accuracy |
| 6 | Ablation-прогоны (без skeptic / без кластеризации); подбор констант агрегатора; стабилизация | Прогон №2; слепое сравнение с коммерческим DR (5 экспертов × 10 вопросов); MVP-отчёт | **Gate-ревью по критериям §9** |

Правило недели: в любую пятницу система end-to-end запускаема на одном вопросе (пусть с заглушками) — интеграция непрерывная, не «большой взрыв» в неделе 5.

---

## 8. Конфигурация, стек, стоимость

- **Бюджеты по умолчанию** (`configs/mvp.yaml`): explore — 5 подвопросов × 2 поиска × top-5 страниц; verify — 12 клеймов × 5 гипотез × (2 запроса, top-3 страницы, ≤10 stance-проверок); жёсткий потолок $6 и 20 мин на прогон, при достижении — graceful degradation (меньше клеймов, не падение).
- **Модели**: planner/skeptic/claimify/composer — frontier; note-taker/summarization — средняя; stance — по §4.5. Всё через `init_chat_model` ODR — провайдеры свопаются конфигом.
- **Поиск**: Tavily (search + extract) как единственный провайдер MVP; trafilatura — fallback-экстракция; Playwright — вне scope.
- **Грубая оценка стоимости прогона**: explore ≈ $0.6–1.2; claimify+skeptic ≈ $0.4; verify-поиск ≈ $0.3 (≈120 Tavily-кредитов); stance: $0.8–2.0 на LLM (→ $0.1–0.3 на MiniCheck); composer+lint ≈ $0.4. Итого **$2.5–4.5** — в пределах критерия ≤ $5; главный рычаг при превышении — MiniCheck.

---

## 9. Definition of Done и демо

**DoD (gate в Phase 2, из основного doc §8.5):** citation precision ≥ 0.85 (ручная подвыборка ≥ 0.9); CER ≥ 0.6; verification accuracy ≥ 0.75; false-disputed ≤ 1/10 на контрольных; ablation подтверждает вклад skeptic (≥ +20 п.п. CER); стоимость ≤ $5, p95 ≤ 20 мин; 3 прогона одного вопроса расходятся по вердиктам ≤ 10% клеймов; все цифры воспроизводятся командой `python evals/run_trapset.py --config mvp`.

**Демо-сценарий (15 мин):** (1) вопрос из спорного домена → отчёт: вердикты, уверенности, раздел «Спорное» с двумя позициями, «Чего мы не знаем»; (2) клик-путь по JSON: предложение → клейм → контрсвидетельство → подсвеченная цитата в снапшоте → Φ «как искали опровержение»; (3) рядом — отчёт GPT Researcher на тот же вопрос: показать конкретный пропущенный им контраргумент; (4) дашборд trap set: наши метрики против двух baseline'ов.

**Если gate не пройден:** по диагнозу. CER низкий из-за поиска → +1 итерация на hunter (переформулировки запросов); из-за stance → форсировать MiniCheck-дообучение; precision низкий → ужесточить линтер и composer-промпт; всё дорого → резать explore, не verify. Решение «продолжать/пивот в verifier-as-a-service» — по итогам слепого сравнения недели 6.

---

## 10. Чеклист известных срезов (вернуться в Phase 2)

☐ Калибровка p_raw (логируем с недели 4, изотония — Phase 2) ☐ debate-эскалация спорных ☐ subjective logic вместо эвристического u ☐ генеалогия источников сверх трёх эвристик ☐ TTL/re-verification кэша клеймов ☐ предохранитель от prompt injection дальше базового (extraction-модель без tool-доступа уже в MVP — единственное security-исключение из срезов, делаем сразу) ☐ ru-retrieval ☐ UI поверх claim_graph.json.

