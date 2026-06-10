-- Generated from src/store/db.py (single source of truth)

create extension if not exists vector;

create table if not exists runs (
    id text primary key, question text, brief jsonb, config jsonb,
    started_at timestamptz, cost_usd numeric, status text);

create table if not exists sources (
    id text primary key, domain text unique, tier char(1), tier_basis text[],
    notes text);

create table if not exists snapshots (
    id text primary key, url text, fetched_at timestamptz,
    content_sha char(64), storage_key text, text_len int);

create table if not exists findings (
    id text primary key, run_id text references runs,
    subquestion text, text text, quote text,
    snapshot_id text references snapshots, char_start int, char_end int);

create table if not exists claims (
    id text primary key, run_id text references runs,
    text text, type text, scope jsonb, decision_relevance real,
    status text, sigma jsonb, flags text[],
    derived_from text[], embedding vector(1024));

create table if not exists evidence (
    id text primary key, claim_id text references claims,
    hypothesis_kind text, stance text, stance_score real,
    quote text, snapshot_id text references snapshots,
    char_start int, char_end int, source_id text references sources,
    etype text, published_at date, origin_cluster text);

create table if not exists protocols (
    claim_id text primary key references claims,
    hypotheses jsonb, searches jsonb, coverage real);

create table if not exists traces (
    id bigserial primary key, run_id text, node text, kind text,
    payload jsonb, cost_usd numeric, latency_ms int, ts timestamptz);
