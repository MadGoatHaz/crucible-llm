-- Crucible-LLM persistence schema (blueprint §8, "Database Relational Schema").
--
-- Applied idempotently by `Database::open` (the migration step): `IF NOT
-- EXISTS` makes re-running the batch on an existing DB a no-op.

CREATE TABLE IF NOT EXISTS benchmark_sessions (
    session_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    target_url TEXT NOT NULL,
    model_name TEXT NOT NULL,
    backend_type TEXT,       -- 'vllm', 'llamacpp', 'sglang', 'ollama'
    quantization TEXT,
    system_gpu TEXT,
    total_duration_sec REAL
);

CREATE TABLE IF NOT EXISTS stream_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES benchmark_sessions(session_id),
    concurrency_level INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    reasoning_tokens INTEGER,
    ttft_ms REAL,
    tpot_ms REAL,
    mtp_efficiency REAL,
    joules_per_token REAL,
    cache_hit BOOLEAN
);

CREATE TABLE IF NOT EXISTS needle_evaluations (
    eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES benchmark_sessions(session_id),
    context_length INTEGER,
    depth_percent REAL,
    retrieved_successfully BOOLEAN,
    latency_ms REAL
);
