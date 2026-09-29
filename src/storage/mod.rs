//! Persistence: SQLite storage engine, serde row models, and the
//! JSON / Markdown / CSV exporters.
//!
//! * [`db`] — `rusqlite` open/migrate (blueprint §8 schema) +
//!   insert/query helpers; the DB lives at
//!   `dirs::data_dir()/crucible/benchmarks.db` (Chunk 12).
//! * [`models`] — serde row structs matching `schema.sql` (Chunk 12).
//! * [`export`] — zero-alloc JSON / GFM Markdown / raw CSV (Chunk 13).

pub mod db;
pub mod export;
pub mod models;

pub use db::{Database, StorageError, DB_FILE_NAME};
pub use models::{BenchmarkSession, NeedleEvaluation, StreamMetricRow};
