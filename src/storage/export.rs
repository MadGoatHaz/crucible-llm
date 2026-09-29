//! JSON / GFM Markdown / raw CSV exporters (plan Chunk 13, blueprint §8).
//!
//! Three exporters over **stored** (SQLite, [`Database`]) or **live**
//! (in-memory) benchmark data:
//!
//! * **JSON** — compact, single-pass `serde_json` serialization (no
//!   pretty-printer walks, one allocation per field at most); the shape a
//!   CI/CD regression gate parses: `session` + `metrics` + `packets` +
//!   `needles`.
//! * **Markdown** — GitHub-Flavored tables (session metadata, per-stream
//!   §7 metrics, needle evaluations) for instant posting into PRs, issues,
//!   and READMEs.
//! * **CSV** — the raw per-packet dump: every SSE frame's arrival time
//!   (nanoseconds since the stream's `T0`) plus the inter-token interval
//!   to the previous token frame, for external analysis in Python, R, or
//!   Grafana.
//!
//! Measurement-isolation note (blueprint §4): exporting only ever reads
//! finished data — it never touches the quanta timing path.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::client::StreamEvent;
use crate::config::{data_dir, ExportFormat};
use crate::sse::Chunk;

use super::db::{Database, StorageError};
use super::models::{BenchmarkSession, NeedleEvaluation, StreamMetricRow};

/// One raw packet-arrival sample (the CSV export's row source).
///
/// Captured from the stream worker's channel events after a run completes
/// (Chunk 5 `StreamEvent::Frame`): `arrival_ns` is the frame's `t_nanos`
/// (nanoseconds since the stream's `T0`); `itl_ns` is the delta since the
/// previous *token* frame (`None` for the first token frame and for all
/// non-token frames — `usage` / `control` / `done`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PacketSample {
    /// 1-based stream / iteration index.
    pub stream: u64,
    /// Frame arrival time, nanoseconds since the stream's `T0`.
    pub arrival_ns: u64,
    /// Inter-token latency, nanoseconds (`None` where not defined above).
    pub itl_ns: Option<u64>,
    /// `reasoning` | `content` | `usage` | `control` | `done`.
    pub kind: String,
}

/// The data an export is rendered from: one session, its `stream_metrics`
/// rows, plus (optionally) raw per-packet samples and needle evaluations.
///
/// Built either from the SQLite storage layer ([`Self::from_stored`] — the
/// headless path re-reads the run it just persisted) or from live in-memory
/// data ([`Self::from_live`] — the TUI `e` key).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportPayload {
    pub session: BenchmarkSession,
    pub metrics: Vec<StreamMetricRow>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub packets: Vec<PacketSample>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub needles: Vec<NeedleEvaluation>,
}

/// Export failures.
#[derive(Debug, Error)]
pub enum ExportError {
    /// A storage-layer failure (opening the DB, reading the session's rows).
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
    /// The session id was not found in the storage layer.
    #[error("no benchmark session with id {0} in the storage layer")]
    MissingSession(String),
    /// The export file could not be written.
    #[error("export I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl ExportPayload {
    /// Build a payload from the SQLite storage layer (blueprint §8): the
    /// session row, all of its `stream_metrics` rows, and all of its
    /// `needle_evaluations` rows.
    pub fn from_stored(db: &Database, session_id: &str) -> Result<Self, ExportError> {
        let session = db
            .get_session(session_id)?
            .ok_or_else(|| ExportError::MissingSession(session_id.to_string()))?;
        let metrics = db.get_stream_metrics(session_id)?;
        let needles = db.get_needle_evaluations(session_id)?;
        Ok(Self {
            session,
            metrics,
            packets: Vec::new(),
            needles,
        })
    }

    /// Build a payload from live in-memory data (the TUI `e` key; the
    /// headless path uses [`Self::from_stored`] after persisting).
    pub fn from_live(
        session: BenchmarkSession,
        metrics: Vec<StreamMetricRow>,
        packets: Vec<PacketSample>,
        needles: Vec<NeedleEvaluation>,
    ) -> Self {
        Self {
            session,
            metrics,
            packets,
            needles,
        }
    }

    // ── the three exporters (blueprint §8 "Export Formats") ────────────

    /// **JSON** (CI/CD regression gating): compact, valid, parseable.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| {
            // `ExportPayload` is plain serde data — serialization cannot
            // realistically fail; if it ever does, emit an explicit error
            // object rather than panicking the run.
            format!("{{\"error\": \"serialization failed: {e}\"}}")
        })
    }

    /// **GitHub-Flavored Markdown**: session metadata + per-stream metric
    /// tables (+ a needle table when present), for PRs/issues/READMEs.
    /// Cell values are escaped so a `|` in a model name can't break the
    /// table (GFM rule).
    pub fn to_markdown(&self) -> String {
        let s = &self.session;
        let mut out = String::new();
        out.push_str("# Crucible-LLM Benchmark Report\n\n");
        out.push_str(&format!("- **Session:** `{}`\n", md_cell(&s.session_id)));
        out.push_str(&format!(
            "- **Timestamp:** {}\n",
            md_cell(or_dash(&s.timestamp))
        ));
        out.push_str(&format!("- **Target:** {}\n", md_cell(&s.target_url)));
        out.push_str(&format!("- **Model:** {}\n", md_cell(&s.model_name)));
        out.push_str(&format!(
            "- **Backend:** {}\n",
            md_cell(or_dash(&s.backend_type))
        ));
        out.push_str(&format!(
            "- **Quantization:** {}\n",
            md_cell(or_dash(&s.quantization))
        ));
        out.push_str(&format!("- **GPU:** {}\n", md_cell(or_dash(&s.system_gpu))));
        out.push_str(&format!(
            "- **Duration:** {}\n\n",
            md_cell(&match s.total_duration_sec {
                Some(d) => format!("{d:.2} s"),
                None => "--".to_string(),
            })
        ));

        out.push_str("## Stream Metrics\n\n");
        out.push_str(
            "| # | Concurrency | Prompt Tokens | Completion Tokens | Reasoning Tokens | TTFT (ms) | TPOT (ms) | MTP η | J/Token | Cache |\n",
        );
        out.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
        for (i, m) in self.metrics.iter().enumerate() {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                i + 1,
                md_cell(&opt_int(m.concurrency_level)),
                md_cell(&opt_int(m.prompt_tokens)),
                md_cell(&opt_int(m.completion_tokens)),
                md_cell(&opt_int(m.reasoning_tokens)),
                md_cell(&opt_f3(m.ttft_ms)),
                md_cell(&opt_f3(m.tpot_ms)),
                md_cell(&opt_f3(m.mtp_efficiency)),
                md_cell(&opt_f3(m.joules_per_token)),
                md_cell(&opt_bool(m.cache_hit)),
            ));
        }

        if !self.needles.is_empty() {
            out.push_str("\n## Needle Evaluations\n\n");
            out.push_str("| # | Context Length | Depth (%) | Retrieved | Latency (ms) |\n");
            out.push_str("|---:|---:|---:|:-:|---:|\n");
            for (i, n) in self.needles.iter().enumerate() {
                out.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    i + 1,
                    md_cell(&opt_int(n.context_length)),
                    md_cell(&match n.depth_percent {
                        Some(d) => format!("{d:.1}"),
                        None => "--".to_string(),
                    }),
                    md_cell(&opt_bool(n.retrieved_successfully)),
                    md_cell(&opt_f3(n.latency_ms)),
                ));
            }
        }
        out
    }

    /// **Raw CSV** (Python/R/Grafana): one row per captured packet with
    /// its arrival time (ns since `T0`) and inter-token interval (ns).
    /// No packets were captured (e.g. a TUI export) → header only.
    pub fn to_csv(&self) -> String {
        let mut out = String::from("stream,arrival_ns,itl_ns,kind\n");
        for p in &self.packets {
            out.push_str(&format!("{},{}", p.stream, p.arrival_ns));
            out.push(',');
            out.push_str(&p.itl_ns.map_or(String::new(), |v| v.to_string()));
            out.push(',');
            push_csv_field(&mut out, &p.kind);
            out.push('\n');
        }
        out
    }

    /// Render in the requested [`ExportFormat`].
    pub fn render(&self, format: ExportFormat) -> String {
        match format {
            ExportFormat::Json => self.to_json(),
            ExportFormat::Md => self.to_markdown(),
            ExportFormat::Csv => self.to_csv(),
        }
    }
}

/// Per-packet samples from one stream's worker events (Chunk 13 CSV source).
///
/// `stream` is the 1-based stream/iteration index. Arrival times are the
/// worker's `t_nanos` stamps (ns since `T0`); the ITL is the delta between
/// consecutive *token* frames (reasoning or content), so a `usage` frame
/// between two tokens does not break the interval.
pub fn samples_from_events(events: &[StreamEvent], stream: u64) -> Vec<PacketSample> {
    let mut out = Vec::new();
    let mut last_token_ns: Option<u64> = None;
    for event in events {
        let StreamEvent::Frame { frame, .. } = event else {
            continue;
        };
        let is_token = matches!(&frame.chunk, Chunk::Reasoning(_) | Chunk::Content(_));
        let itl = is_token
            .then(|| last_token_ns.map(|prev| (frame.t_nanos as i128 - prev as i128).max(0) as u64))
            .flatten();
        if is_token {
            last_token_ns = Some(frame.t_nanos);
        }
        let kind = if frame.done {
            "done"
        } else {
            match &frame.chunk {
                Chunk::Reasoning(_) => "reasoning",
                Chunk::Content(_) => "content",
                Chunk::Usage(_) => "usage",
                Chunk::Control => "control",
            }
        };
        out.push(PacketSample {
            stream,
            arrival_ns: frame.t_nanos,
            itl_ns: itl,
            kind: kind.to_string(),
        });
    }
    out
}

/// The file extension for an export format.
pub fn extension(format: ExportFormat) -> &'static str {
    match format {
        ExportFormat::Json => "json",
        ExportFormat::Md => "md",
        ExportFormat::Csv => "csv",
    }
}

/// Default export destination for a persisted session:
/// `data_dir()/exports/crucible-<session-id[:8]>.<ext>`.
pub fn default_path(session_id: &str, format: ExportFormat) -> PathBuf {
    data_dir().join("exports").join(format!(
        "crucible-{}.{}",
        short_id(session_id),
        extension(format)
    ))
}

/// Default export destination for a live (TUI) export:
/// `data_dir()/exports/live-<unix-seconds>.<ext>`.
pub fn default_path_live(format: ExportFormat) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    data_dir()
        .join("exports")
        .join(format!("live-{ts}.{}", extension(format)))
}

/// Write the payload in `format` to `path` (creating parent dirs) and
/// return the destination.
pub fn write(
    path: &Path,
    format: ExportFormat,
    payload: &ExportPayload,
) -> Result<PathBuf, ExportError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, payload.render(format))?;
    Ok(path.to_path_buf())
}

fn short_id(id: &str) -> &str {
    &id[..id.len().min(8)]
}

/// `None` → `--` (the dashboard's N/A marker).
fn or_dash(v: &Option<String>) -> &str {
    v.as_deref().unwrap_or("--")
}

/// Escape a value for a GFM table cell: `|` → `\|`, newlines → spaces.
fn md_cell(v: &str) -> String {
    v.replace('\n', " ").replace('|', "\\|")
}

fn opt_int(v: Option<i64>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "--".to_string())
}

fn opt_f3(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "--".to_string())
}

fn opt_bool(v: Option<bool>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "--".to_string())
}

/// Append a CSV field to `out`, quoting it (RFC 4180) when it contains a
/// comma, quote, or newline.
fn push_csv_field(out: &mut String, field: &str) {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        out.push('"');
        out.push_str(field.replace('"', "\"\"").as_str());
        out.push('"');
    } else {
        out.push_str(field);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::{ParsedFrame, Usage};
    use crate::timing::{MonotonicInstant, StreamTimestamps};

    fn sample_session() -> BenchmarkSession {
        BenchmarkSession {
            session_id: "12345678-aaaa-bbbb-cccc-dddddddddddd".to_string(),
            timestamp: Some("2026-09-29 22:00:00".to_string()),
            target_url: "http://127.0.0.1:8000/v1/chat/completions".to_string(),
            model_name: "qwen3-8b".to_string(),
            backend_type: Some("vllm".to_string()),
            quantization: Some("Q4_K_M".to_string()),
            system_gpu: Some("RTX 4090".to_string()),
            total_duration_sec: Some(12.34),
        }
    }

    fn sample_metric(id: i64) -> StreamMetricRow {
        StreamMetricRow {
            metric_id: Some(id),
            session_id: "12345678-aaaa-bbbb-cccc-dddddddddddd".to_string(),
            concurrency_level: Some(1),
            prompt_tokens: Some(128),
            completion_tokens: Some(34),
            reasoning_tokens: None,
            ttft_ms: Some(182.5),
            tpot_ms: Some(13.8),
            mtp_efficiency: Some(1.84),
            joules_per_token: None,
            cache_hit: None,
        }
    }

    fn sample_payload() -> ExportPayload {
        let packets = vec![
            PacketSample {
                stream: 1,
                arrival_ns: 1_000_000,
                itl_ns: None,
                kind: "reasoning".into(),
            },
            PacketSample {
                stream: 1,
                arrival_ns: 1_012_000,
                itl_ns: Some(12_000),
                kind: "content".into(),
            },
            PacketSample {
                stream: 1,
                arrival_ns: 1_030_000,
                itl_ns: Some(18_000),
                kind: "content".into(),
            },
            PacketSample {
                stream: 1,
                arrival_ns: 1_031_000,
                itl_ns: None,
                kind: "usage".into(),
            },
            PacketSample {
                stream: 1,
                arrival_ns: 1_031_500,
                itl_ns: None,
                kind: "done".into(),
            },
        ];
        ExportPayload::from_live(
            sample_session(),
            vec![sample_metric(1), sample_metric(2)],
            packets,
            Vec::new(),
        )
    }

    #[test]
    fn json_export_is_valid_and_parseable() {
        let payload = sample_payload();
        let json = payload.to_json();
        // Acceptance: `--export json` produces valid, parseable JSON.
        let back: ExportPayload = serde_json::from_str(&json).expect("must be valid JSON");
        assert_eq!(back.session.session_id, payload.session.session_id);
        assert_eq!(back.session.model_name, "qwen3-8b");
        assert_eq!(back.metrics.len(), 2);
        assert_eq!(back.metrics[0].ttft_ms, Some(182.5));
        assert_eq!(back.packets.len(), 5);
        assert_eq!(back.packets[1].itl_ns, Some(12_000));
        // Compact (no pretty-print newlines outside the string values).
        assert!(!json.contains("\n"));
    }

    #[test]
    fn markdown_export_has_gfm_tables_with_key_metrics() {
        let payload = sample_payload();
        let md = payload.to_markdown();
        // Acceptance: `--export md` produces a GFM table with the key metrics.
        assert!(md.contains("# Crucible-LLM Benchmark Report"));
        assert!(md.contains("| # | Concurrency | Prompt Tokens | Completion Tokens"));
        assert!(md.contains("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"));
        assert!(md.contains("| 1 | 1 | 128 | 34 | -- | 182.500 | 13.800 | 1.840 | -- | -- |"));
        assert!(md.contains("**Session:** `12345678-aaaa-bbbb-cccc-dddddddddddd`"));
        assert!(md.contains("**Target:** http://127.0.0.1:8000/v1/chat/completions"));
        assert!(md.contains("**Duration:** 12.34 s"));
        // No needle data → no needle section.
        assert!(!md.contains("## Needle Evaluations"));
    }

    #[test]
    fn markdown_needle_section_appears_only_with_data() {
        let mut payload = sample_payload();
        payload.needles = vec![NeedleEvaluation {
            eval_id: Some(1),
            session_id: payload.session.session_id.clone(),
            context_length: Some(8192),
            depth_percent: Some(50.0),
            retrieved_successfully: Some(true),
            latency_ms: Some(123.4),
        }];
        let md = payload.to_markdown();
        assert!(md.contains("## Needle Evaluations"));
        assert!(md.contains("| # | Context Length | Depth (%) | Retrieved | Latency (ms) |"));
        assert!(md.contains("| 1 | 8192 | 50.0 | true | 123.400 |"));
    }

    #[test]
    fn markdown_escapes_pipes_in_cells() {
        let mut payload = sample_payload();
        payload.session.model_name = "weird|model".to_string();
        let md = payload.to_markdown();
        assert!(md.contains("**Model:** weird\\|model"));
    }

    #[test]
    fn csv_export_includes_packet_timestamps_and_itl() {
        let payload = sample_payload();
        let csv = payload.to_csv();
        // Acceptance: `--export csv` includes per-packet timestamps and ITL values.
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "stream,arrival_ns,itl_ns,kind");
        assert_eq!(lines.len(), 6); // header + 5 packets
        assert_eq!(lines[1], "1,1000000,,reasoning"); // first token frame: no ITL
        assert_eq!(lines[2], "1,1012000,12000,content");
        assert_eq!(lines[3], "1,1030000,18000,content");
        assert_eq!(lines[4], "1,1031000,,usage"); // non-token frame: no ITL
        assert_eq!(lines[5], "1,1031500,,done");
    }

    #[test]
    fn csv_with_no_packets_is_header_only() {
        let payload = ExportPayload::from_live(
            sample_session(),
            vec![sample_metric(1)],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(payload.to_csv(), "stream,arrival_ns,itl_ns,kind\n");
    }

    fn frame_events() -> (StreamTimestamps, Vec<StreamEvent>) {
        let t0 = MonotonicInstant::now();
        let ts = StreamTimestamps {
            t0: Some(t0),
            ..StreamTimestamps::default()
        };
        // t_nanos are fixed synthetic values (ns since T0): the worker
        // stamps them; the parser leaves them 0 by contract.
        let mk = |chunk: Chunk, t_nanos: u64, done: bool| StreamEvent::Frame {
            frame: ParsedFrame {
                chunk,
                t_nanos,
                done,
            },
            at: t0,
            timestamps: ts,
        };
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 4,
        };
        let events = vec![
            mk(Chunk::Reasoning("think".into()), 100, false),
            mk(Chunk::Content("a".into()), 130, false),
            mk(Chunk::Usage(usage), 135, false),
            mk(Chunk::Content("b".into()), 150, false),
            mk(Chunk::Control, 160, true),
        ];
        (ts, events)
    }

    #[test]
    fn samples_from_events_computes_arrivals_and_itl() {
        let (_ts, events) = frame_events();
        let samples = samples_from_events(&events, 3);
        assert_eq!(samples.len(), 5);
        assert_eq!(
            samples[0],
            PacketSample {
                stream: 3,
                arrival_ns: 100,
                itl_ns: None,
                kind: "reasoning".into(),
            }
        );
        // ITL spans token frames only: 130−100, then 150−130 (the usage
        // frame at 135 does not reset the last-token anchor).
        assert_eq!(samples[1].itl_ns, Some(30));
        assert_eq!(samples[2].itl_ns, None);
        assert_eq!(samples[3].itl_ns, Some(20));
        assert_eq!(samples[3].kind, "content");
        assert_eq!(samples[4].kind, "done");
        assert_eq!(samples[4].itl_ns, None);
        // Arrivals are monotonic.
        for w in samples.windows(2) {
            assert!(w[1].arrival_ns >= w[0].arrival_ns);
        }
    }

    #[test]
    fn samples_from_events_skips_non_frame_events() {
        let (ts, mut events) = frame_events();
        events.push(StreamEvent::Complete {
            timestamps: ts,
            usage: None,
            premature: false,
            malformed_frames: 0,
        });
        assert_eq!(samples_from_events(&events, 1).len(), 5);
    }

    #[test]
    fn from_stored_reads_session_metrics_and_needles() {
        let dir = std::env::temp_dir().join(format!("crucible-export-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.join("benchmarks.db");
        let mut db = Database::open(&db_path).unwrap();
        let session = sample_session();
        db.persist_run(&session, &[sample_metric(1), sample_metric(2)])
            .unwrap();
        db.insert_needle_evaluation(&NeedleEvaluation {
            eval_id: None,
            session_id: session.session_id.clone(),
            context_length: Some(4096),
            depth_percent: Some(10.0),
            retrieved_successfully: Some(false),
            latency_ms: Some(99.0),
        })
        .unwrap();

        let payload = ExportPayload::from_stored(&db, &session.session_id).unwrap();
        assert_eq!(payload.session.session_id, session.session_id);
        assert_eq!(payload.metrics.len(), 2);
        assert_eq!(payload.needles.len(), 1);
        assert!(payload.needles[0].retrieved_successfully == Some(false));
        assert!(payload.packets.is_empty());

        // A missing session is a typed error, not a panic.
        let e = ExportPayload::from_stored(&db, "nope").unwrap_err();
        assert!(matches!(e, ExportError::MissingSession(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_creates_parent_dirs_and_round_trips() {
        let dir = std::env::temp_dir().join(format!("crucible-export-w-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let payload = sample_payload();

        for format in [ExportFormat::Json, ExportFormat::Md, ExportFormat::Csv] {
            let path = write(
                &dir.join("nested")
                    .join(format!("out.{}", extension(format))),
                format,
                &payload,
            )
            .unwrap();
            let on_disk = std::fs::read_to_string(&path).unwrap();
            assert_eq!(on_disk, payload.render(format));
        }
        // The JSON on disk parses.
        let json_path = dir.join("nested").join("out.json");
        let back: ExportPayload =
            serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
        assert_eq!(back.session.model_name, "qwen3-8b");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_paths_live_under_data_dir_exports() {
        let p = default_path("12345678-aaaa-bbbb-cccc-dddddddddddd", ExportFormat::Csv);
        assert!(p.starts_with(data_dir()));
        assert!(p.to_string_lossy().ends_with("crucible-12345678.csv"));
        let pl = default_path_live(ExportFormat::Md);
        assert!(pl.starts_with(data_dir()));
        assert!(pl.to_string_lossy().contains("live-"));
        assert!(pl.to_string_lossy().ends_with(".md"));
    }

    #[test]
    fn render_dispatches_all_three_formats() {
        let payload = sample_payload();
        assert!(payload.render(ExportFormat::Json).starts_with('{'));
        assert!(payload.render(ExportFormat::Md).starts_with('#'));
        assert!(payload.render(ExportFormat::Csv).starts_with("stream,"));
    }
}
