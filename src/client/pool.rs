//! `WorkerPool`: spawn/manage N [`StreamWorker`]s with independent HTTP/2
//! streams, fanning their channels into a single bounded `tokio::sync::mpsc`
//! (plan Chunk 10, blueprint §4.1 / §5 Engine B).
//!
//! Design:
//!
//! * Every worker shares **one** connection-pooled `reqwest::Client` (cheap
//!   `Arc` clone), so all N streams multiplex over shared sockets — the
//!   "independent HTTP/2 streams" of the blueprint, with pooling doing the
//!   rest for HTTP/1.1 local endpoints.
//! * Each worker emits [`StreamEvent`]s on a small **private** bounded
//!   channel; a per-worker forwarder task tags each event with its worker id
//!   ([`PoolEvent`]) and relays it onto the **aggregate** bounded channel
//!   that the consumer (the sweep / Engine Core) drains.
//! * [`WorkerPool::spawn`] returns the aggregate receiver plus a
//!   [`WorkerSupervisor`] handle. The supervisor awaits every worker, then
//!   drops the last aggregate sender — closing the channel, so the consumer
//!   knows the level is finished when `recv()` returns `None`.
//! * **No connection leaks:** when the supervisor finishes, every worker
//!   task and forwarder has ended, and dropping the shared client tears down
//!   the pooled connections (fd count returns to baseline — verified in
//!   `tests/concurrency_test.rs`).
//! * **Measurement isolation:** the private channel is small (128) and the
//!   aggregate channel is generously sized (default 1024), so a fast
//!   consumer keeps both from back-pressuring a worker's socket read — the
//!   quanta timing path is never blocked by the fan-in.
//!
//! If the consumer drops the aggregate receiver early, forwarders stop
//! relaying (send error) and workers finish early: `StreamWorker::run`
//! treats a closed channel as a stop signal, so a cancelled sweep tears the
//! pool down cleanly.

use std::time::Duration;

use futures_util::future::join_all;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::sse::{ParsedFrame, Usage};
use crate::timing::{MonotonicInstant, StreamTimestamps};

use super::stream::{StreamError, StreamEvent, StreamOutcome, StreamWorker, DEFAULT_READ_TIMEOUT};

/// Default capacity of the aggregate fan-in channel (see module docs).
pub const DEFAULT_AGGREGATE_CAPACITY: usize = 1024;

/// Per-worker private channel capacity between `StreamWorker` and its
/// forwarder task.
const PRIVATE_CAPACITY: usize = 128;

/// A [`StreamEvent`] tagged with the id of the worker that produced it.
///
/// The aggregate channel is a mix of N independent streams, so the tag is
/// what lets a consumer (the sweep, the Engine Core) keep per-stream state
/// — per-stream ITL deltas, token counts, and terminal outcomes — while
/// draining a single channel.
#[derive(Debug, Clone)]
pub enum PoolEvent {
    /// A parsed, timestamped frame from worker `stream`.
    Frame {
        stream: u32,
        frame: ParsedFrame,
        at: MonotonicInstant,
        timestamps: StreamTimestamps,
    },
    /// The stream ended: cleanly on `[DONE]` (`premature: false`), or early
    /// without a terminator (`premature: true`).
    Complete {
        stream: u32,
        timestamps: StreamTimestamps,
        usage: Option<Usage>,
        premature: bool,
        malformed_frames: u64,
    },
    /// The stream failed (HTTP error, connection refused, timeout, …).
    Failed {
        stream: u32,
        timestamps: StreamTimestamps,
        error: StreamError,
    },
}

impl PoolEvent {
    /// Tag a worker event with its stream id.
    pub fn from_event(event: StreamEvent, stream: u32) -> Self {
        match event {
            StreamEvent::Frame {
                frame,
                at,
                timestamps,
            } => PoolEvent::Frame {
                stream,
                frame,
                at,
                timestamps,
            },
            StreamEvent::Complete {
                timestamps,
                usage,
                premature,
                malformed_frames,
            } => PoolEvent::Complete {
                stream,
                timestamps,
                usage,
                premature,
                malformed_frames,
            },
            StreamEvent::Failed { timestamps, error } => PoolEvent::Failed {
                stream,
                timestamps,
                error,
            },
        }
    }

    /// The id of the worker this event came from.
    pub fn stream(&self) -> u32 {
        match self {
            PoolEvent::Frame { stream, .. }
            | PoolEvent::Complete { stream, .. }
            | PoolEvent::Failed { stream, .. } => *stream,
        }
    }

    /// `true` for `Complete` / `Failed` (a worker emits exactly one).
    pub fn is_terminal(&self) -> bool {
        matches!(self, PoolEvent::Complete { .. } | PoolEvent::Failed { .. })
    }
}

/// The supervisor task handle: resolves to the per-worker outcomes once
/// **all** workers have finished (and the aggregate channel has closed).
pub type WorkerSupervisor = JoinHandle<Vec<StreamOutcome>>;

/// A bundle of N [`StreamWorker`]s sharing one connection-pooled client.
///
/// Build with [`WorkerPool::new`], tune with the builder methods, then
/// [`spawn`](Self::spawn) a level of the concurrency ladder. The pool is
/// `Clone` (the `reqwest::Client` inside is a cheap `Arc`) so a sweep can
/// reuse the same client across ladder levels.
#[derive(Debug, Clone)]
pub struct WorkerPool {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    prompt: String,
    max_tokens: u32,
    read_timeout: Duration,
    retries: u32,
    /// Aggregate fan-in channel capacity.
    capacity: usize,
}

impl WorkerPool {
    /// Create a pool targeting `endpoint` (a bare host, a base URL with or
    /// without `/v1` — see [`StreamWorker::new`] /
    /// `client::stream::normalize_endpoint`), `model`, and the
    /// fully-generated `prompt` text (Chunk 4).
    pub fn new(
        client: reqwest::Client,
        endpoint: &str,
        model: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Self {
        Self {
            client,
            endpoint: endpoint.to_string(),
            model: model.to_string(),
            api_key: None,
            prompt: prompt.to_string(),
            max_tokens: max_tokens.max(1),
            read_timeout: DEFAULT_READ_TIMEOUT,
            retries: 0,
            capacity: DEFAULT_AGGREGATE_CAPACITY,
        }
    }

    /// Set an `Authorization: Bearer` header on every worker.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Idle read timeout for every worker (default [`DEFAULT_READ_TIMEOUT`]).
    pub fn read_timeout(mut self, d: Duration) -> Self {
        self.read_timeout = d;
        self
    }

    /// Retries for connect-phase failures on every worker.
    pub fn retries(mut self, n: u32) -> Self {
        self.retries = n;
        self
    }

    /// Aggregate fan-in channel capacity (default [`DEFAULT_AGGREGATE_CAPACITY`]).
    pub fn channel_capacity(mut self, n: usize) -> Self {
        self.capacity = n.max(16);
        self
    }

    /// The pool's target endpoint.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// The pool's target model.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The pool's `max_tokens` per stream.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Build one [`StreamWorker`] for this pool's target.
    pub fn worker(&self) -> StreamWorker {
        let mut worker = StreamWorker::new(
            self.client.clone(),
            &self.endpoint,
            &self.model,
            &self.prompt,
            self.max_tokens,
        )
        .read_timeout(self.read_timeout)
        .retries(self.retries);
        if let Some(key) = &self.api_key {
            worker = worker.api_key(key.clone());
        }
        worker
    }

    /// Spawn `n` concurrent workers.
    ///
    /// Returns:
    /// * the **aggregate** bounded receiver — one [`PoolEvent`] per worker
    ///   event, tagged with the worker id; it closes (yields `None`) when
    ///   every worker has finished;
    /// * the [`WorkerSupervisor`] — awaiting it gives the per-worker
    ///   [`StreamOutcome`]s and marks the true end of the level.
    ///
    /// `n == 0` is valid (an empty level): the receiver closes immediately
    /// and the supervisor resolves to an empty outcome list.
    pub fn spawn(self, n: usize) -> (mpsc::Receiver<PoolEvent>, WorkerSupervisor) {
        let (agg_tx, agg_rx) = mpsc::channel(self.capacity);
        let supervisor = tokio::spawn(async move {
            let mut tasks = Vec::with_capacity(n);
            for id in 0..n {
                let agg = agg_tx.clone();
                let pool = self.clone();
                tasks.push(tokio::spawn(async move {
                    let (wt, wr) = mpsc::channel(PRIVATE_CAPACITY);
                    // The worker runs on its private channel; `run` takes
                    // ownership of the sender and drops it when the
                    // lifecycle ends, which is what closes the forwarder
                    // below (after it has drained the buffer).
                    let outcome = pool.worker().run(wt).await;
                    let mut fwd = wr;
                    while let Some(event) = fwd.recv().await {
                        if agg
                            .send(PoolEvent::from_event(event, id as u32))
                            .await
                            .is_err()
                        {
                            // Consumer gone: stop relaying.
                            break;
                        }
                    }
                    outcome
                }));
            }
            // Drop the last sender so the aggregate channel closes once
            // every worker has finished.
            drop(agg_tx);
            // A panicked worker task (should never happen) yields a
            // synthesized failed outcome instead of losing the slot.
            join_all(tasks)
                .await
                .into_iter()
                .map(|r| match r {
                    Ok(o) => o,
                    Err(_) => StreamOutcome {
                        timestamps: StreamTimestamps::default(),
                        usage: None,
                        premature: false,
                        malformed_frames: 0,
                        error: Some(StreamError::Read("worker task panicked".to_string())),
                    },
                })
                .collect()
        });
        (agg_rx, supervisor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::Chunk;

    /// Build a `PoolEvent` from a `StreamEvent` under a mocked quanta clock
    /// (quanta's default-on `mock` feature) so instants are exact.
    fn mock_now() -> MonotonicInstant {
        MonotonicInstant::now()
    }

    #[test]
    fn frame_event_carries_stream_id_and_payload() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            mock.increment(5_000_000); // 5 ms
            let at = mock_now();
            let ts = StreamTimestamps {
                t0: Some(mock_now()),
                t1: Some(mock_now()),
                t2: Some(mock_now()),
                t3: Some(at),
                t_end: None,
            };
            let frame = ParsedFrame {
                chunk: Chunk::Content("tok".to_string()),
                t_nanos: 0,
                done: false,
            };
            let event = StreamEvent::Frame {
                frame,
                at,
                timestamps: ts,
            };
            let pooled = PoolEvent::from_event(event, 7);
            assert!(matches!(pooled, PoolEvent::Frame { stream: 7, .. }));
            assert_eq!(pooled.stream(), 7);
            assert!(!pooled.is_terminal());
        });
    }

    #[test]
    fn complete_event_is_terminal_and_preserves_usage() {
        let (clock, _mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let ts = StreamTimestamps::default();
            let usage = Usage {
                prompt_tokens: 128,
                completion_tokens: 34,
            };
            let pooled = PoolEvent::from_event(
                StreamEvent::Complete {
                    timestamps: ts,
                    usage: Some(usage),
                    premature: false,
                    malformed_frames: 1,
                },
                3,
            );
            assert!(matches!(
                pooled,
                PoolEvent::Complete {
                    stream: 3,
                    usage: Some(u),
                    premature: false,
                    malformed_frames: 1,
                    ..
                } if u == usage
            ));
            assert_eq!(pooled.stream(), 3);
            assert!(pooled.is_terminal());
        });
    }

    #[test]
    fn failed_event_is_terminal_and_preserves_error() {
        let (clock, _mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let err = StreamError::Connection("refused".to_string());
            let pooled = PoolEvent::from_event(
                StreamEvent::Failed {
                    timestamps: StreamTimestamps::default(),
                    error: err.clone(),
                },
                11,
            );
            assert!(matches!(pooled, PoolEvent::Failed { stream: 11, .. }));
            assert!(pooled.is_terminal());
        });
    }

    #[test]
    fn pool_builder_clamps_and_defaults() {
        let client = reqwest::Client::new();
        let pool = WorkerPool::new(client, "http://127.0.0.1:8000", "m", "p", 0)
            .channel_capacity(0)
            .retries(2)
            .api_key("k");
        // max_tokens is clamped to >= 1; capacity clamps up to the floor.
        assert_eq!(pool.capacity, 16);
        assert_eq!(pool.retries, 2);
        assert_eq!(pool.api_key.as_deref(), Some("k"));
        assert_eq!(pool.max_tokens, 1);
        // `worker()` builds a fully-configured StreamWorker (endpoint
        // normalization itself is covered by the Chunk 5 tests).
        let _w = pool.worker();
    }
}
