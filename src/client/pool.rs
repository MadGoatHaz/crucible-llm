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

use std::sync::Arc;
use std::time::Duration;

use futures_util::future::join_all;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::log::RunLogger;
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
    ///
    /// `looping` / `loop_excluded_tokens` (v0.1.1 decode-loop guard): the
    /// stream's guard verdict, so the sweep can exclude its tokens from
    /// throughput.
    Complete {
        stream: u32,
        timestamps: StreamTimestamps,
        usage: Option<Usage>,
        premature: bool,
        malformed_frames: u64,
        looping: bool,
        loop_excluded_tokens: u64,
    },
    /// The stream failed (HTTP error, connection refused, timeout, …).
    Failed {
        stream: u32,
        timestamps: StreamTimestamps,
        error: StreamError,
        looping: bool,
        loop_excluded_tokens: u64,
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
                looping,
                loop_excluded_tokens,
            } => PoolEvent::Complete {
                stream,
                timestamps,
                usage,
                premature,
                malformed_frames,
                looping,
                loop_excluded_tokens,
            },
            StreamEvent::Failed {
                timestamps,
                error,
                looping,
                loop_excluded_tokens,
            } => PoolEvent::Failed {
                stream,
                timestamps,
                error,
                looping,
                loop_excluded_tokens,
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
#[derive(Clone)]
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
    /// Optional run logger shared by every worker (HTTP / SSE lifecycle
    /// lines). `Debug` is implemented manually below (`RunLogger` has no
    /// `Clone`-free `Debug` derive issue, but the field is printed by
    /// hand to keep the derive simple).
    logger: Option<Arc<RunLogger>>,
    /// The per-worker max lifetime (the sweep's timeout fix): a worker
    /// that has not finished within this window is killed and recorded
    /// as a timeout failure. `None` (the default) means no cap.
    worker_timeout: Option<Duration>,
    /// Disallow the model's own end-token from stopping generation on
    /// every worker (`ignore_eos`; honored by llama.cpp servers, ignored
    /// by other backends). Engine F sets this so the 60-second window —
    /// not the model — is the only stop.
    ignore_eos: bool,
}

impl std::fmt::Debug for WorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerPool")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .field("max_tokens", &self.max_tokens)
            .field("read_timeout", &self.read_timeout)
            .field("retries", &self.retries)
            .field("capacity", &self.capacity)
            .field("logger", &self.logger.is_some())
            .field("worker_timeout", &self.worker_timeout)
            .field("ignore_eos", &self.ignore_eos)
            .finish()
    }
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
            logger: None,
            worker_timeout: None,
            ignore_eos: false,
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

    /// Attach the shared run logger: every worker then records its HTTP /
    /// SSE lifecycle lines (tagged with its worker tag).
    pub fn logger(mut self, logger: Arc<RunLogger>) -> Self {
        self.logger = Some(logger);
        self
    }

    /// The per-worker max lifetime (the sweep's timeout fix). A worker
    /// that has not finished within `d` is killed (`tokio::time::timeout`
    /// drops its future, closing its socket) and recorded as a
    /// [`StreamError::WorkerTimeout`] failure with its partial results.
    /// `None` (the default) leaves workers unbounded.
    pub fn worker_timeout(mut self, d: Duration) -> Self {
        self.worker_timeout = Some(d);
        self
    }

    /// The pool's per-worker max lifetime (for the sweep's step budget).
    pub fn worker_timeout_cap(&self) -> Option<Duration> {
        self.worker_timeout
    }

    /// Disallow the model's own end-token from stopping generation on
    /// every worker (`"ignore_eos": true`; honored by llama.cpp servers,
    /// unknown to — and ignored by — other backends). Engine F sets this
    /// so the 60-second window, not the model, is the only stop.
    pub fn ignore_eos(mut self, v: bool) -> Self {
        self.ignore_eos = v;
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

    /// A copy of this pool with a different prompt text (v0.1.1 2D
    /// concurrency × context matrix: the same pool settings, a
    /// context-sized prompt).
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Build one [`StreamWorker`] for this pool's target (tag `stream`).
    pub fn worker(&self) -> StreamWorker {
        self.worker_tagged("stream")
    }

    /// Build one [`StreamWorker`] for this pool's target with the given
    /// log tag (e.g. `B5-3` — Engine B, step 5, worker 3).
    pub fn worker_tagged(&self, tag: &str) -> StreamWorker {
        let mut worker = StreamWorker::new(
            self.client.clone(),
            &self.endpoint,
            &self.model,
            &self.prompt,
            self.max_tokens,
        )
        .read_timeout(self.read_timeout)
        .retries(self.retries)
        .tag(tag);
        if self.ignore_eos {
            worker = worker.ignore_eos(true);
        }
        if let Some(key) = &self.api_key {
            worker = worker.api_key(key.clone());
        }
        if let Some(logger) = &self.logger {
            worker = worker.logger(logger.clone());
        }
        worker
    }

    /// Spawn `n` concurrent workers (log tag prefix `pool`).
    ///
    /// Returns:
    /// * the **aggregate** bounded receiver — one [`PoolEvent`] per worker
    ///   event, tagged with the worker id; it closes (yields `None`) when
    ///   every worker has finished (or been killed by the per-worker
    ///   timeout, if one is set);
    /// * the [`WorkerSupervisor`] — awaiting it gives the per-worker
    ///   [`StreamOutcome`]s and marks the true end of the level.
    ///
    /// `n == 0` is valid (an empty level): the receiver closes immediately
    /// and the supervisor resolves to an empty outcome list.
    pub fn spawn(self, n: usize) -> (mpsc::Receiver<PoolEvent>, WorkerSupervisor) {
        self.spawn_tagged(n, "pool")
    }

    /// [`spawn`](Self::spawn) with a per-worker log tag prefix: worker
    /// `id` is tagged `{prefix}-{id}` in its HTTP / SSE log lines (the
    /// sweep passes `B{step}` so a step-5 worker 3 logs as `B5-3`).
    pub fn spawn_tagged(
        self,
        n: usize,
        prefix: &str,
    ) -> (mpsc::Receiver<PoolEvent>, WorkerSupervisor) {
        // Owned by the `'static` worker tasks (the `&str` cannot escape).
        let prefix = prefix.to_string();
        let (agg_tx, agg_rx) = mpsc::channel(self.capacity);
        let supervisor = tokio::spawn(async move {
            let mut tasks = Vec::with_capacity(n);
            for id in 0..n {
                let agg = agg_tx.clone();
                let pool = self.clone();
                let prefix = prefix.clone();
                tasks.push(tokio::spawn(async move {
                    let (wt, wr) = mpsc::channel(PRIVATE_CAPACITY);
                    // The forwarder runs **concurrently** with the worker,
                    // draining the private channel as frames arrive. (The
                    // old sequential "run to completion, then drain" order
                    // deadlocked any stream longer than the 128-slot
                    // private buffer: the worker blocked on a full
                    // channel with no consumer.)
                    let fwd_task = tokio::spawn(async move {
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
                    });
                    let tag = format!("{prefix}-{id}");
                    // The per-worker timeout (the freeze fix): the whole
                    // worker future is bounded; on expiry it is dropped
                    // (the socket closes) and a synthesized `Failed`
                    // terminal event is relayed so the consumer sees the
                    // timeout as an ordinary stream failure.
                    let keep = wt.clone();
                    let outcome = match pool.worker_timeout {
                        Some(d) => {
                            match tokio::time::timeout(d, pool.worker_tagged(&tag).run(wt)).await {
                                Ok(o) => o,
                                Err(_elapsed) => {
                                    let _ = keep
                                        .send(StreamEvent::Failed {
                                            timestamps: StreamTimestamps::default(),
                                            error: StreamError::WorkerTimeout(d),
                                            looping: false,
                                            loop_excluded_tokens: 0,
                                        })
                                        .await;
                                    StreamOutcome {
                                        timestamps: StreamTimestamps::default(),
                                        usage: None,
                                        premature: false,
                                        malformed_frames: 0,
                                        error: Some(StreamError::WorkerTimeout(d)),
                                        looping: false,
                                        loop_excluded_tokens: 0,
                                        frames: 0,
                                    }
                                }
                            }
                        }
                        None => pool.worker_tagged(&tag).run(wt).await,
                    };
                    // The worker dropped its sender; drop the timeout
                    // clone too so the forwarder sees the channel close,
                    // then let it drain the remainder. (A forwarder panic
                    // is impossible — it only `recv`s and `send`s — the
                    // `JoinError` is discarded.)
                    drop(keep);
                    let _ = fwd_task.await;
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
                        looping: false,
                        loop_excluded_tokens: 0,
                        frames: 0,
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
                    looping: false,
                    loop_excluded_tokens: 0,
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
                    looping: false,
                    loop_excluded_tokens: 0,
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
        // No per-worker cap by default.
        assert_eq!(pool.worker_timeout_cap(), None);
        let capped = pool.worker_timeout(Duration::from_secs(5));
        assert_eq!(capped.worker_timeout_cap(), Some(Duration::from_secs(5)));
    }

    // ── Live-network pool tests (loopback mock only) ─────────────────────

    /// A minimal HTTP/1.1 mock: accepts one connection, reads the request
    /// head, then writes `frames` SSE `data:` frames (plus `[DONE]`) and
    /// closes. `hang` accepts but never writes (the stall scenario).
    async fn mock_server(frames: usize, hang: bool) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (sock, _) = match listener.accept().await {
                Ok(c) => c,
                Err(_) => return,
            };
            tokio::spawn(async move {
                let mut sock = sock;
                let mut buf = [0u8; 8192];
                let mut acc = Vec::new();
                while !acc.windows(4).any(|w| w == b"\r\n\r\n") {
                    match sock.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => acc.extend_from_slice(&buf[..n]),
                    }
                }
                if hang {
                    // The user's freeze scenario: the server stops sending
                    // and never closes. Release the socket when the client
                    // (the killed worker) disconnects, so no fd outlives
                    // the test.
                    let mut hb = [0u8; 64];
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_secs(300)) => {}
                        _ = async {
                            loop {
                                match sock.read(&mut hb).await {
                                    Ok(0) | Err(_) => return,
                                    Ok(_) => continue,
                                }
                            }
                        } => {}
                    }
                    return;
                }
                let head =
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
                let _ = sock.write_all(head).await;
                for i in 0..frames {
                    let frame = format!(
                        "data: {{\"choices\":[{{\"delta\":{{\"content\":\"t{i} \"}}}}]}}\n\n"
                    );
                    let _ = sock.write_all(frame.as_bytes()).await;
                }
                let _ = sock.write_all(b"data: [DONE]\n\n").await;
                let _ = sock.shutdown().await;
            });
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn long_stream_over_private_capacity_does_not_deadlock() {
        // 300 frames > the 128-slot private channel: the forwarder must
        // drain concurrently or the worker blocks forever (regression for
        // the sequential run-then-drain deadlock).
        let url = mock_server(300, false).await;
        let (mut rx, supervisor) =
            WorkerPool::new(reqwest::Client::new(), &url, "m", "p", 300).spawn(1);
        let mut terminal = None;
        while let Some(event) = rx.recv().await {
            if event.is_terminal() {
                terminal = Some(event);
            }
        }
        let outcomes = tokio::time::timeout(Duration::from_secs(10), supervisor)
            .await
            .expect("a >128-frame stream must not deadlock the pool")
            .unwrap();
        assert!(outcomes[0].is_ok());
        assert!(matches!(
            terminal,
            Some(PoolEvent::Complete {
                premature: false,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn worker_timeout_kills_a_hung_worker_and_reports_partial() {
        // The user's freeze: the server accepts but stops sending. A 2s
        // per-worker cap must kill the worker, close the level, and report
        // a `WorkerTimeout` failure.
        let url = mock_server(0, true).await;
        let (mut rx, supervisor) = WorkerPool::new(reqwest::Client::new(), &url, "m", "p", 64)
            .read_timeout(Duration::from_secs(30)) // stall window > cap
            .worker_timeout(Duration::from_secs(2))
            .spawn(1);
        let mut terminal = None;
        while let Some(event) = rx.recv().await {
            if event.is_terminal() {
                terminal = Some(event);
            }
        }
        let outcomes = tokio::time::timeout(Duration::from_secs(10), supervisor)
            .await
            .expect("the per-worker timeout must end the level")
            .unwrap();
        assert!(matches!(
            outcomes[0].error,
            Some(StreamError::WorkerTimeout(_))
        ));
        assert!(matches!(
            terminal,
            Some(PoolEvent::Failed {
                error: StreamError::WorkerTimeout(_),
                ..
            })
        ));
    }

    #[tokio::test]
    async fn worker_timeout_lets_fast_workers_finish() {
        // A stream that completes well inside the cap is unaffected.
        let url = mock_server(20, false).await;
        let (mut rx, supervisor) = WorkerPool::new(reqwest::Client::new(), &url, "m", "p", 64)
            .worker_timeout(Duration::from_secs(10))
            .spawn(1);
        let mut terminals = 0;
        while let Some(event) = rx.recv().await {
            if event.is_terminal() {
                terminals += 1;
            }
        }
        let outcomes = supervisor.await.unwrap();
        assert_eq!(terminals, 1);
        assert!(outcomes[0].is_ok());
    }
}
