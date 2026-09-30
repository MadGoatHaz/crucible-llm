//! Model discovery client: `GET {base}/models` against an
//! OpenAI-compatible endpoint (vLLM, llama.cpp, LM Studio, …).
//!
//! [`list_models`] resolves a user-supplied URL (bare host, `/v1` base,
//! or full completions path — see [`normalize_base_url`]) to the models
//! endpoint, sends a `GET` (with an `Authorization: Bearer` header when
//! an API key is provided), and parses the OpenAI-shaped response
//! `{"object": "list", "data": [ModelInfo, …]}` into a deduped,
//! id-sorted list of [`ModelInfo`]s.
//!
//! Error handling mirrors the [`crate::client::stream::StreamError`]
//! taxonomy: non-2xx → [`ModelError::Http`], connection failure →
//! [`ModelError::Connection`], stalled request → [`ModelError::Timeout`],
//! unparseable body → [`ModelError::InvalidJson`]. A discovery failure
//! never panics — the TUI degrades to a free-text model field (the
//! N/A-never-fail rule, blueprint §5D).

use std::time::Duration;

use serde::Deserialize;
use thiserror::Error;

/// Chat-completions suffix a user URL may end in (stripped for discovery,
/// keeping the `/v1` it sits under).
const CHAT_COMPLETIONS_SUFFIX: &str = "/chat/completions";

/// A base URL that already ends in `/v1` only needs `/models` appended.
const V1_SUFFIX: &str = "/v1";

/// Overall request timeout for a discovery call: model lists are small,
/// so a server that cannot answer within this window is stuck.
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);

/// One entry of an OpenAI-compatible `/v1/models` response
/// (`data[i]`). All fields except `id` degrade to `None`/empty when the
/// server omits them.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct ModelInfo {
    /// The model identifier (e.g. `Qwen3.6-35B-A3B-UD-Q4_K_XL`).
    pub id: String,
    /// Usually `"model"`.
    pub object: Option<String>,
    /// Creation timestamp (Unix seconds), when the server reports one.
    pub created: Option<u64>,
    /// The organization that owns the model (e.g. `vllm`, `openai`).
    pub owned_by: Option<String>,
}

/// The envelope of a `/v1/models` response.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

/// Failure modes for a model discovery call (mirrors
/// [`crate::client::stream::StreamError`]).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ModelError {
    /// The server answered with a non-2xx status (e.g. `404` — the
    /// endpoint has no `/models` route; `401` — bad API key).
    #[error("HTTP {status}: {body}")]
    Http {
        status: u16,
        /// The response body, truncated for logging.
        body: String,
    },
    /// The connection could not be established (refused, DNS failure,
    /// connect timeout, reset before a response).
    #[error("connection failed: {0}")]
    Connection(String),
    /// No response arrived within the discovery timeout.
    #[error("discovery timed out after {0:?}")]
    Timeout(Duration),
    /// A 2xx response body that is not valid models JSON.
    #[error("models response is not valid JSON: {0}")]
    InvalidJson(String),
}

impl ModelError {
    /// Whether a fresh attempt is likely to succeed.
    ///
    /// Only connect-phase failures are retriable — same rule as
    /// [`crate::client::stream::StreamError::is_retriable`]: a
    /// definitive HTTP answer is a *measurement* of the endpoint, not a
    /// transient fault.
    pub fn is_retriable(&self) -> bool {
        matches!(self, Self::Connection(_))
    }
}

/// Resolve a user-supplied URL to the **base** of an OpenAI-compatible
/// API, so [`list_models] can append `/models`:
///
/// * `http://host:8000` → `http://host:8000/v1`
/// * `http://host:8000/` → same (trailing slashes ignored)
/// * `http://host:8000/v1` → unchanged
/// * `http://host:8000/v1/chat/completions` → `http://host:8000/v1`
pub fn normalize_base_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if let Some(stripped) = base.strip_suffix(CHAT_COMPLETIONS_SUFFIX) {
        stripped.trim_end_matches('/').to_string()
    } else if base.ends_with(V1_SUFFIX) {
        base.to_string()
    } else {
        format!("{base}{V1_SUFFIX}")
    }
}

/// List the models served by an OpenAI-compatible endpoint.
///
/// `base_url` may be a bare host, a `/v1` base, or a full completions
/// URL (it is normalized via [`normalize_base_url`]); the request goes to
/// `{base}/models`. When `api_key` is `Some`, it is sent as
/// `Authorization: Bearer`.
///
/// The result is deduped by `id` and sorted alphabetically; entries
/// without an `id` are dropped.
pub async fn list_models(
    base_url: &str,
    api_key: Option<&str>,
) -> Result<Vec<ModelInfo>, ModelError> {
    let base = normalize_base_url(base_url);
    let url = format!("{base}/models");

    let client = reqwest::Client::builder()
        .timeout(DISCOVERY_TIMEOUT)
        .build()
        .map_err(|e| ModelError::Connection(e.to_string()))?;

    let mut req = client.get(&url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            if e.is_timeout() {
                return Err(ModelError::Timeout(DISCOVERY_TIMEOUT));
            }
            return Err(ModelError::Connection(e.to_string()));
        }
    };

    let status = resp.status().as_u16();
    let body = resp
        .text()
        .await
        .map_err(|e| ModelError::Connection(e.to_string()))?;
    handle_models_response(status, &body)
}

/// Pure response handling (unit-testable without a live server):
/// non-2xx → [`ModelError::Http`]; a 2xx body is parsed into the
/// [`ModelInfo`] list and cleaned (drop empty ids, dedupe, sort).
pub fn handle_models_response(status: u16, body: &str) -> Result<Vec<ModelInfo>, ModelError> {
    if !(200..300).contains(&status) {
        return Err(ModelError::Http {
            status,
            body: truncate_body(body, 256),
        });
    }
    let envelope: ModelsResponse =
        serde_json::from_str(body).map_err(|e| ModelError::InvalidJson(e.to_string()))?;
    Ok(clean_models(envelope.data))
}

/// Drop entries without an `id`, dedupe by `id`, sort alphabetically.
pub fn clean_models(mut data: Vec<ModelInfo>) -> Vec<ModelInfo> {
    data.retain(|m| !m.id.is_empty());
    data.sort_by(|a, b| a.id.cmp(&b.id));
    data.dedup();
    data
}

/// Truncate `s` to at most `max` bytes on a character boundary.
fn truncate_body(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── normalize_base_url ──────────────────────────────────────────────

    #[test]
    fn normalize_bare_host_gets_v1() {
        assert_eq!(
            normalize_base_url("http://localhost:8000"),
            "http://localhost:8000/v1"
        );
    }

    #[test]
    fn normalize_strips_trailing_slash() {
        assert_eq!(
            normalize_base_url("http://localhost:8000/"),
            "http://localhost:8000/v1"
        );
        assert_eq!(
            normalize_base_url("http://localhost:8000/v1/"),
            "http://localhost:8000/v1"
        );
    }

    #[test]
    fn normalize_v1_base_unchanged() {
        assert_eq!(
            normalize_base_url("https://api.example.com/v1"),
            "https://api.example.com/v1"
        );
    }

    #[test]
    fn normalize_full_completions_path_stripped_to_v1() {
        assert_eq!(
            normalize_base_url("http://localhost:8000/v1/chat/completions"),
            "http://localhost:8000/v1"
        );
    }

    // ── handle_models_response (mock bodies, no live server) ────────────

    #[test]
    fn response_200_parses_model_entries() {
        let body = r#"{
            "object": "list",
            "data": [
                {
                    "id": "qwen3-32b",
                    "object": "model",
                    "created": 1718000000,
                    "owned_by": "vllm"
                },
                {
                    "id": "llama-3.1-8b-instruct",
                    "object": "model",
                    "created": 1710000000,
                    "owned_by": "meta"
                }
            ]
        }"#;
        let models = handle_models_response(200, body).unwrap();
        // Sorted by id.
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["llama-3.1-8b-instruct", "qwen3-32b"]
        );
        let qwen = &models[1];
        assert_eq!(qwen.object.as_deref(), Some("model"));
        assert_eq!(qwen.created, Some(1718000000));
        assert_eq!(qwen.owned_by.as_deref(), Some("vllm"));
    }

    #[test]
    fn response_200_tolerates_missing_optional_fields() {
        let body = r#"{"data": [{"id": "bare-model"}]}"#;
        let models = handle_models_response(200, body).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "bare-model");
        assert_eq!(models[0].object, None);
        assert_eq!(models[0].created, None);
        assert_eq!(models[0].owned_by, None);
    }

    #[test]
    fn response_200_empty_data_is_an_empty_list() {
        let models = handle_models_response(200, r#"{"object": "list", "data": []}"#).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn response_404_is_an_http_error_with_truncated_body() {
        let long = "x".repeat(500);
        let e = handle_models_response(404, &long).unwrap_err();
        match e {
            ModelError::Http { status, body } => {
                assert_eq!(status, 404);
                assert!(
                    body.len() < 500,
                    "body must be truncated: {len}",
                    len = body.len()
                );
            }
            other => panic!("expected Http, got {other:?}"),
        }
    }

    #[test]
    fn response_500_is_an_http_error() {
        let e = handle_models_response(500, "internal error").unwrap_err();
        assert!(matches!(e, ModelError::Http { status: 500, .. }));
    }

    #[test]
    fn response_invalid_json_is_an_invalid_json_error() {
        let e = handle_models_response(200, "this is not json").unwrap_err();
        assert!(matches!(e, ModelError::InvalidJson(_)));
    }

    #[test]
    fn clean_models_drops_empty_ids_dedupes_and_sorts() {
        let models = clean_models(vec![
            ModelInfo {
                id: "zeta".into(),
                ..Default::default()
            },
            ModelInfo {
                id: String::new(),
                ..Default::default()
            },
            ModelInfo {
                id: "alpha".into(),
                ..Default::default()
            },
            ModelInfo {
                id: "zeta".into(),
                ..Default::default()
            },
        ]);
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["alpha", "zeta"]
        );
    }

    // ── list_models (real network, loopback only) ───────────────────────

    #[test]
    fn list_models_connection_refused_is_a_connection_error() {
        // Port 1 on loopback is not listening: a connect-phase failure,
        // not an HTTP answer. No external network is touched.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time() // reqwest's client timeout needs the time driver
            .build()
            .unwrap();
        let e = rt
            .block_on(list_models("http://127.0.0.1:1", None))
            .unwrap_err();
        assert!(
            matches!(e, ModelError::Connection(_)),
            "expected Connection, got {e:?}"
        );
        assert!(e.is_retriable());
    }

    #[test]
    fn only_connection_errors_are_retriable() {
        assert!(ModelError::Connection("refused".into()).is_retriable());
        assert!(!ModelError::Http {
            status: 404,
            body: "no route".into()
        }
        .is_retriable());
        assert!(!ModelError::Timeout(Duration::from_secs(10)).is_retriable());
        assert!(!ModelError::InvalidJson("nope".into()).is_retriable());
    }
}
