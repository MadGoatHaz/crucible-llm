//! Server-Sent Events (SSE) parsing: incremental low-allocation frame parser
//! and the `Chunk` model (Reasoning / Content / Control / Usage).

pub mod chunk;
pub mod parser;
