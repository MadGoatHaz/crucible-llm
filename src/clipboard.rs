//! Clipboard — a zero-dependency **OSC 52** copy path (design:
//! `work/plans/MATRIX_REPORT.md`, §2).
//!
//! The Matrix Report block is copied to the terminal's clipboard by writing
//! the standard DEC/X3.100 part-52 sequence,
//! `\x1b]52;c;<base64>\x07`, to a fresh [`std::io::stdout`]. No
//! `base64` / `arboard` / `clipboard` crate is involved — the project is
//! zero-runtime-dependency (`Cargo.toml`), so the base64 encoder here is
//! hand-rolled (RFC 4648 §4, **encode-only**: the copy direction never
//! decodes).
//!
//! **N/A-never-fail:** any `io` error is returned (not panicked) so the
//! caller can log `"clipboard unsupported by terminal — use [S]ave"`. Legacy
//! terminals (e.g. xterm) silently ignore OSC 52; save-to-file is the
//! guaranteed path. Raw mode + the alternate screen do not block an
//! independent stdout write (the codebase already does this:
//! `event.rs` `EnterAlternateScreen` / `LeaveAlternateScreen`).

/// The RFC 4648 §4 standard base64 alphabet (`A-Z a-z 0-9 + /`).
const B64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode `data` to standard base64 (RFC 4648 §4).
///
/// 3-byte groups map to 4 alphabet chars; a 1-byte remainder yields 2 chars +
/// `==`, a 2-byte remainder yields 3 chars + `=`. The input is consumed in
/// place — no allocation beyond the output `String`.
#[must_use]
pub fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);

    for chunk in data.as_chunks::<3>().0 {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]);
        out.push(B64_ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(B64_ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(B64_ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push(B64_ALPHABET[(n & 0x3F) as usize] as char);
    }

    match data.len() % 3 {
        1 => {
            let n = u32::from(data[data.len() - 1]) << 16;
            out.push(B64_ALPHABET[((n >> 18) & 0x3F) as usize] as char);
            out.push(B64_ALPHABET[((n >> 12) & 0x3F) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let n = (u32::from(data[data.len() - 2]) << 16) | (u32::from(data[data.len() - 1]) << 8);
            out.push(B64_ALPHABET[((n >> 18) & 0x3F) as usize] as char);
            out.push(B64_ALPHABET[((n >> 12) & 0x3F) as usize] as char);
            out.push(B64_ALPHABET[((n >> 6) & 0x3F) as usize] as char);
            out.push('=');
        }
        _ => {}
    }

    out
}

/// Build the full **OSC 52** copy sequence for `text` (the pure, testable
/// seam): `"\x1b]52;c;<base64>\x07"`.
#[must_use]
pub fn osc52(text: &str) -> String {
    format!("\x1b]52;c;{b64}\x07", b64 = b64_encode(text.as_bytes()))
}

/// Write a sequence to any sink and flush it (the testable seam for the
/// error path — `copy_to_clipboard` is this over a fresh stdout).
fn write_seq(out: &mut impl std::io::Write, seq: &str) -> std::io::Result<()> {
    out.write_all(seq.as_bytes())?;
    out.flush()
}

/// Copy `text` to the terminal's clipboard via **OSC 52**, written to a
/// **fresh** [`std::io::stdout`].
///
/// **N/A-never-fail:** any `io` error is returned as a `String` (never
/// panicked), so the caller turns it into a logged
/// `"clipboard unsupported by terminal — use [S]ave"` message. A terminal
/// that drops the sequence simply shows nothing; save-to-file is the
/// guaranteed path.
pub fn copy_to_clipboard(text: &str) -> Result<(), String> {
    let seq = osc52(text);
    let mut stdout = std::io::stdout();
    write_seq(&mut stdout, &seq)
        .map_err(|e| format!("OSC 52 clipboard write failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── b64_encode: known vectors ─────────────────────────────────────────

    #[test]
    fn b64_known_vectors() {
        assert_eq!(b64_encode(b""), "");
        assert_eq!(b64_encode(b"a"), "YQ==");
        assert_eq!(b64_encode(b"ab"), "YWI=");
        assert_eq!(b64_encode(b"abc"), "YWJj");
        // 4-byte and 5-byte cases exercise the 1- and 2-byte padding
        // remainders.
        assert_eq!(b64_encode(b"abcd"), "YWJjZA==");
        assert_eq!(b64_encode(b"hello"), "aGVsbG8=");
        // A space is an ordinary byte.
        assert_eq!(b64_encode(b"a b"), "YSBi");
    }

    #[test]
    fn b64_length_rule() {
        // For every length 0..300 the output is 4·ceil(n/3), with the right
        // `=` padding count.
        for n in 0..300u32 {
            let data = vec![0xAB; n as usize];
            let enc = b64_encode(&data);
            let want_len = n.div_ceil(3) * 4;
            assert_eq!(enc.len(), want_len as usize, "n={n}");
            let pad = match n % 3 {
                0 => 0,
                1 => 2,
                _ => 1,
            };
            let trailing_eq = enc.ends_with("==") as u32 * 2
                + (enc.ends_with('=') && !enc.ends_with("==")) as u32;
            assert_eq!(trailing_eq, pad, "n={n}");
        }
    }

    #[test]
    fn b64_alphabet_only() {
        // Encoding all 256 byte values uses only the RFC 4648 alphabet.
        let all: Vec<u8> = (0..=255u16).map(u8::try_from).map(Result::unwrap).collect();
        let enc = b64_encode(&all);
        for c in enc.chars() {
            assert!(
                c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=',
                "unexpected base64 char {c:?}"
            );
        }
    }

    // ── osc52: the exact sequence shape ───────────────────────────────────

    #[test]
    fn osc52_shape() {
        let seq = osc52("a");
        assert_eq!(seq, "\x1b]52;c;YQ==\x07");
        // General shape: prefix, a base64 middle, the BEL terminator.
        let seq = osc52("hello");
        assert!(seq.starts_with("\x1b]52;c;"), "missing OSC 52 prefix");
        assert!(seq.ends_with('\x07'), "missing BEL terminator");
        let middle = seq
            .strip_prefix("\x1b]52;c;")
            .and_then(|s| s.strip_suffix('\x07'))
            .expect("prefix+suffix must strip to the base64 middle");
        assert_eq!(middle, "aGVsbG8=");
    }

    // ── write_seq / copy_to_clipboard: the error path (N/A-never-fail) ────

    /// A writer that always fails — the seam that proves a broken sink yields
    /// `Err`, never a panic.
    struct FailingWriter;
    impl std::io::Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("simulated write failure"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn write_seq_failing_writer_returns_err_not_panic() {
        let mut w = FailingWriter;
        let res = write_seq(&mut w, "any-seq");
        assert!(res.is_err(), "a failing writer must yield Err, never panic");
    }

    #[test]
    fn write_seq_ok_writer_succeeds() {
        let mut buf = Vec::new();
        let res = write_seq(&mut buf, "\x1b]52;c;YQ==\x07");
        assert!(res.is_ok());
        assert_eq!(buf, b"\x1b]52;c;YQ==\x07");
    }

    // Smoke: the real entry point returns `Ok` when it can write to stdout.
    // In a piped `cargo test` there is no terminal to interpret the escape
    // sequence — it is inert bytes on the pipe.
    #[test]
    fn copy_to_clipboard_returns_ok_on_real_stdout() {
        assert!(copy_to_clipboard("a").is_ok());
    }
}
