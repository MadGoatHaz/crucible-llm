//! Matrix Report — the pure, UI-agnostic core of the `[COMMUNITY-SUBMISSION]`
//! feature (design: `work/plans/MATRIX_REPORT.md`, §1).
//!
//! A [`MatrixReport`] is the eight-field block that the
//! <https://madgoathaz.github.io/llm-matrix/> auto-parser ingests. This module
//! owns the **serialization** (`render`) and the **assembly** (`build`) and the
//! single non-trivial pure rule — [`vram_tier`], which buckets a binary-MiB
//! VRAM total into the community's tier bands **without ever forcing a gap
//! value into a neighbouring tier** (the honesty rule: a number means what it
//! says).
//!
//! Everything here is `std`-only and fully unit-testable: no I/O, no TUI, no
//! hardware probes. The TUI (`M3`) and the headless path (`M4`) both call
//! [`MatrixReport::build`]; only *they* decide where the inputs come from.

/// One community-matrix submission — the eight `[COMMUNITY-SUBMISSION]`
/// fields. Pure data; [`MatrixReport::render`] is the only serialization.
///
/// The field semantics (which are auto vs. user-filled, and when each is
/// blank) are fixed by the design; see `MATRIX_REPORT.md` §1.1.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MatrixReport {
    /// Auto: `"{count}x {gpu} {vram}GB + {ram}GB RAM"` (degrades honestly —
    /// missing telemetry drops its token rather than inventing a number).
    pub hardware: String,
    /// Auto: `"{vram}GB — {tier}"` when a tier matches, `"{vram}GB"` in a
    /// gap, `""` when there is no VRAM telemetry.
    pub vram: String,
    /// Auto: the target model (from config).
    pub model: String,
    /// User: the inference backend (`vLLM` / `llama.cpp` / `LM Studio` / …).
    pub backend: String,
    /// User: the quantization (e.g. `Q4_K_M`).
    pub quant: String,
    /// Auto (optional): the real Engine A mean decode speed, `""` when there
    /// are no valid runs (never a silent `0`). A pure-CoT run's mean is its
    /// full-window **blended** decode rate (B1) — a real number, honestly
    /// labeled by [`Self::fill_pure_cot_note`].
    pub speed: String,
    /// User (optional): the use case (e.g. `daily chat`).
    pub use_case: String,
    /// User (optional): notes (launch flags, quirks). Auto-filled with
    /// [`Self::PURE_COT_NOTE`] when the speed is a pure-CoT blended rate
    /// and the user has not written their own (B1).
    pub notes: String,
}

impl MatrixReport {
    /// Serialize to the **exact** `[COMMUNITY-SUBMISSION]` block the
    /// auto-parser ingests.
    ///
    /// The block is the hard contract (`MATRIX_REPORT.md` invariant 6.6): two
    /// marker lines plus eight `key: value` lines in a fixed order. Every key
    /// is padded so the value column starts at a fixed offset — the prefix is
    /// the key plus `:` left-justified to 9 columns, then one space (so the
    /// value lands at column 11). **All eight lines are always emitted**, even
    /// when an optional value is blank: a key-based parser reads by key and a
    /// blank value parses as empty, so omitting a line is strictly riskier.
    ///
    /// The trailing newline is deliberate (file-save friendly; a
    /// line-oriented parser is unaffected).
    #[must_use]
    pub fn render(&self) -> String {
        format!(
            "[COMMUNITY-SUBMISSION]\n\
             hardware: {hardware}\n\
             vram:     {vram}\n\
             model:    {model}\n\
             backend:  {backend}\n\
             quant:    {quant}\n\
             speed:    {speed}\n\
             use_case: {use_case}\n\
             notes:    {notes}\n\
             [/COMMUNITY-SUBMISSION]\n",
            hardware = self.hardware,
            vram = self.vram,
            model = self.model,
            backend = self.backend,
            quant = self.quant,
            speed = self.speed,
            use_case = self.use_case,
            notes = self.notes,
        )
    }

    /// Assemble a report from the raw run/hardware facts. The single pure
    /// entry point both the TUI and the headless path call.
    ///
    /// Inputs:
    /// - `gpu_name` — the display name, `""` when no GPU is present.
    /// - `gpu_count` — the (vendor-agnostic) device count, `0` when no GPU.
    /// - `vram_total_mib` — **binary** MiB (`GpuSample.memory_total_mb`), `0`
    ///   when there is no VRAM telemetry.
    /// - `ram_total_bytes` — host RAM in bytes, `0` when no telemetry.
    /// - `model` / `backend` / `quant` / `use_case` / `notes` — copied
    ///   (trimmed) into the report; the user fields are typically `""` on the
    ///   headless path (the user completes them before submitting).
    /// - `speed_tps` — the real Engine A mean decode speed; `0.0` when there
    ///   are no valid runs.
    ///
    /// `#[allow(clippy::too_many_arguments)]`: 10 inputs is the established
    /// codebase precedent for an assembly fn (cf. `BenchmarkSequence::new`).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn build(
        gpu_name: &str,
        gpu_count: u32,
        vram_total_mib: u64,
        ram_total_bytes: u64,
        model: &str,
        speed_tps: f64,
        backend: &str,
        quant: &str,
        use_case: &str,
        notes: &str,
    ) -> Self {
        let vram = vram_tier(vram_total_mib);

        // VRAM is **binary** GiB (a "24 GB" card reports 24564 MiB); RAM is
        // **decimal** GB (matching the template's "128GB RAM"). Two bases in
        // one line is deliberate (design R6).
        let vram_gb = (vram_total_mib as f64 / 1024.0).round() as u64;
        let ram_gb = (ram_total_bytes as f64 / 1e9).round() as u64;

        let has_gpu = !gpu_name.is_empty();
        let has_vram = vram_total_mib > 0;
        let has_ram = ram_total_bytes > 0;

        // Honest degradation ladder: a missing fact drops its own token
        // (never a fabricated `0GB` / `1x …`).
        let hardware = if has_gpu && has_vram {
            // B2: a multi-GPU **pool** is unambiguous — the per-card VRAM
            // and the pool total. The old `{count}x {name} {total}GB`
            // read as count×total ("8x NVIDIA RTX A4000 128GB" = 8×128 GB);
            // a tp8 pool is 8×16 GB = 128 GB. Single-GPU form unchanged.
            let vram_part = if gpu_count > 1 {
                let per_gb = (vram_total_mib / gpu_count.max(1) as u64 / 1024) as f64;
                format!("({per:.0} GB each, {vram_gb} GB total)", per = per_gb.round())
            } else {
                format!("{vram_gb}GB")
            };
            if has_ram {
                format!("{gpu_count}x {gpu_name} {vram_part} + {ram_gb}GB RAM")
            } else {
                format!("{gpu_count}x {gpu_name} {vram_part}")
            }
        } else if has_gpu {
            // GPU present but no VRAM telemetry.
            if has_ram {
                format!("{gpu_count}x {gpu_name} + {ram_gb}GB RAM")
            } else {
                format!("{gpu_count}x {gpu_name}")
            }
        } else if has_ram {
            // No GPU (CPU-only host).
            format!("{ram_gb}GB RAM")
        } else {
            String::new()
        };

        // The real Engine A mean — never an estimate, never a silent 0.
        let speed = if speed_tps > 0.0 {
            format!("{speed_tps:.0} tok/s")
        } else {
            String::new()
        };

        Self {
            hardware,
            vram,
            model: model.trim().to_string(),
            backend: backend.trim().to_string(),
            quant: quant.trim().to_string(),
            speed,
            use_case: use_case.trim().to_string(),
            notes: notes.trim().to_string(),
        }
    }

    /// The honest label for a blended-fallback speed (B1): appended to
    /// `notes` when the speed field is a pure-CoT blended rate and the user
    /// has not written their own notes. The auto-parser reads by key
    /// ([`Self::render`]), so an auto-filled `notes` is contract-safe.
    pub const PURE_COT_NOTE: &str =
        "speed is the blended decode rate — all completion tokens were reasoning_content (pure CoT); the answer phase is N/A";

    /// B1: fill the `notes` field with [`Self::PURE_COT_NOTE`] when the
    /// report has a speed and the user has not written their own notes.
    ///
    /// The call sites (the TUI `build_matrix_report` and the headless
    /// `headless_matrix_report`) invoke this **only** when
    /// [`crate::engines::sequence::speed_mean_is_blended`] is `true` —
    /// i.e. every valid Engine A run was pure-CoT, so the speed field is
    /// the full-window blended decode rate, not an answer-phase
    /// measurement. A no-op when the user typed notes (preserved) or the
    /// speed is blank (nothing to explain).
    pub fn fill_pure_cot_note(&mut self) {
        if !self.speed.is_empty() && self.notes.is_empty() {
            self.notes = Self::PURE_COT_NOTE.to_string();
        }
    }
}

/// Bucket a **binary-MiB** VRAM total into the community's tier bands.
///
/// Returns the **full `vram` field string** (value + tier), so
/// [`MatrixReport::build`] assigns it directly:
/// - `"16GB — ≤16"` — a tier matched (the em-dash separates value from band).
/// - `"40GB"` — a **gap** value: never forced into a neighbouring tier.
/// - `""` — no telemetry (`total_mib == 0`); the N/A rule, never fabricated.
///
/// **Units:** the input is binary MiB (`GpuSample.memory_total_mb` — NVML
/// bytes ÷ 1024²). A "24 GB" card often reports 24564 MiB = 23.998 GiB and
/// must still land in `16-24`, so the comparison is fractional (`g` in GiB),
/// not a truncated integer.
///
/// **Tier table** (`g` in GiB; lower/upper bounds inclusive; top tier
/// open-ended):
///
/// | Condition on `g` | Band |
/// |---|---|
/// | `g == 0.0` | `""` (no telemetry) |
/// | `0 < g <= 16` | `≤16` |
/// | `16 < g <= 24` | `16-24` |
/// | `24 < g <= 32` | `24-32` |
/// | `32 < g < 48` | **GAP** → raw value |
/// | `48 <= g <= 64` | `48-64` |
/// | `64 < g < 96` | **GAP** → raw value |
/// | `96 <= g <= 128` | `96-128` |
/// | `128 < g < 196` | **GAP** → raw value |
/// | `g >= 196` | `196-256GB+` |
///
/// The display value is `g.round()` as an integer GB. Rounding is stable
/// across tier edges: `g = 24.6` → `25GB — 24-32`; `g = 24.4` → `24GB — 16-24`.
#[must_use]
pub fn vram_tier(total_mib: u64) -> String {
    let g = total_mib as f64 / 1024.0;

    // No telemetry: the N/A rule — a blank, never a fabricated tier.
    if g == 0.0 {
        return String::new();
    }

    // The band label, or `None` for a gap (which reports the raw value).
    let band = if g <= 16.0 {
        Some("≤16")
    } else if g <= 24.0 {
        Some("16-24")
    } else if g <= 32.0 {
        Some("24-32")
    } else if g < 48.0 {
        None // 32 < g < 48
    } else if g <= 64.0 {
        Some("48-64")
    } else if g < 96.0 {
        None // 64 < g < 96
    } else if g <= 128.0 {
        Some("96-128")
    } else if g < 196.0 {
        None // 128 < g < 196
    } else {
        Some("196-256GB+")
    };

    let gb = g.round() as u64;
    match band {
        Some(band) => format!("{gb}GB — {band}"),
        None => format!("{gb}GB"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── vram_tier: the full worked table (MATRIX_REPORT.md §1.2) ──────────

    #[test]
    fn vram_tier_full_table() {
        // (total_mib, expected string)
        let cases = [
            (0, ""),
            (8192, "8GB — ≤16"),
            (16384, "16GB — ≤16"),
            (16385, "16GB — 16-24"),
            (24576, "24GB — 16-24"),
            (24577, "24GB — 24-32"),
            (32768, "32GB — 24-32"),
            (33792, "33GB"),
            (40960, "40GB"),
            (47104, "46GB"),
            (48128, "47GB"),
            (49152, "48GB — 48-64"),
            (65536, "64GB — 48-64"),
            (66560, "65GB"),
            (96256, "94GB"),
            (98304, "96GB — 96-128"),
            (131072, "128GB — 96-128"),
            (132096, "129GB"),
            (144384, "141GB"),
            (199680, "195GB"),
            (200704, "196GB — 196-256GB+"),
            (262144, "256GB — 196-256GB+"),
            (263168, "257GB — 196-256GB+"),
        ];
        for (mib, want) in cases {
            assert_eq!(
                vram_tier(mib),
                want,
                "vram_tier({mib}) — got {:?}, want {want:?}",
                vram_tier(mib)
            );
        }
    }

    // The boundaries the task pins by integer GiB.
    #[test]
    fn vram_tier_task_boundaries() {
        assert_eq!(vram_tier(16 * 1024), "16GB — ≤16");
        assert_eq!(vram_tier(24 * 1024), "24GB — 16-24");
        assert_eq!(vram_tier(32 * 1024), "32GB — 24-32");
        assert_eq!(vram_tier(40 * 1024), "40GB"); // gap
        assert_eq!(vram_tier(48 * 1024), "48GB — 48-64");
        assert_eq!(vram_tier(64 * 1024), "64GB — 48-64");
        assert_eq!(vram_tier(80 * 1024), "80GB"); // gap
        assert_eq!(vram_tier(96 * 1024), "96GB — 96-128");
        assert_eq!(vram_tier(128 * 1024), "128GB — 96-128");
        assert_eq!(vram_tier(141 * 1024), "141GB"); // gap
        assert_eq!(vram_tier(196 * 1024), "196GB — 196-256GB+");
        assert_eq!(vram_tier(256 * 1024), "256GB — 196-256GB+");
        // No telemetry → blank (the N/A rule), NOT a fabricated tier.
        assert_eq!(vram_tier(0), "");
    }

    // The honesty rule: every value in the three gaps reports the raw number
    // and is never forced into a neighbouring tier.
    #[test]
    fn vram_tier_gap_never_forces_a_tier() {
        for gap in [33..=47, 65..=95, 129..=195] {
            for gi in gap {
                let s = vram_tier((gi * 1024) as u64);
                assert!(
                    s == format!("{gi}GB"),
                    "gap {gi} GiB must be raw — got {s:?}"
                );
                assert!(
                    !s.contains('—'),
                    "gap {gi} GiB must carry no tier — got {s:?}"
                );
            }
        }
    }

    // ── render: the exact block (markers + 8 lines + alignment) ───────────

    fn populated() -> MatrixReport {
        MatrixReport {
            hardware: "1x RTX 4090 24GB + 128GB RAM".to_string(),
            vram: "24GB — 16-24".to_string(),
            model: "qwen3.8-27b".to_string(),
            backend: "vLLM".to_string(),
            quant: "Q4_K_M".to_string(),
            speed: "45 tok/s".to_string(),
            use_case: "daily chat".to_string(),
            notes: "enable_thinking=false".to_string(),
        }
    }

    #[test]
    fn render_exact_block() {
        let got = populated().render();
        let want = "[COMMUNITY-SUBMISSION]\n\
                    hardware: 1x RTX 4090 24GB + 128GB RAM\n\
                    vram:     24GB — 16-24\n\
                    model:    qwen3.8-27b\n\
                    backend:  vLLM\n\
                    quant:    Q4_K_M\n\
                    speed:    45 tok/s\n\
                    use_case: daily chat\n\
                    notes:    enable_thinking=false\n\
                    [/COMMUNITY-SUBMISSION]\n";
        assert_eq!(got, want, "render() must be byte-for-byte the contract");
    }

    #[test]
    fn render_markers_and_line_order() {
        let rendered = populated().render();
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines.first().copied(), Some("[COMMUNITY-SUBMISSION]"));
        assert_eq!(lines.last().copied(), Some("[/COMMUNITY-SUBMISSION]"));
        // The eight keys, in order, each as `key:` at the start of its line.
        let keys = [
            "hardware:",
            "vram:",
            "model:",
            "backend:",
            "quant:",
            "speed:",
            "use_case:",
            "notes:",
        ];
        for (line, key) in lines[1..9].iter().zip(keys) {
            assert!(
                line.starts_with(key),
                "line {line:?} must start with {key:?}"
            );
        }
    }

    #[test]
    fn render_empty_optionals_are_blank_lines() {
        let mut r = populated();
        r.speed.clear();
        r.use_case.clear();
        r.notes.clear();
        let got = r.render();
        // The lines are still present (with a blank value) — never omitted.
        assert!(
            got.contains("speed:    \n"),
            "blank speed line must still be emitted"
        );
        assert!(
            got.contains("use_case: \n"),
            "blank use_case line must still be emitted"
        );
        assert!(
            got.contains("notes:    \n"),
            "blank notes line must still be emitted"
        );
        // No `null` / `N/A` fabrication.
        assert!(!got.contains("null") && !got.contains("N/A"));
    }

    // ── build ─────────────────────────────────────────────────────────────

    #[test]
    fn build_hardware_full() {
        let r = MatrixReport::build(
            "RTX 4090",
            1,
            24576,
            128 * 1_000_000_000,
            "qwen3.8-27b",
            45.4,
            "vLLM",
            "Q4_K_M",
            "",
            "",
        );
        assert_eq!(r.hardware, "1x RTX 4090 24GB + 128GB RAM");
    }

    #[test]
    fn build_hardware_multi_gpu() {
        // B2: a multi-GPU pool is unambiguous — the per-card VRAM and the
        // pool total (the old form read "2x … 24GB" as 2×24 GB).
        let r = MatrixReport::build(
            "RTX 4090",
            2,
            24576,
            64 * 1_000_000_000,
            "m",
            0.0,
            "",
            "",
            "",
            "",
        );
        assert_eq!(r.hardware, "2x RTX 4090 (12 GB each, 24 GB total) + 64GB RAM");
    }

    #[test]
    fn build_hardware_8x_a4000_pool_is_unambiguous() {
        // The operator's v0.1.7 host: a tp8 pool of 16 GB A4000s (131072
        // MiB total) + 118 decimal GB RAM. The old form read "8x … 128GB"
        // as 8×128 GB; the new form names both the per-card and pool sizes.
        let r = MatrixReport::build(
            "NVIDIA RTX A4000",
            8,
            131072,
            118 * 1_000_000_000,
            "qwen3.8-27b",
            56.4,
            "",
            "",
            "",
            "",
        );
        assert_eq!(
            r.hardware,
            "8x NVIDIA RTX A4000 (16 GB each, 128 GB total) + 118GB RAM"
        );
        // B3 (no change): the `vram` tier stays the pool aggregate — a 27B
        // model cannot fit on one 16 GB card; the tier describes what the
        // *deployment* needs.
        assert_eq!(r.vram, "128GB — 96-128");
        // And the pure-CoT blended mean renders as a real speed.
        assert_eq!(r.speed, "56 tok/s");
    }

    #[test]
    fn build_hardware_two_gpu_no_ram() {
        // Multi-GPU form with no RAM telemetry: the `+ …GB RAM` tail drops,
        // the per-card/total qualification stays.
        let r = MatrixReport::build("RTX 4090", 2, 24576, 0, "m", 0.0, "", "", "", "");
        assert_eq!(r.hardware, "2x RTX 4090 (12 GB each, 24 GB total)");
    }

    #[test]
    fn build_hardware_no_gpu() {
        let r = MatrixReport::build(
            "",
            0,
            0,
            32 * 1_000_000_000,
            "m",
            0.0,
            "",
            "",
            "",
            "",
        );
        assert_eq!(r.hardware, "32GB RAM");
    }

    #[test]
    fn build_hardware_gpu_no_vram() {
        // A GPU present but with no VRAM telemetry drops the VRAM token.
        let r = MatrixReport::build(
            "RTX 4090",
            1,
            0,
            32 * 1_000_000_000,
            "m",
            0.0,
            "",
            "",
            "",
            "",
        );
        assert_eq!(r.hardware, "1x RTX 4090 + 32GB RAM");
    }

    #[test]
    fn build_hardware_drops_ram_when_no_telemetry() {
        // GPU + VRAM but no RAM telemetry: the `+ …GB RAM` tail is dropped.
        let r = MatrixReport::build(
            "RTX 4090",
            1,
            24576,
            0,
            "m",
            0.0,
            "",
            "",
            "",
            "",
        );
        assert_eq!(r.hardware, "1x RTX 4090 24GB");
    }

    #[test]
    fn build_vram_via_tier() {
        // 24576 MiB = 24 GiB → the `16-24` band.
        let r = MatrixReport::build("g", 1, 24576, 0, "m", 0.0, "", "", "", "");
        assert_eq!(r.vram, "24GB — 16-24");
        // A gap value stays raw.
        let r = MatrixReport::build("g", 1, 40 * 1024, 0, "m", 0.0, "", "", "", "");
        assert_eq!(r.vram, "40GB");
        // No telemetry → blank.
        let r = MatrixReport::build("g", 1, 0, 0, "m", 0.0, "", "", "", "");
        assert_eq!(r.vram, "");
    }

    #[test]
    fn build_speed() {
        let r = MatrixReport::build("g", 1, 0, 0, "m", 45.4, "", "", "", "");
        assert_eq!(r.speed, "45 tok/s");
        // No valid runs → blank, never a silent 0.
        let r = MatrixReport::build("g", 1, 0, 0, "m", 0.0, "", "", "", "");
        assert_eq!(r.speed, "");
    }

    #[test]
    fn build_copies_trimmed_fields() {
        let r = MatrixReport::build(
            "g",
            1,
            0,
            0,
            "  qwen3.8-27b  ",
            0.0,
            " vLLM ",
            " Q4_K_M ",
            " daily chat ",
            " flags here ",
        );
        assert_eq!(r.model, "qwen3.8-27b");
        assert_eq!(r.backend, "vLLM");
        assert_eq!(r.quant, "Q4_K_M");
        assert_eq!(r.use_case, "daily chat");
        assert_eq!(r.notes, "flags here");
    }

    // ── B1: a pure-CoT blended speed renders, honestly noted ─────────

    #[test]
    fn build_pure_cot_blended_speed_populates_and_fills_the_note() {
        // The call-site flow (W2-int): `build` renders the
        // (blended-fallback) mean; `fill_pure_cot_note` — invoked only
        // when `speed_mean_is_blended` is true — appends the honest
        // explanation. A normal (non-pure-CoT) speed is never noted.
        let mut r = MatrixReport::build("g", 1, 0, 0, "m", 56.4, "", "", "", "");
        assert_eq!(r.speed, "56 tok/s", "the blended rate is a real number");
        assert_eq!(r.notes, "", "build alone never fabricates a note");
        r.fill_pure_cot_note();
        assert_eq!(r.notes, MatrixReport::PURE_COT_NOTE);
        // The filled report keeps the 8-key parser contract (value
        // column intact; the notes line carries the explanation).
        let rendered = r.render();
        assert!(rendered.contains("speed:    56 tok/s"), "{rendered}");
        assert!(
            rendered.contains(&format!("notes:    {}", MatrixReport::PURE_COT_NOTE)),
            "{rendered}"
        );
    }

    #[test]
    fn fill_pure_cot_note_never_touches_user_notes() {
        // A user who typed notes keeps them (the TUI's `open_report`
        // preserves in-progress edits; `fill` is a no-op on non-empty
        // notes).
        let mut r = MatrixReport::build("g", 1, 0, 0, "m", 56.4, "", "", "", "my notes");
        r.fill_pure_cot_note();
        assert_eq!(r.notes, "my notes");
    }

    #[test]
    fn fill_pure_cot_note_is_a_no_op_on_a_blank_speed() {
        // Genuinely no speed (0 and not pure-CoT) → the existing honesty
        // rule stands: blank, and no note to explain nothing.
        let mut r = MatrixReport::build("g", 1, 0, 0, "m", 0.0, "", "", "", "");
        assert_eq!(r.speed, "");
        r.fill_pure_cot_note();
        assert_eq!(r.notes, "");
    }
}
