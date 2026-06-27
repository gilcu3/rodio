//! Pitch-preserving playback speed control (time-scale modification).
//!
//! The main concept of this module is the [`TimeStretch`] struct.
//! [`Speed`](crate::source::Speed) changes tempo by relabeling the sample rate,
//! and therefore also shifts the pitch. `TimeStretch` instead changes the
//! *tempo* of the audio while keeping the *pitch* constant, the way modern media
//! players do when you speed up or slow down playback.
//!
//! This is achieved with **WSOLA** (Waveform Similarity Overlap-Add): the signal
//! is cut into overlapping windowed grains that are laid back down at a different
//! rate. A short similarity search aligns successive grains so the waveform stays
//! phase-continuous, which avoids the echo/phasiness of plain overlap-add.
//!
//! To speed up a source while preserving pitch, call
//! [`speed_preserve_pitch`](crate::Source::speed_preserve_pitch):
//!
#![cfg_attr(not(feature = "playback"), doc = "```ignore")]
#![cfg_attr(feature = "playback", doc = "```no_run")]
//!# use std::fs::File;
//!# use rodio::{Decoder, source::Source};
//! let handle = rodio::DeviceSinkBuilder::open_default_sink()
//!         .expect("open default audio sink");
//! let file = File::open("examples/music.ogg").unwrap();
//! let source = Decoder::try_from(file).unwrap();
//! // Play the sound 1.5x faster, keeping the original pitch.
//! handle.mixer().add(source.speed_preserve_pitch(1.5));
//! std::thread::sleep(std::time::Duration::from_secs(5));
//! ```
//!
//! When the factor is exactly `1.0`, or when pitch preservation is turned off via
//! [`TimeStretch::set_preserve_pitch`], the source is a zero-overhead passthrough
//! that behaves exactly like [`Speed`](crate::source::Speed).

use std::collections::VecDeque;
use std::time::Duration;

use super::SeekError;
use crate::common::{ChannelCount, Float, SampleRate};
use crate::{math, Sample, Source};

/// Analysis/synthesis window length, in milliseconds.
const WINDOW_MS: Float = 28.0;
/// Maximum similarity-search deviation, in milliseconds. Must exceed one pitch
/// period of the lowest expected voice (~70 Hz ≈ 14 ms) so the search can align
/// grains in phase; otherwise low voices warble.
const SEARCH_MS: Float = 14.0;
/// Reported span length, in frames, while time-stretching. Decoder-sized so the
/// downstream resampler gets useful chunks. Must stay small enough that
/// `frames * channels` does not exceed the resampler's internal span cap.
const SPAN_FRAMES: usize = 2048;

/// Internal function that builds a [`TimeStretch`] object.
pub(crate) fn time_stretch<I: Source>(input: I, factor: f32) -> TimeStretch<I> {
    let sample_rate = input.sample_rate();
    let channels = input.channels();
    TimeStretch {
        input,
        factor,
        preserve_pitch: true,
        sample_rate,
        channels,
        params_init: false,
        seg: 0,
        hs: 0,
        delta: 0,
        window: Vec::new(),
        in_buf: VecDeque::new(),
        base: 0,
        input_done: false,
        acc: Vec::new(),
        out_base: 0,
        out_buf: VecDeque::new(),
        ana_pos: 0.0,
        prev_seg_start: 0,
        first_grain: true,
        m: 0,
        finished: false,
    }
}

/// Filter that changes the playback tempo while preserving the pitch.
///
/// Unlike [`Speed`](crate::source::Speed), which shifts the pitch along with the
/// tempo, this performs time-scale modification so the pitch stays constant. It
/// uses **WSOLA** (Waveform Similarity Overlap-Add), the same algorithm as
/// MPlayer/ffmpeg's `scaletempo` filter and the `playbackRate` of web browsers.
/// Build one with
/// [`Source::speed_preserve_pitch`](crate::Source::speed_preserve_pitch).
#[doc(alias = "scaletempo")]
#[doc(alias = "wsola")]
#[doc(alias = "timestretch")]
#[derive(Clone, Debug)]
pub struct TimeStretch<I> {
    input: I,
    /// Tempo factor. `2.0` plays twice as fast, `0.5` half as fast.
    factor: f32,
    /// When `false` this behaves like [`Speed`](crate::source::Speed) (pitch is
    /// shifted along with the tempo).
    preserve_pitch: bool,

    // Cached input format. WSOLA parameters are derived from these.
    sample_rate: SampleRate,
    channels: ChannelCount,

    // WSOLA parameters (in frames, i.e. samples per channel).
    params_init: bool,
    /// Grain length.
    seg: usize,
    /// Synthesis hop (`seg / 2`, giving 50% overlap).
    hs: usize,
    /// Maximum similarity-search deviation.
    delta: usize,
    /// Hann analysis/synthesis window (one weight per frame).
    window: Vec<Float>,

    // Interleaved look-ahead buffer over the input.
    in_buf: VecDeque<Sample>,
    /// Absolute input frame index of the first frame in `in_buf`.
    base: i64,
    input_done: bool,

    // Interleaved overlap-add accumulator and the ready-to-emit output queue.
    acc: Vec<Sample>,
    /// Absolute output frame index of the first frame in `acc`.
    out_base: i64,
    out_buf: VecDeque<Sample>,

    /// Ideal analysis position (in input frames) of the next grain.
    ana_pos: f64,
    /// Input frame index where the previously emitted grain started.
    prev_seg_start: i64,
    first_grain: bool,
    /// Synthesis grain counter.
    m: i64,
    finished: bool,
}

impl<I> TimeStretch<I> {
    /// Modifies the tempo factor. `2.0` plays twice as fast, `0.5` half as fast.
    #[inline]
    pub fn set_factor(&mut self, factor: f32) {
        self.factor = factor;
    }

    /// Enables or disables pitch preservation.
    ///
    /// When disabled this source behaves like [`Speed`](crate::source::Speed):
    /// the tempo changes but so does the pitch, at no CPU cost.
    #[inline]
    pub fn set_preserve_pitch(&mut self, preserve_pitch: bool) {
        self.preserve_pitch = preserve_pitch;
    }

    /// Returns whether pitch is currently being preserved.
    #[inline]
    pub fn preserves_pitch(&self) -> bool {
        self.preserve_pitch
    }

    /// Returns a reference to the inner source.
    #[inline]
    pub fn inner(&self) -> &I {
        &self.input
    }

    /// Returns a mutable reference to the inner source.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut I {
        &mut self.input
    }

    /// Returns the inner source.
    #[inline]
    pub fn into_inner(self) -> I {
        self.input
    }

    /// When `true`, samples are passed through unmodified.
    #[inline]
    fn passthrough(&self) -> bool {
        !self.preserve_pitch || self.factor == 1.0
    }

    /// Resets all time-stretching state. Does not touch the input position.
    fn reset_state(&mut self) {
        self.in_buf.clear();
        self.base = 0;
        self.input_done = false;
        self.acc.clear();
        self.out_base = 0;
        self.out_buf.clear();
        self.ana_pos = 0.0;
        self.prev_seg_start = 0;
        self.first_grain = true;
        self.m = 0;
        self.finished = false;
    }
}

impl<I: Source> TimeStretch<I> {
    /// (Re)computes the WSOLA parameters when the input format changes.
    fn ensure_params(&mut self) {
        let sample_rate = self.input.sample_rate();
        let channels = self.input.channels();
        if self.params_init && sample_rate == self.sample_rate && channels == self.channels {
            return;
        }
        if self.params_init {
            // Format changed mid-stream: flush what we have and start fresh.
            self.acc.drain(..).for_each(|v| self.out_buf.push_back(v));
            self.in_buf.clear();
            self.base = 0;
            self.out_base = 0;
            self.ana_pos = 0.0;
            self.prev_seg_start = 0;
            self.first_grain = true;
            self.m = 0;
        }

        self.sample_rate = sample_rate;
        self.channels = channels;
        let sr = sample_rate.get() as Float;
        let mut seg = ((sr * WINDOW_MS / 1000.0) as usize).max(64);
        seg &= !1; // even, so the synthesis hop is exactly seg / 2
        self.seg = seg;
        self.hs = seg / 2;
        self.delta = ((sr * SEARCH_MS / 1000.0) as usize).max(1);
        self.window = (0..seg)
            .map(|n| 0.5 - 0.5 * (math::TAU * n as Float / seg as Float).cos())
            .collect();
        self.params_init = true;
    }

    /// Absolute frame index one past the last buffered input frame.
    #[inline]
    fn frames_avail(&self) -> i64 {
        let ch = self.channels.get() as usize;
        self.base + (self.in_buf.len() / ch) as i64
    }

    /// Pulls input frames until `in_buf` covers up to `up_to` (absolute frame).
    fn ensure(&mut self, up_to: i64) {
        let ch = self.channels.get() as usize;
        while !self.input_done && self.frames_avail() < up_to {
            match self.input.next() {
                Some(s) => self.in_buf.push_back(s),
                None => {
                    // Pad a trailing partial frame with silence.
                    while !self.in_buf.len().is_multiple_of(ch) {
                        self.in_buf.push_back(0.0);
                    }
                    self.input_done = true;
                }
            }
        }
    }

    /// Drops buffered input frames before `frame` to bound memory use.
    fn drop_before(&mut self, frame: i64) {
        if frame <= self.base {
            return;
        }
        let ch = self.channels.get() as usize;
        let avail = self.in_buf.len() / ch;
        let drop_frames = ((frame - self.base) as usize).min(avail);
        self.in_buf.drain(0..drop_frames * ch);
        self.base += drop_frames as i64;
    }

    /// Reads one buffered input sample, zero-padding outside the buffered range.
    #[inline]
    fn get(&self, frame: i64, c: usize) -> Sample {
        if frame < self.base {
            return 0.0;
        }
        let ch = self.channels.get() as usize;
        let idx = (frame - self.base) as usize * ch + c;
        self.in_buf.get(idx).copied().unwrap_or(0.0)
    }

    /// Finds, within `[-delta, delta]` of `ideal`, the grain start that best
    /// matches the natural continuation `template_start` of the previous grain.
    ///
    /// For speed this works on a contiguous mono downmix, correlates only over
    /// the overlap region (`hs` frames, not the full grain), and uses a
    /// coarse-to-fine search. That keeps it well within real-time.
    // Accumulating in `f64` keeps the correlation exact when `Sample` is `f32`;
    // the casts are no-ops (and redundant) under the `64bit` feature.
    #[allow(clippy::unnecessary_cast)]
    fn best_match(&self, ideal: i64, delta: i64, template_start: i64) -> i64 {
        let ch = self.channels.get() as usize;
        // Only the overlap region needs to match well.
        let corr_len = self.hs;

        // Copy the region spanning the template and every candidate into a
        // contiguous mono buffer, so the inner loop avoids per-channel and
        // `VecDeque` indexing overhead.
        let lo = (ideal - delta).min(template_start).max(self.base);
        let hi = (ideal + delta + corr_len as i64).max(template_start + corr_len as i64);
        let len = (hi - lo) as usize;
        let mut mono = vec![0.0 as Sample; len];
        for (f, slot) in mono.iter_mut().enumerate() {
            let mut s = 0.0 as Sample;
            for c in 0..ch {
                s += self.get(lo + f as i64, c);
            }
            *slot = s;
        }

        let template = &mono[(template_start - lo) as usize..][..corr_len];
        // Normalized cross-correlation of the candidate at offset `k` against
        // the template (the template energy is constant in `k`, so it is omitted
        // from the score).
        let score = |k: i64| -> f64 {
            let cand = ideal + k;
            if cand < 0 {
                return f64::NEG_INFINITY;
            }
            let candidate = &mono[(cand - lo) as usize..][..corr_len];
            let mut dot = 0.0f64;
            let mut norm = 0.0f64;
            for (x, t) in candidate.iter().zip(template) {
                dot += *x as f64 * *t as f64;
                norm += *x as f64 * *x as f64;
            }
            dot / (norm.sqrt() + 1e-9)
        };

        // Coarse pass over the search range, then refine around the best offset.
        // A ~0.5 ms coarse step samples the correlation lobe (≈ one pitch period
        // wide) finely enough that the refine pass lands on the true peak.
        let stride = (self.sample_rate.get() as i64 / 2000).max(1);
        let mut best_k = 0i64;
        let mut best_score = f64::NEG_INFINITY;
        let mut k = -delta;
        while k <= delta {
            let s = score(k);
            if s > best_score {
                best_score = s;
                best_k = k;
            }
            k += stride;
        }
        for k in (best_k - stride + 1).max(-delta)..=(best_k + stride - 1).min(delta) {
            let s = score(k);
            if s > best_score {
                best_score = s;
                best_k = k;
            }
        }
        ideal + best_k
    }

    /// Windows the grain starting at `start` and overlap-adds it into `acc`.
    fn add_grain(&mut self, start: i64) {
        let ch = self.channels.get() as usize;
        let out_off = (self.m * self.hs as i64 - self.out_base) as usize;
        let needed = (out_off + self.seg) * ch;
        if self.acc.len() < needed {
            self.acc.resize(needed, 0.0);
        }
        for f in 0..self.seg {
            let w = self.window[f];
            let dst = (out_off + f) * ch;
            for c in 0..ch {
                self.acc[dst + c] += w * self.get(start + f as i64, c);
            }
        }
    }

    /// Moves finalized output frames (those before `target`) into `out_buf`.
    fn flush_until(&mut self, target: i64) {
        if target <= self.out_base {
            return;
        }
        let ch = self.channels.get() as usize;
        let n = ((target - self.out_base) as usize).min(self.acc.len() / ch);
        self.out_buf.extend(self.acc.drain(0..n * ch));
        self.out_base += n as i64;
    }

    /// Runs a single WSOLA synthesis step, refilling `out_buf`.
    fn synthesize(&mut self) {
        self.ensure_params();
        let delta = self.delta as i64;
        let ideal = self.ana_pos.round() as i64;

        // Keep only the frames the current and next steps can still reference.
        let lower = (self.prev_seg_start + self.hs as i64)
            .min(ideal - delta)
            .max(0);
        self.drop_before(lower);
        self.ensure(ideal + delta + self.seg as i64 + 1);

        if self.input_done && ideal >= self.frames_avail() {
            // No more input to draw a meaningful grain from: flush the tail.
            self.acc.drain(..).for_each(|v| self.out_buf.push_back(v));
            self.finished = true;
            return;
        }

        let chosen = if self.first_grain {
            self.first_grain = false;
            ideal.max(0)
        } else {
            let template_start = self.prev_seg_start + self.hs as i64;
            self.best_match(ideal, delta, template_start)
        };

        self.add_grain(chosen);
        self.flush_until(self.m * self.hs as i64);

        self.prev_seg_start = chosen;
        self.ana_pos += self.hs as f64 * self.factor.max(0.01) as f64;
        self.m += 1;
    }
}

impl<I> Iterator for TimeStretch<I>
where
    I: Source,
{
    type Item = Sample;

    #[inline]
    fn next(&mut self) -> Option<Sample> {
        if self.passthrough() {
            // Emit anything we may have buffered before being switched to
            // passthrough, then read straight from the input.
            return self.out_buf.pop_front().or_else(|| self.input.next());
        }
        loop {
            if let Some(s) = self.out_buf.pop_front() {
                return Some(s);
            }
            if self.finished {
                return None;
            }
            self.synthesize();
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.passthrough() {
            self.input.size_hint()
        } else {
            (0, None)
        }
    }
}

impl<I> Source for TimeStretch<I>
where
    I: Source,
{
    #[inline]
    fn current_span_len(&self) -> Option<usize> {
        if self.passthrough() {
            self.input.current_span_len()
        } else if self.finished && self.out_buf.is_empty() {
            Some(0)
        } else {
            // The format is constant, but we must report a real (frame-aligned)
            // span rather than `None`: the queue turns `None` into a single-frame
            // span, which starves the downstream resampler and yields silence.
            Some(self.channels().get() as usize * SPAN_FRAMES)
        }
    }

    #[inline]
    fn channels(&self) -> ChannelCount {
        self.input.channels()
    }

    #[inline]
    fn sample_rate(&self) -> SampleRate {
        if self.preserve_pitch {
            self.input.sample_rate()
        } else {
            // Old `Speed` behaviour: relabel the rate so pitch shifts too.
            SampleRate::new((self.input.sample_rate().get() as f32 * self.factor).max(1.0) as u32)
                .expect("minimum is 1.0 > 0")
        }
    }

    #[inline]
    fn total_duration(&self) -> Option<Duration> {
        self.input.total_duration().map(|d| d.div_f32(self.factor))
    }

    #[inline]
    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        let pos_accounting_for_speedup = pos.mul_f32(self.factor);
        self.input.try_seek(pos_accounting_for_speedup)?;
        self.reset_state();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::SamplesBuffer;
    use crate::math::nz;

    const SR: u32 = 44_100;

    /// A finite mono sine wave as a `SamplesBuffer`.
    fn sine(freq: f32, samples: usize) -> SamplesBuffer {
        let data: Vec<Sample> = (0..samples)
            .map(|n| (math::TAU * freq as Float * n as Float / SR as Float).sin())
            .collect();
        SamplesBuffer::new(nz!(1), SampleRate::new(SR).unwrap(), data)
    }

    /// Estimates the dominant frequency of a mono signal via its zero-crossing rate.
    fn estimate_freq(samples: &[Sample]) -> f32 {
        let crossings = samples
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        (crossings as f32 / 2.0) / (samples.len() as f32 / SR as f32)
    }

    #[test]
    fn factor_one_is_identity() {
        let input: Vec<Sample> = (0..1000).map(|n| n as Sample * 0.001).collect();
        let buf = SamplesBuffer::new(nz!(1), SampleRate::new(SR).unwrap(), input.clone());
        let out: Vec<Sample> = buf.speed_preserve_pitch(1.0).collect();
        assert_eq!(out, input);
    }

    #[test]
    fn disabled_is_passthrough_with_pitch_shift() {
        let input: Vec<Sample> = (0..1000).map(|n| n as Sample * 0.001).collect();
        let buf = SamplesBuffer::new(nz!(1), SampleRate::new(SR).unwrap(), input.clone());
        let mut src = buf.speed_preserve_pitch(2.0);
        src.set_preserve_pitch(false);
        // Old `Speed` behaviour: rate is relabeled, samples are untouched.
        assert_eq!(src.sample_rate().get(), SR * 2);
        let out: Vec<Sample> = src.collect();
        assert_eq!(out, input);
    }

    #[test]
    fn preserves_pitch_when_sped_up() {
        let n = SR as usize; // 1 second
        let out: Vec<Sample> = sine(440.0, n).speed_preserve_pitch(2.0).collect();

        // Tempo doubled: about half as many output samples.
        let ratio = out.len() as f32 / n as f32;
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "expected ~0.5x length, got {ratio}"
        );

        // Pitch unchanged: still ~440 Hz (drop the windowed edges).
        let core = &out[out.len() / 8..out.len() * 7 / 8];
        let freq = estimate_freq(core);
        assert!((freq - 440.0).abs() < 15.0, "expected ~440 Hz, got {freq}");
    }

    #[test]
    fn preserves_pitch_when_slowed_down() {
        let n = SR as usize;
        let out: Vec<Sample> = sine(440.0, n).speed_preserve_pitch(0.5).collect();

        let ratio = out.len() as f32 / n as f32;
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "expected ~2x length, got {ratio}"
        );

        let core = &out[out.len() / 8..out.len() * 7 / 8];
        let freq = estimate_freq(core);
        assert!((freq - 440.0).abs() < 15.0, "expected ~440 Hz, got {freq}");
    }

    #[test]
    fn total_duration_scales_inversely() {
        let n = SR as usize;
        let src = sine(440.0, n).speed_preserve_pitch(2.0);
        let dur = src.total_duration().expect("buffer has a duration");
        assert!((dur.as_secs_f32() - 0.5).abs() < 0.01);
    }

    #[test]
    fn reports_exhausted_after_draining() {
        let mut src = sine(440.0, 5000).speed_preserve_pitch(1.5);
        let count = (&mut src).count();
        assert!(count > 0);
        assert_eq!(src.current_span_len(), Some(0));
        assert!(src.is_exhausted());
    }

    #[allow(clippy::unnecessary_cast)]
    fn rms(samples: &[Sample]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        (sum / samples.len() as f64).sqrt()
    }

    #[test]
    fn output_is_not_silent_mono() {
        let out: Vec<Sample> = sine(440.0, 48_000).speed_preserve_pitch(1.5).collect();
        let core = &out[out.len() / 8..out.len() * 7 / 8];
        assert!(rms(core) > 0.3, "rms too low: {}", rms(core));
    }

    #[test]
    fn output_is_not_silent_stereo() {
        let data: Vec<Sample> = (0..96_000)
            .map(|n| (math::TAU * 440.0 * (n / 2) as Float / SR as Float).sin())
            .collect();
        let buf = SamplesBuffer::new(nz!(2), SampleRate::new(SR).unwrap(), data);
        let out: Vec<Sample> = buf.speed_preserve_pitch(1.5).collect();
        let core = &out[out.len() / 8..out.len() * 7 / 8];
        assert!(rms(core) > 0.3, "rms too low: {}", rms(core));
    }

    #[test]
    fn not_silent_through_queue_and_resampler() {
        // Reproduces the Player path: the chain goes through the queue (which
        // rewrites `None` spans) and then a resampler to the device rate.
        use crate::queue;
        use crate::source::UniformSourceIterator;

        let n = 96_000;
        let data: Vec<Sample> = (0..n)
            .map(|i| (math::TAU * 440.0 * (i / 2) as Float / SR as Float).sin())
            .collect();
        let buf = SamplesBuffer::new(nz!(2), SampleRate::new(SR).unwrap(), data);

        let (input, output) = queue::queue(false);
        input.append(buf.speed_preserve_pitch(1.5));
        // Resample 44.1 kHz -> 48 kHz, as the device mixer would.
        let resampled =
            UniformSourceIterator::new(output, nz!(2), SampleRate::new(48_000).unwrap());

        let out: Vec<Sample> = resampled.collect();
        let core = &out[out.len() / 8..out.len() * 7 / 8];
        assert!(
            rms(core) > 0.3,
            "audio is (near) silent through the player path: rms {}",
            rms(core)
        );
    }

    #[test]
    #[cfg(feature = "playback")]
    fn not_silent_through_real_player() {
        // Highest-fidelity repro of the example: a real `Player` (factor starts
        // at 1.0 and transitions to 1.5 via periodic_access) feeding a resampler.
        use crate::source::UniformSourceIterator;

        let n = 192_000;
        let data: Vec<Sample> = (0..n)
            .map(|i| (math::TAU * 440.0 * (i / 2) as Float / SR as Float).sin())
            .collect();
        let buf = SamplesBuffer::new(nz!(2), SampleRate::new(SR).unwrap(), data);

        let (sink, queue_out) = crate::Player::new();
        sink.append(buf);
        sink.set_speed(1.5); // preserve_pitch defaults to true

        let mut uni =
            UniformSourceIterator::new(queue_out, nz!(2), SampleRate::new(48_000).unwrap());
        // Skip the first chunk (factor is still ramping in via periodic_access).
        let out: Vec<Sample> = (&mut uni).skip(20_000).take(80_000).collect();
        assert!(
            rms(&out) > 0.3,
            "audio is (near) silent through the real player: rms {}",
            rms(&out)
        );
    }

    #[test]
    fn stereo_keeps_channel_count() {
        let data: Vec<Sample> = (0..2000)
            .map(|n| (math::TAU * 440.0 * (n / 2) as Float / SR as Float).sin())
            .collect();
        let buf = SamplesBuffer::new(nz!(2), SampleRate::new(SR).unwrap(), data);
        let mut src = buf.speed_preserve_pitch(2.0);
        assert_eq!(src.channels().get(), 2);
        let count = (&mut src).count();
        // Output must stay frame-aligned (even number of samples for 2 channels).
        assert_eq!(count % 2, 0);
    }
}
