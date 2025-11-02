import React, { useMemo, useEffect, useRef, useState } from 'react';

/**
 * DeepfakeCard
 * - Horizontal scroller with arrows + snap
 * - Modal viewer for clicked frame (shows large image + meta + prev/next + keyboard nav)
 * - Accepts both .jpg and .png (tries original then fallback to the other extension, then /favicon.png)
 *
 * Props:
 *  - data: { label, confidence, metadata: { duration_sec, frames_analyzed }, top_frames: [...] }
 *  - animationMs: ms for the percent animation
 */
export default function DeepfakeCard({ data = {}, animationMs = 900 }) {
  const d = useMemo(() => data || {}, [data]);
  const frames = d.top_frames || [];
  const isFake = (d.label || '').toString().toUpperCase() === 'FAKE';

  // animated percent
  const targetPct = Math.round((d.confidence ?? 0) * 100);
  const [displayPct, setDisplayPct] = useState(0);
  const rafRef = useRef(null);
  const startRef = useRef(null);
  const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

  useEffect(() => {
    cancelAnimationFrame(rafRef.current);
    setDisplayPct(0);
    startRef.current = null;
    const animate = (ts) => {
      if (startRef.current == null) startRef.current = ts;
      const elapsed = ts - startRef.current;
      const t = Math.min(1, animationMs > 0 ? elapsed / animationMs : 1);
      const eased = easeOutCubic(t);
      const cur = Math.round(eased * targetPct);
      setDisplayPct(cur);
      if (t < 1) rafRef.current = requestAnimationFrame(animate);
      else setDisplayPct(targetPct);
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [targetPct, animationMs]);

  // modal index state (null = closed)
  const [modalIndex, setModalIndex] = useState(null);

  // keyboard handlers for modal navigation (Esc, ←, →)
  useEffect(() => {
    const onKey = (e) => {
      if (modalIndex == null) return;
      if (e.key === 'Escape') setModalIndex(null);
      if (e.key === 'ArrowLeft') setModalIndex((idx) => (idx > 0 ? idx - 1 : idx));
      if (e.key === 'ArrowRight') setModalIndex((idx) => (idx < frames.length - 1 ? idx + 1 : idx));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [modalIndex, frames.length]);

  // scroller refs + scroll controls
  const scrollerRef = useRef(null);
  const CARD_W = 220; // card width
  const GAP = 16; // gap used in layout
  const cardWidth = CARD_W + GAP;
  const scrollByCard = (dir = 1) => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollBy({ left: dir * cardWidth, behavior: 'smooth' });
  };

  // helpers
  const pct = (v) => Math.round((v ?? 0) * 100);
  const redGradient = 'linear-gradient(90deg, rgba(244,63,94,0.95) 0%, rgba(185,28,28,0.95) 100%)';
  const redShadow = (color) => `0 0 8px ${color}33`;
  const meterColor = (score) =>
    score >= 0.9 ? '#F43F5E' : score >= 0.75 ? '#FB7185' : '#FCA5A5';

  // compute most fake (for badge)
  const maxFake = frames.length ? Math.max(...frames.map((f) => f.fake_confidence ?? 0)) : 0;

  // IMAGE FALLBACK LOGIC: try original, then swap jpg<->png, then /favicon.png
  const makeSwapCandidates = (url) => {
    if (!url || typeof url !== 'string') return ['/favicon.png'];
    const lower = url.toLowerCase();
    const candidates = [url];
    if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) {
      candidates.push(url.replace(/\.(jpe?g)$/i, '.png'));
    } else if (lower.endsWith('.png')) {
      candidates.push(url.replace(/\.png$/i, '.jpg'));
    } else {
      // if no ext, try adding png/jpg
      candidates.push(`${url}.png`, `${url}.jpg`);
    }
    candidates.push('/favicon.png');
    return candidates;
  };

  // custom hook: returns handler to progressively try fallbacks when <img> onError fires
  const makeImgOnErrorHandler = (candidates) => {
    let idx = 0;
    return (e) => {
      try {
        e.currentTarget.onerror = null;
        idx += 1;
        if (idx < candidates.length) {
          e.currentTarget.src = candidates[idx];
          // reattach handler to attempt next fallback if this fails
          e.currentTarget.onerror = makeImgOnErrorHandler(candidates.slice(idx));
        } else {
          e.currentTarget.src = '/favicon.png';
        }
      } catch {
        e.currentTarget.src = '/favicon.png';
      }
    };
  };

  return (
    <>
      <section className="bg-slate-800 text-slate-100 rounded-2xl p-6 shadow-lg w-full">
        <div className="text-lg font-semibold tracking-wider text-white mb-4">Deepfake Verdict</div>

        <div className="bg-slate-900/60 border border-slate-700 rounded-xl p-4 flex flex-col md:flex-row items-center justify-between gap-4">
          {/* capsule */}
          <div className="flex items-center gap-4">
            <span
              className="inline-flex items-center justify-center px-6 py-2 rounded-full text-lg font-semibold tracking-wide border bg-rose-800/50 text-rose-100 border-rose-800/40"
              style={{ minWidth: 110 }}
            >
              Fake
            </span>
          </div>

          <div className="flex-1" />

          {/* animated percent */}
          <div className="flex flex-col items-end">
            <div className="text-xs text-slate-400">confidence</div>
            <div className="text-4xl font-semibold tabular-nums mt-1" aria-live="polite">{displayPct}%</div>
          </div>
        </div>

        {/* stats pills */}
        <div className="mt-4 flex gap-3">
          <div className="px-3 py-2 rounded-md bg-slate-900/60 border border-slate-700 text-slate-200 text-sm">
            Duration: <span className="font-medium ml-2">{d.metadata?.duration_sec ?? '—'}s</span>
          </div>

          <div className="px-3 py-2 rounded-md bg-slate-900/60 border border-slate-700 text-slate-200 text-sm">
            Frames analyzed: <span className="font-medium ml-2">{d.metadata?.frames_analyzed ?? '—'}</span>
          </div>
        </div>

        {/* Frames (single-row scroller with arrows) */}
        {isFake && (
          <section className="mt-6 bg-transparent relative">
            <div className="text-lg font-semibold tracking-wider text-white mb-4">Top Frames</div>

            {/* arrows (absolute, appear on hover of container) */}
            <button
              onClick={() => scrollByCard(-1)}
              aria-label="Scroll left"
              className="hidden md:flex items-center justify-center absolute left-0 top-1/2 -translate-y-1/2 z-20 w-10 h-10 rounded-full bg-slate-900/60 border border-slate-700 text-slate-200 hover:bg-slate-800 transition"
              style={{ marginLeft: -14 }}
            >
              ‹
            </button>

            <button
              onClick={() => scrollByCard(1)}
              aria-label="Scroll right"
              className="hidden md:flex items-center justify-center absolute right-0 top-1/2 -translate-y-1/2 z-20 w-10 h-10 rounded-full bg-slate-900/60 border border-slate-700 text-slate-200 hover:bg-slate-800 transition"
              style={{ marginRight: -14 }}
            >
              ›
            </button>

            <div
              ref={scrollerRef}
              className="flex gap-4 overflow-x-auto pb-2 pl-1 pr-6"
              style={{ scrollSnapType: 'x mandatory', WebkitOverflowScrolling: 'touch' }}
            >
              {frames.map((f, i) => {
                const score = f.fake_confidence ?? 0;
                const color = meterColor(score);
                const candidates = makeSwapCandidates(f.url);
                return (
                  <div
                    key={i}
                    className="relative bg-slate-700/40 rounded-xl p-3 w-[220px] flex-shrink-0 flex flex-col items-start hover:bg-slate-700/60 transition"
                    style={{ scrollSnapAlign: 'start' }}
                  >
                    {score === maxFake && (
                      <div className="absolute top-3 left-3 bg-rose-600/90 text-white text-xs px-2 py-0.5 rounded-full">Most fake</div>
                    )}

                    <div
                      className="relative w-full h-32 rounded-lg overflow-hidden mb-3 cursor-pointer group"
                      onClick={() => setModalIndex(i)}
                    >
                      <img
                        src={candidates[0]}
                        alt={`frame-${f.frame_index}`}
                        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                        onError={makeImgOnErrorHandler(candidates)}
                      />

                      <div className="absolute inset-0 bg-slate-900/0 group-hover:bg-slate-900/45 transition-opacity flex flex-col items-center justify-center opacity-0 group-hover:opacity-100">
                        <div className="text-white font-medium text-sm">{pct(score)}% fake</div>
                        <div className="text-slate-300 text-xs mt-1">{f.timestamp_sec}s</div>
                      </div>
                    </div>

                    <div className="w-full text-sm flex justify-between items-center">
                      <div className="text-slate-200 font-medium">{pct(score)}% fake</div>
                      <div className="text-xs text-slate-400">{f.timestamp_sec}s</div>
                    </div>

                    <div className="mt-2 w-full h-2 bg-slate-700/50 rounded overflow-hidden">
                      <div
                        className="h-full rounded transition-all duration-700"
                        style={{
                          width: `${pct(score)}%`,
                          background: redGradient,
                          boxShadow: redShadow(color),
                        }}
                      />
                    </div>

                    <div className="mt-2 text-xs text-slate-400">frame #{f.frame_index}</div>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </section>

      {/* Modal viewer */}
      {modalIndex != null && frames[modalIndex] && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setModalIndex(null)}
        >
          <div className="relative max-w-[92%] max-h-[92%] rounded-lg overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <button onClick={() => setModalIndex(null)} className="absolute top-3 right-3 z-10 bg-black/50 text-white rounded-full p-2 hover:bg-black/70" aria-label="Close">✕</button>

            {/* Prev / Next inside modal */}
            <button
              onClick={() => setModalIndex((idx) => Math.max(0, idx - 1))}
              className="absolute left-3 top-1/2 -translate-y-1/2 z-10 bg-black/40 text-white rounded-full p-2 hover:bg-black/60"
              aria-label="Previous"
            >
              ‹
            </button>

            <button
              onClick={() => setModalIndex((idx) => Math.min(frames.length - 1, idx + 1))}
              className="absolute right-3 top-1/2 -translate-y-1/2 z-10 bg-black/40 text-white rounded-full p-2 hover:bg-black/60"
              aria-label="Next"
            >
              ›
            </button>

            <div className="bg-slate-800 p-4 rounded-lg flex flex-col items-center">
              {/* modal image with fallback candidates */}
              <img
                src={makeSwapCandidates(frames[modalIndex].url)[0]}
                alt={`frame-large-${frames[modalIndex].frame_index}`}
                className="block max-w-full max-h-[70vh] object-contain bg-black"
                onError={makeImgOnErrorHandler(makeSwapCandidates(frames[modalIndex].url))}
              />

              <div className="mt-3 text-center">
                <div className="text-lg font-semibold text-white">{pct(frames[modalIndex].fake_confidence)}% fake</div>
                <div className="text-sm text-slate-300 mt-1">frame #{frames[modalIndex].frame_index} • {frames[modalIndex].timestamp_sec}s</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
