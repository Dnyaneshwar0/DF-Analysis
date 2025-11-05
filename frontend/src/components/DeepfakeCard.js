// src/components/DeepfakeCard.js
import React, { useMemo, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

// ---------- Helper: build swap candidates ----------
const makeSwapCandidates = (url) => {
  if (!url || typeof url !== 'string') return ['/favicon.png'];
  const lower = url.toLowerCase();
  const candidates = [url];
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) {
    candidates.push(url.replace(/\.(jpe?g)$/i, '.png'));
  } else if (lower.endsWith('.png')) {
    candidates.push(url.replace(/\.png$/i, '.jpg'));
  } else {
    candidates.push(`${url}.png`, `${url}.jpg`);
  }
  // final fallback
  candidates.push('/favicon.png');
  return candidates;
};

// ---------- FrameThumb: memoized component ----------
const FrameThumb = React.memo(function FrameThumb({ url, alt, className = '', onClick }) {
  const candidates = useMemo(() => makeSwapCandidates(url), [url]);
  const [idx, setIdx] = useState(0);
  const finalSrc = idx < candidates.length ? candidates[idx] : '/favicon.png';
  const srcRef = useRef(finalSrc);

  // Keep cached src stable across re-renders (helps avoid duplicate requests)
  useEffect(() => {
    if (srcRef.current !== finalSrc) srcRef.current = finalSrc;
  }, [finalSrc]);

  const handleError = () => {
    setIdx((prev) => Math.min(prev + 1, candidates.length - 1));
    // eslint-disable-next-line no-console
    console.warn('FrameThumb: image failed to load, trying next candidate', candidates);
  };

  const handleClick = (e) => {
    if (e && typeof e.stopPropagation === 'function') e.stopPropagation();
    if (typeof onClick === 'function') onClick();
  };

  return (
    <img
      src={srcRef.current || '/favicon.png'}
      alt={alt}
      className={className}
      onError={handleError}
      onClick={handleClick}
      loading="lazy"
      decoding="async"
      draggable={false}
      style={{ cursor: 'pointer' }}
    />
  );
});

// ---------- Small Portal utility ----------
function ModalPortal({ children }) {
  const elRef = useRef(null);
  if (elRef.current == null && typeof document !== 'undefined') {
    elRef.current = document.createElement('div');
    elRef.current.setAttribute('data-portal', 'deepfake-modal');
  }

  useEffect(() => {
    if (!elRef.current) return;
    document.body.appendChild(elRef.current);
    return () => {
      if (elRef.current && elRef.current.parentNode) {
        elRef.current.parentNode.removeChild(elRef.current);
      }
    };
  }, []);

  return elRef.current ? createPortal(children, elRef.current) : null;
}

// ---------- DeepfakeCard ----------
export default function DeepfakeCard({ data = {}, animationMs = 2000 }) {
  const d = useMemo(() => data || {}, [data]);
  const frames = d.top_frames || [];
  const label = (d.label || '').toString().toUpperCase();
  const isFake = label === 'FAKE';
  const isReal = label === 'REAL';

  // ---------- Helpers ----------
  const formatTime = (secs) => {
    if (secs == null || isNaN(secs)) return '—';
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = Math.floor(secs % 60);
    const parts = [];
    if (h > 0) parts.push(String(h).padStart(2, '0'));
    parts.push(String(m).padStart(2, '0'));
    parts.push(String(s).padStart(2, '0'));
    return parts.join(':');
  };

  // ---------- Confidence Animation ----------
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

  // ---------- Styling ----------
  const verdictBg = isFake
    ? 'bg-gradient-to-br from-rose-900 via-rose-800 to-rose-900'
    : isReal
    ? 'bg-gradient-to-br from-emerald-900 via-emerald-800 to-emerald-900'
    : 'bg-slate-800';

  const verdictBorder = isFake
    ? 'border-rose-800'
    : isReal
    ? 'border-emerald-700'
    : 'border-slate-700';

  // ---------- Frames + Modal ----------
  const [modalIndex, setModalIndex] = useState(null);
  const [modalBarPct, setModalBarPct] = useState(0);
  const scrollerRef = useRef(null);
  const CARD_W = 220;
  const GAP = 16;

  const scrollByCard = (dir = 1) => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollBy({ left: dir * (CARD_W + GAP), behavior: 'smooth' });
  };

  const pct = (v) => Math.round((v ?? 0) * 100);
  const redGradient = 'linear-gradient(90deg, rgba(244,63,94,0.95) 0%, rgba(185,28,28,0.95) 100%)';
  const redShadow = (color) => `0 0 8px ${color}33`;
  const meterColor = (score) =>
    score >= 0.9 ? '#F43F5E' : score >= 0.75 ? '#FB7185' : '#FCA5A5';
  const maxFake = frames.length ? Math.max(...frames.map((f) => f.fake_confidence ?? 0)) : 0;

  // ---------- Animate modal footer bar ----------
  useEffect(() => {
    let raf = null;
    let start = null;
    const duration = 700;
    if (modalIndex == null) {
      setModalBarPct(0);
      return;
    }
    const target = pct(frames[modalIndex]?.fake_confidence ?? 0);
    setModalBarPct(0);
    const step = (ts) => {
      if (start == null) start = ts;
      const elapsed = ts - start;
      const t = Math.min(1, elapsed / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const cur = Math.round(eased * target);
      setModalBarPct(cur);
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => {
      if (raf) cancelAnimationFrame(raf);
    };
  }, [modalIndex, frames]);

  // Keyboard navigation
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

  // ---------- Inline Icons ----------
  const IconClock = ({ className = 'w-4 h-4' }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M12 7v5l3 1" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="12" cy="12" r="8" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );

  const IconFrame = ({ className = 'w-4 h-4' }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect x="3" y="5" width="18" height="14" rx="2" stroke="currentColor" strokeWidth="1.4" />
      <path d="M8 9v6M12 9v6M16 9v6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );

  const IconPercent = ({ className = 'w-4 h-4' }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M19 5L5 19" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="6.5" cy="6.5" r="1.5" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="17.5" cy="17.5" r="1.5" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );

  // ---------- Render ----------
  return (
    <>
      {/* Verdict Card */}
      <section className={`text-slate-100 rounded-2xl p-6 shadow-lg w-full border ${verdictBorder} ${verdictBg} transition-colors duration-500`}>
        <div className="text-lg font-semibold tracking-wider text-white mb-4">Deepfake Verdict</div>

        <div className="bg-black/20 border border-white/10 rounded-xl p-4 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <span
              className={`inline-flex items-center justify-center px-6 py-2 rounded-full text-4xl font-bold tracking-wide border ${
                isFake
                  ? 'bg-rose-700/70 text-rose-100 border-rose-700/40'
                  : isReal
                  ? 'bg-emerald-700/70 text-emerald-100 border-emerald-700/40'
                  : 'bg-slate-600/60 text-slate-200 border-slate-500/40'
              }`}
              style={{ minWidth: 110 }}
            >
              {label || 'Unknown'}
            </span>
          </div>

          <div className="flex-1" />

          <div className="flex flex-col items-end gap-3">
            <div className="text-sm text-rose-200">Confidence</div>
            <div className="text-5xl font-bold tabular-nums mt-1" aria-live="polite">
              {displayPct}%
            </div>

            <div className="flex gap-2 mt-1">
              <div className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-black/25 border border-white/10 text-slate-100 text-sm">
                <IconClock className="w-4 h-4" />
                <span className="font-medium ml-1">Duration:</span>
                <span className="ml-2">{formatTime(d.metadata?.duration_sec)}</span>
              </div>

              <div className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-black/25 border border-white/10 text-slate-100 text-sm">
                <IconFrame className="w-4 h-4" />
                <span className="font-medium ml-1">Frames analyzed:</span>
                <span className="ml-2">{d.metadata?.frames_analyzed ?? '—'}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Top Frames */}
      {isFake && (
        <section className="bg-slate-800 text-slate-100 rounded-2xl p-6 shadow-lg w-full mt-6">
          <div className="text-lg font-semibold tracking-wider text-white mb-4">Top Frames</div>

          <div className="mt-2 bg-transparent relative">
            <button onClick={() => scrollByCard(-1)} aria-label="Scroll left" className="hidden md:flex items-center justify-center absolute left-0 top-1/2 -translate-y-1/2 z-20 w-10 h-10 rounded-full bg-slate-900/60 border border-slate-700 text-slate-200 hover:bg-slate-800 transition" style={{ marginLeft: -14 }}>
              ‹
            </button>

            <button onClick={() => scrollByCard(1)} aria-label="Scroll right" className="hidden md:flex items-center justify-center absolute right-0 top-1/2 -translate-y-1/2 z-20 w-10 h-10 rounded-full bg-slate-900/60 border border-slate-700 text-slate-200 hover:bg-slate-800 transition" style={{ marginRight: -14 }}>
              ›
            </button>

            <div ref={scrollerRef} className="flex gap-4 overflow-x-auto pb-2 pl-1 pr-6" style={{ scrollSnapType: 'x mandatory', WebkitOverflowScrolling: 'touch' }}>
              {frames.map((f, i) => {
                const score = f.fake_confidence ?? 0;
                const color = meterColor(score);

                return (
                  <div key={f.url || f.frame_index || i} className="relative bg-slate-700/40 rounded-xl p-3 w-[220px] flex-shrink-0 flex flex-col items-start hover:bg-slate-700/60 transition" style={{ scrollSnapAlign: 'start' }}>
                    {score === maxFake && <div className="absolute top-3 left-3 bg-rose-600/90 text-white text-xs px-2 py-0.5 rounded-full">Most fake</div>}

                    <div className="relative w-full h-32 rounded-lg overflow-hidden mb-3 group" onClick={(e) => { e.stopPropagation(); setModalIndex(i); }}>
                      <FrameThumb
                        url={f.url}
                        alt={`frame-${f.frame_index}`}
                        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                      />

                      <div className="absolute inset-0 bg-slate-900/0 group-hover:bg-slate-900/45 transition-opacity flex flex-col items-center justify-center opacity-0 group-hover:opacity-100 pointer-events-none group-hover:pointer-events-auto">
                        <div className="text-white font-medium text-sm">{pct(score)}% fake</div>
                        <div className="text-slate-300 text-xs mt-1">{formatTime(f.timestamp_sec)}</div>
                      </div>
                    </div>

                    <div className="w-full text-sm flex justify-between items-center">
                      <div className="text-slate-200 font-medium">{pct(score)}% fake</div>
                      <div className="text-xs text-slate-400">{formatTime(f.timestamp_sec)}</div>
                    </div>

                    <div className="mt-2 w-full h-2 bg-slate-700/50 rounded overflow-hidden">
                      <div className="h-full rounded transition-all duration-700" style={{ width: `${pct(score)}%`, background: redGradient, boxShadow: redShadow(color) }} />
                    </div>

                    <div className="mt-2 text-xs text-slate-400">frame #{f.frame_index}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>
      )}

      {/* Modal Viewer (rendered into document.body via portal so it won't be clipped) */}
      {modalIndex != null && frames[modalIndex] && (
        <ModalPortal>
          <div
            className="fixed inset-0 z-[99999] flex items-center justify-center bg-black/70 p-4"
            onClick={() => setModalIndex(null)}
            style={{ overflow: 'visible' }}
          >
            <div className="relative w-[94%] max-w-[1200px] max-h-[92%] rounded-lg overflow-hidden" onClick={(e) => e.stopPropagation()}>
              <button onClick={() => setModalIndex(null)} className="absolute top-3 right-3 z-10 bg-black/50 text-white rounded-full p-2 hover:bg-black/70" aria-label="Close">✕</button>

              <button onClick={() => setModalIndex((idx) => Math.max(0, idx - 1))} className="absolute left-3 top-1/2 -translate-y-1/2 z-10 bg-black/40 text-white rounded-full p-2 hover:bg-black/60" aria-label="Previous">‹</button>

              <button onClick={() => setModalIndex((idx) => Math.min(frames.length - 1, idx + 1))} className="absolute right-3 top-1/2 -translate-y-1/2 z-10 bg-black/40 text-white rounded-full p-2 hover:bg-black/60" aria-label="Next">›</button>

              <div className="flex flex-col bg-slate-800">
                <div className="p-4 flex-1 flex items-center justify-center bg-black">
                  <FrameThumb
                    url={frames[modalIndex].url}
                    alt={`frame-large-${frames[modalIndex].frame_index}`}
                    className="block w-full h-[70vh] object-contain bg-black max-h-[70vh]"
                    onClick={() => {
                      // clicking big image shouldn't close modal — just log
                      // eslint-disable-next-line no-console
                    }}
                  />
                </div>

                {/* Footer */}
                <div className="bg-slate-900/80 p-4 border-t border-white/5 flex flex-col gap-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-black/20 border border-white/5 text-slate-100 text-sm">
                        <IconPercent className="w-4 h-4 text-rose-300" />
                        <span className="font-medium">{pct(frames[modalIndex].fake_confidence)}% fake</span>
                      </div>

                      <div className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-black/20 border border-white/5 text-slate-100 text-sm">
                        <IconFrame className="w-4 h-4" />
                        <span className="font-medium">frame #{frames[modalIndex].frame_index}</span>
                      </div>

                      <div className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-black/20 border border-white/5 text-slate-100 text-sm">
                        <IconClock className="w-4 h-4" />
                        <span className="font-medium">{formatTime(frames[modalIndex].timestamp_sec)}</span>
                      </div>
                    </div>
                  </div>

                  {/* Progress bar container */}
                  <div className="w-full">
                    <div className="w-full h-2 bg-slate-700/40 rounded overflow-hidden">
                      <div aria-hidden className="h-full rounded" style={{ width: `${modalBarPct}%`, background: redGradient, boxShadow: '0 0 8px rgba(244,63,94,0.2)', transition: 'width 120ms linear' }} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </ModalPortal>
      )}
    </>
  );
}
