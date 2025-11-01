import React, { useMemo, useState, useEffect, useRef } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';

import mockData from '../mock/mockData';

// try static import first, fallback to require at runtime
let eddiewoo;
try {
  // bundlers will resolve this; if undefined will throw in some setups
 
  eddiewoo = require('../mock/eddiewoo.mp4');
} catch (e) {
  // if require fails, leave undefined — we'll handle it in component
  eddiewoo = undefined;
}

/* ------------------ Helpers ------------------ */
const formatTimeTick = (s) => {
  if (s == null || isNaN(s)) return '';
  const sec = Math.floor(s);
  if (sec < 60) return `${sec}s`;
  const mm = Math.floor(sec / 60);
  const ss = String(sec % 60).padStart(2, '0');
  return `${mm}:${ss}`;
};
const formatTimeRange = (start, end) => `${formatTimeTick(start)} — ${formatTimeTick(end)}`;

const downloadCSV = (filename, rows) => {
  const header = [
    'index',
    'start',
    'end',
    'dominant_emotion',
    'dominant_score',
    'other_emotion_1',
    'other_score_1',
    'other_emotion_2',
    'other_score_2',
    'confidence',
    'text',
  ];
  const csv = [
    header.join(','),
    ...rows.map((r) =>
      [
        r.index,
        r.start,
        r.end,
        r.dominant,
        r.dominant_score,
        r.other1?.emotion || '',
        r.other1?.score ?? '',
        r.other2?.emotion || '',
        r.other2?.score ?? '',
        r.confidence,
        `"${(r.text || '').replace(/"/g, '""')}"`,
      ].join(',')
    ),
  ].join('\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

/* ------------------ Line Graph ------------------ */
function EmotionLineGraph({ linegraphData, colorMap }) {
  const defaultShape = useMemo(() => ({ video: '', top_emotions: [], data: [] }), []);
  const payload = linegraphData || defaultShape;
  const data = useMemo(() => payload.data || [], [payload]);
  const topEmotions = useMemo(() => payload.top_emotions || [], [payload]);

  const [isAnimating, setIsAnimating] = useState(true);
  const BASE_DURATION = 900;
  const STAGGER = 160;

  useEffect(() => {
    const total = BASE_DURATION + STAGGER * Math.max(0, topEmotions.length - 1);
    const t = setTimeout(() => setIsAnimating(false), total + 80);
    return () => clearTimeout(t);
  }, [payload, topEmotions.length]);

  return (
    <div className="w-full">
      <div className="bg-slate-800 rounded-lg p-4 shadow-sm border border-slate-700">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-white text-lg font-semibold">Text Emotion Timeline</h4>
          <span className="text-slate-400 text-sm">{payload.video || ''}</span>
        </div>

        <div style={{ height: 320 }} className="w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 6 }}>
              <CartesianGrid stroke="#0b1220" strokeOpacity={0.12} />
              <XAxis
                dataKey="time"
                tickFormatter={formatTimeTick}
                axisLine={false}
                tick={{ fill: '#9CA3AF', fontSize: 12 }}
                domain={[data.length ? data[0]?.time : 'dataMin', 'dataMax']}
                type="number"
              />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                tick={{ fill: '#9CA3AF', fontSize: 12 }}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8 }}
                labelFormatter={(label) => `Time: ${formatTimeTick(label)}`}
                formatter={(value, name) => [Number(value).toFixed(3), name]}
                cursor={{ stroke: '#1f2937' }}
                wrapperStyle={{ pointerEvents: isAnimating ? 'none' : 'auto' }}
              />

              {topEmotions.map((emo, idx) => (
                <Line
                  key={emo}
                  type="monotone"
                  dataKey={emo}
                  stroke={colorMap[emo] || '#94a3b8'}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={true}
                  animationDuration={BASE_DURATION}
                  animationBegin={idx * STAGGER}
                  opacity={1}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

/* ------------------ Table ------------------ */
function SegmentsTable({ segments, onSeek }) {
  const [query, setQuery] = useState('');
  const [sortKey, setSortKey] = useState('start');
  const [sortDir, setSortDir] = useState('asc');
  const [expanded, setExpanded] = useState(null);

  const colorMap = {
    neutral: '#93C5FD',
    admiration: '#FDE68A',
    annoyance: '#FCA5A5',
    approval: '#60A5FA',
    love: '#FB7185',
    curiosity: '#C7F9CC',
  };

  const rows = useMemo(() => {
    return (segments || []).map((s, i) => {
      const paired = (s.emotions || []).map((emo, idx) => ({
        emotion: emo,
        score: Number(s.intensities?.[idx] ?? 0),
      }));
      paired.sort((a, b) => b.score - a.score);
      const dominant = paired[0] || { emotion: null, score: 0 };
      const other1 = paired[1] || null;
      const other2 = paired[2] || null;
      return {
        index: i + 1,
        start: s.start ?? 0,
        end: s.end ?? (s.start ? s.start + 1 : 1),
        timeLabel: formatTimeRange(s.start ?? 0, s.end ?? (s.start ? s.start + 1 : 1)),
        text: s.text ?? '',
        dominant: dominant.emotion,
        dominant_score: dominant.score,
        other1,
        other2,
        confidence: Number(s.confidence ?? dominant.score ?? 0),
      };
    });
  }, [segments]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = rows.filter((r) => {
      if (!q) return true;
      return (
        (r.text || '').toLowerCase().includes(q) ||
        (r.dominant || '').toLowerCase().includes(q) ||
        (r.other1?.emotion || '').toLowerCase().includes(q)
      );
    });
    list.sort((a, b) => {
      const dir = sortDir === 'asc' ? 1 : -1;
      if (sortKey === 'start') return dir * (a.start - b.start);
      if (sortKey === 'dominant') return dir * ((a.dominant_score || 0) - (b.dominant_score || 0));
      if (sortKey === 'confidence') return dir * ((a.confidence || 0) - (b.confidence || 0));
      return 0;
    });
    return list;
  }, [rows, query, sortKey, sortDir]);

  return (
    <div className="mt-4 bg-slate-800 rounded-lg p-3 border border-slate-700">
      <div className="flex items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <input
            className="px-2 py-1 rounded bg-slate-900 text-slate-200 text-sm border border-slate-700"
            placeholder="Search transcript or emotion..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            className="px-3 py-1 text-sm rounded bg-slate-700 text-slate-200 hover:bg-slate-600"
            onClick={() => downloadCSV(`${mockData.video || 'data'}_segments.csv`, filtered)}
          >
            Export CSV
          </button>
        </div>

        <div className="flex items-center gap-2 text-sm text-slate-300">
          <label className="text-slate-400 mr-1">Sort:</label>
          <select
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value)}
            className="bg-slate-900 text-slate-200 px-2 py-1 rounded border border-slate-700"
          >
            <option value="start">Time</option>
            <option value="dominant">Dominant intensity</option>
            <option value="confidence">Confidence</option>
          </select>
          <button
            onClick={() => setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))}
            className="px-2 py-1 rounded bg-slate-700 text-slate-200"
            aria-label="Toggle sort direction"
          >
            {sortDir === 'asc' ? '▲' : '▼'}
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-slate-800">
            <tr className="text-slate-300 text-left">
              <th className="p-2 w-8">#</th>
              <th className="p-2 w-40">Time</th>
              <th className="p-2 w-36">Dominant</th>
              <th className="p-2 w-28">Intensity</th>
              <th className="p-2">Other</th>
              <th className="p-2 w-24">Confidence</th>
              <th className="p-2">Text</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr
                key={r.index}
                className="border-t border-slate-700 hover:bg-slate-900 transition-colors"
                onDoubleClick={() => onSeek?.(r.start)}
              >
                <td className="p-2 text-slate-300">{r.index}</td>
                <td className="p-2 text-slate-300">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => onSeek?.(r.start)}
                      className="px-2 py-1 bg-slate-700 text-slate-200 rounded text-xs"
                      aria-label={`Seek to ${formatTimeTick(r.start)}`}
                    >
                      ▶
                    </button>
                    <div className="text-xs text-slate-400">{r.timeLabel}</div>
                  </div>
                </td>

                <td className="p-2">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-3 h-3 rounded-full inline-block"
                      style={{ background: colorMap[r.dominant] || '#94a3b8' }}
                    />
                    <span className="text-slate-200">{r.dominant || '-'}</span>
                  </div>
                </td>

                <td className="p-2">
                  <div className="flex items-center gap-2">
                    <div className="text-slate-200">{((r.dominant_score || 0) * 100).toFixed(2)}%</div>
                    <div className="flex-1 bg-slate-900 h-2 rounded overflow-hidden" style={{ width: 80 }}>
                      <div
                        style={{
                          width: `${Math.min(100, (r.dominant_score || 0) * 100)}%`,
                          background: colorMap[r.dominant] || '#94a3b8',
                          height: '100%',
                        }}
                      />
                    </div>
                  </div>
                </td>

                <td className="p-2 text-slate-200">
                  {r.other1 ? `${r.other1.emotion} (${(r.other1.score * 100).toFixed(2)}%)` : '-'}
                  {r.other2 ? `, ${r.other2.emotion} (${(r.other2.score * 100).toFixed(2)}%)` : ''}
                </td>

                <td className="p-2 text-slate-200">{((r.confidence || 0) * 100).toFixed(2)}%</td>

                <td className="p-2 text-slate-200">
                  <div className="max-w-xl">
                    {expanded === r.index ? (
                      <>
                        <div className="whitespace-pre-wrap">{r.text}</div>
                        <button className="mt-2 text-xs text-slate-400" onClick={() => setExpanded(null)}>
                          collapse
                        </button>
                      </>
                    ) : (
                      <>
                        <div>{r.text?.length > 100 ? `${r.text.slice(0, 100)}...` : r.text}</div>
                        {r.text?.length > 100 && (
                          <button className="mt-1 text-xs text-slate-400" onClick={() => setExpanded(r.index)}>
                            view full
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={7} className="p-4 text-center text-slate-400">
                  No segments match your filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ------------------ EmotionCard (full) ------------------ */
export default function EmotionCard() {
  const tabs = [
    { id: 'audio', label: 'Audio Analysis' },
    { id: 'video', label: 'Video Analysis' },
    { id: 'text', label: 'Text Analysis' },
  ];
  const [active, setActive] = useState('audio');

  // player ref & autoplay toggle (default: seek-only)
  const videoRef = useRef(null);
  const [autoplayOnSeek, setAutoplayOnSeek] = useState(false);
  const [videoStatus, setVideoStatus] = useState({ loaded: false, error: null, src: eddiewoo || null });

  useEffect(() => {
    // if require didn't resolve at import time, try dynamic require (some bundlers)
    if (!videoStatus.src) {
      try {
        // eslint-disable-next-line global-require
        const v = require('../mock/eddiewoo.mp4');
        setVideoStatus((s) => ({ ...s, src: v }));
      } catch (e) {
        // leave src null; UI will show error
        console.error('eddiewoo import failed:', e);
      }
    }
   
  }, []);

  // linegraph (prefer linegraph payload, fallback to timeline)
  const linegraph = useMemo(
    () =>
      mockData?.emotion?.linegraph ||
      { video: mockData.video || '', top_emotions: [], data: mockData?.emotion?.timeline || [] },
    [mockData]
  );

  // robust segments builder (prefer detailed segments)
  const segments = useMemo(() => {
    if (mockData?.segments && Array.isArray(mockData.segments) && mockData.segments.length) return mockData.segments;
    if (mockData?.emotion?.segments && Array.isArray(mockData.emotion.segments) && mockData.emotion.segments.length)
      return mockData.emotion.segments;

    const lg = mockData?.emotion?.linegraph?.data || mockData?.emotion?.timeline || [];
    const topEmotions = mockData?.emotion?.linegraph?.top_emotions || [];

    if (!Array.isArray(lg) || lg.length === 0) return [];

    return lg.map((pt, i) => {
      const start = Number(pt.time ?? 0);
      const next = lg[i + 1];
      const end = next ? Number(next.time ?? start + 1) : start + 1;

      const emotions = [];
      const intensities = [];

      if (topEmotions.length) {
        topEmotions.forEach((emo) => {
          emotions.push(emo);
          intensities.push(Number(pt[emo] ?? 0));
        });
      } else {
        Object.keys(pt).forEach((k) => {
          if (k === 'time') return;
          const v = Number(pt[k]);
          if (!isNaN(v)) {
            emotions.push(k);
            intensities.push(v);
          }
        });
      }

      const confidence = typeof pt.confidence === 'number' ? pt.confidence : Math.max(...(intensities.length ? intensities : [0]));

      return {
        start,
        end,
        text: pt.text ?? '',
        emotions,
        intensities,
        confidence,
      };
    });
  }, [mockData]);

  const colorMap = {
    neutral: '#93C5FD',
    admiration: '#FDE68A',
    annoyance: '#FCA5A5',
    approval: '#60A5FA',
    love: '#FB7185',
    curiosity: '#C7F9CC',
  };

  // seek handler: waits for metadata if necessary
  const handleSeek = (t) => {
    if (!videoRef.current) {
      console.warn('No videoRef available for seek.');
      return;
    }
    const player = videoRef.current;

    const doSeek = () => {
      const safeT = Math.max(0, Math.min(t, player.duration || t));
      player.currentTime = safeT;
      if (autoplayOnSeek) {
        player
          .play()
          .then(() => {
            /* playing */
          })
          .catch((err) => {
            console.warn('Play prevented:', err);
          });
      }
    };

    if (typeof player.duration === 'number' && !isNaN(player.duration) && player.duration > 0) {
      doSeek();
    } else {
      // wait for loadedmetadata then seek once
      const onMeta = () => {
        doSeek();
        player.removeEventListener('loadedmetadata', onMeta);
      };
      player.addEventListener('loadedmetadata', onMeta);
    }
  };

  // video handlers for diagnostics & UI feedback
  const onVideoLoaded = () => {
    setVideoStatus((s) => ({ ...s, loaded: true, error: null }));
    // log quick diagnostics
    try {
      const player = videoRef.current;
      // eslint-disable-next-line no-console
      console.log('video ready, duration:', player.duration, 'src:', player.currentSrc || player.src);
    } catch (e) {
      // noop
    }
  };

  const onVideoError = (e) => {
    console.error('Video error event:', e);
    setVideoStatus((s) => ({ ...s, error: 'Failed to load video — check path, server, or codec.' }));
  };

  return (
    <div>
      {/* Video card above analysis */}
      <div className="bg-slate-800 rounded-xl p-4 mb-4 shadow-sm border border-slate-700">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-white font-semibold">Video: {mockData?.video || 'sample'}</div>
            <div className="text-slate-400 text-sm">Filename: eddiewoo.mp4 (local mock)</div>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-slate-300 text-sm flex items-center gap-2">
              <input
                type="checkbox"
                checked={autoplayOnSeek}
                onChange={(e) => setAutoplayOnSeek(e.target.checked)}
                className="accent-slate-500"
              />
              <span className="text-slate-400 text-sm">Auto-play on seek</span>
            </label>
          </div>
        </div>

        <div className="w-full bg-black rounded overflow-hidden relative">
          {!videoStatus.src && (
            <div className="p-6 text-center text-slate-300">
              Video not found in <code>src/mock/eddiewoo.mp4</code>. Put the file in <code>src/mock/</code> or use
              <code>public/</code> and reference <code>/eddiewoo.mp4</code>.
            </div>
          )}

          {videoStatus.error && (
            <div className="p-4 text-center text-amber-300 bg-slate-900">
              {videoStatus.error}
              <div className="text-xs text-slate-400 mt-2">Open DevTools → Network/Console for diagnostics.</div>
            </div>
          )}

          {videoStatus.src && (
            <>
              <video
                ref={videoRef}
                src={videoStatus.src}
                controls
                className="w-full"
                preload="metadata"
                onLoadedMetadata={onVideoLoaded}
                onError={onVideoError}
              />

              {/* overlay play button — provides an explicit user gesture if native controls don't register */}
              <button
                onClick={() => {
                  if (!videoRef.current) return;
                  // explicit user gesture: play (some browsers require gesture)
                  videoRef.current
                    .play()
                    .then(() => {})
                    .catch(() => {
                      // if play fails, try seeking to 0 then play on user click
                      try {
                        videoRef.current.currentTime = 0;
                        videoRef.current.play().catch(() => {});
                      } catch (e) {
                        // ignore
                      }
                    });
                }}
                className="absolute left-4 bottom-4 bg-slate-700 hover:bg-slate-600 text-white px-3 py-1 rounded text-sm"
                aria-label="Play video"
              >
                Play
              </button>
            </>
          )}
        </div>
      </div>

      {/* Analysis card */}
      <section className="bg-slate-800 rounded-xl p-6 w-full shadow-lg border border-slate-700">
        <div className="flex flex-wrap items-center justify-between mb-4 gap-3">
          <h3 className="text-xl font-semibold text-white">Model-wise Analysis</h3>

          <div className="flex gap-3 flex-wrap">
            <button
              onClick={() => setActive('audio')}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-150 focus:outline-none ${
                active === 'audio' ? 'bg-white text-slate-900 shadow' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              Audio Analysis
            </button>
            <button
              onClick={() => setActive('video')}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-150 focus:outline-none ${
                active === 'video' ? 'bg-white text-slate-900 shadow' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              Video Analysis
            </button>
            <button
              onClick={() => setActive('text')}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-150 focus:outline-none ${
                active === 'text' ? 'bg-white text-slate-900 shadow' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              Text Analysis
            </button>
          </div>
        </div>

        <div className="bg-slate-900 rounded-lg p-6 text-slate-200 min-h-[420px] md:min-h-[480px]">
          {active === 'audio' && <div className="text-slate-400 text-sm">Audio analysis results will appear here.</div>}

          {active === 'video' && <div className="text-slate-400 text-sm">Video analysis results will appear here.</div>}

          {active === 'text' && (
            <>
              <EmotionLineGraph linegraphData={linegraph} colorMap={colorMap} />
              <SegmentsTable segments={segments} onSeek={handleSeek} />
            </>
          )}
        </div>
      </section>
    </div>
  );
}
