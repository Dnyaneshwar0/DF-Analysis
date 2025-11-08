import React, { useMemo, useState, useEffect, useRef, Fragment } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Label,
  Area,
  ComposedChart,
  PieChart,
  Pie,
  Cell,
  ReferenceArea,
} from 'recharts';

/**
 * EmotionCard.jsx
 * - Uses live `data` prop (passed from ResultsPage: <EmotionCard data={data.emotion} />)
 * - Defensive: handles missing fields gracefully (empty charts / placeholders)
 * - Does NOT import or use heavy mock data
 */

// ---------------- STYLE CONSTANTS ----------------
const UI = {
  gridStroke: '#0b1220',
  gridOpacity: 0.15,
  axisTick: '#9CA3AF',
  axisTick2: '#CBD5E1',
  axisLabel: '#94a3b8',
  panelBg: '#0f172a',
  panelText: '#e6eef8',
  border: '#334155',
};

const TEXT_COLOR_MAP = Object.freeze({
  neutral: '#60A5FA',
  admiration: '#FBBF24',
  annoyance: '#F87171',
  approval: '#34D399',
  love: '#FB7185',
  curiosity: '#A78BFA',
  fearful: '#F43F5E',
  surprised: '#38BDF8',
  sad: '#94A3B8',
});

const VIDEO_COLOR_MAP = Object.freeze({
  neutral: '#5EEAD4',
  sad: '#3B82F6',
  happy: '#FACC15',
  angry: '#EF4444',
  surprise: '#06B6D4',
  uncertain: '#A855F7',
});

const AUDIO_COLOR_MAP = Object.freeze({
  neutral: '#64748B',
  calm: '#22C55E',
  happy: '#F59E0B',
  sad: '#3B82F6',
  angry: '#EF4444',
  fearful: '#A855F7',
  disgust: '#10B981',
  surprised: '#06B6D4',
});

// ---------------- HELPERS ----------------
const clamp01 = (x) => (Number.isFinite(x) ? Math.min(1, Math.max(0, x)) : 0);

const formatTimeTick = (s) => {
  if (s == null || Number.isNaN(Number(s))) return '';
  const sec = Math.floor(Number(s));
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
      ].join(','),
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

// ---------------- TOOLTIP COMPONENTS ----------------
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  const seen = new Set();
  const items = [];

  for (const p of payload) {
    if (!p) continue;
    const rawKey = (p.dataKey ?? p.name ?? '').toString();
    const key = rawKey.toLowerCase().trim();
    if (!key) continue;
    if (seen.has(key)) continue;
    seen.add(key);
    const display = p.name ?? rawKey;
    items.push({ key, label: display, value: p.value, color: p.color ?? p.stroke ?? '#94a3b8' });
  }

  if (items.length === 0) return null;

  return (
    <div style={{ background: UI.panelBg, color: UI.panelText, padding: 10, borderRadius: 8, border: `1px solid ${UI.border}` }}>
      <div style={{ fontSize: 12, marginBottom: 6 }}>{`Time: ${formatTimeTick(Number(label))}`}</div>
      {items.map((it) => (
        <div key={it.key} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginTop: 6 }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div style={{ width: 10, height: 10, background: it.color, borderRadius: 3 }} />
            <div>{it.label}</div>
          </div>
          <div>{`${(Number(it.value || 0) * 100).toFixed(2)}%`}</div>
        </div>
      ))}
    </div>
  );
}

function WaveformTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const first = payload.find(Boolean);
  const val = Number(first?.value ?? 0);

  return (
    <div style={{ background: UI.panelBg, color: UI.panelText, padding: 10, borderRadius: 8, border: `1px solid ${UI.border}` }}>
      <div style={{ fontSize: 12, marginBottom: 6 }}>{`Time: ${formatTimeTick(Number(label))}`}</div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <div style={{ width: 10, height: 10, background: '#22d3ee', borderRadius: 3 }} />
          <div>amplitude</div>
        </div>
        <div>{val.toFixed(2)}</div>
      </div>
    </div>
  );
}

// ---------------- TEXT: Line Graph ----------------
function EmotionLineGraph({ linegraphData, colorMap }) {
  const defaultShape = useMemo(() => ({ video: '', top_emotions: [], data: [] }), []);
  const payload = linegraphData || defaultShape;
  const rawData = useMemo(() => payload.data || [], [payload]);

  const topEmotions = useMemo(() => {
    const raw = payload.top_emotions || [];
    const seen = new Set();
    const list = [];
    for (const e of raw) {
      if (typeof e !== 'string') continue;
      const key = e.toLowerCase().trim();
      if (seen.has(key)) continue;
      seen.add(key);
      list.push({ key, label: e });
    }
    return list;
  }, [payload]);

  const data = useMemo(() => {
    return (rawData || []).map((pt) => {
      const out = {};
      out.time = Number(pt.time ?? 0);
      if (pt.text != null) out.text = pt.text;
      if (pt.confidence != null) out.confidence = pt.confidence;
      Object.keys(pt || {}).forEach((k) => {
        if (k === 'time' || k === 'text' || k === 'confidence') return;
        const normalizedKey = k.toLowerCase().trim();
        const maybeNum = Number(pt[k]);
        out[normalizedKey] = Number.isNaN(maybeNum) ? pt[k] : maybeNum;
      });
      return out;
    });
  }, [rawData]);

  const initialVis = useMemo(() => {
    const m = {};
    (topEmotions || []).forEach((e) => (m[e.key] = true));
    return m;
  }, [topEmotions]);

  const [visible, setVisible] = useState(initialVis);
  useEffect(() => setVisible(initialVis), [payload.video]); // reset when video changes

  const latestValue = (emoKey) => {
    if (!data || data.length === 0) return 0;
    const last = data[data.length - 1];
    return Number(last[emoKey] ?? 0);
  };

  const toggle = (emoKey) => setVisible((v) => ({ ...v, [emoKey]: !v[emoKey] }));

  const xDomain = useMemo(() => {
    if (!data || data.length === 0) return ['dataMin', 'dataMax'];
    const arr = data.map((d) => Number(d.time ?? 0));
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    return [min, max];
  }, [data]);

  const SERIES = topEmotions.filter((t) => visible[t.key]);

  return (
    <div className="w-full">
      <div className="bg-slate-800 rounded-lg p-4 shadow-sm border border-slate-700">
        <div className="flex items-start justify-between mb-2">
          <h4 className="text-white text-lg font-semibold">Intensity Variation</h4>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-slate-900 px-2 py-1 rounded">
              {topEmotions.map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => toggle(key)}
                  className={`flex items-center gap-2 px-2 py-1 rounded text-xs focus:outline-none ${visible[key] ? 'opacity-100' : 'opacity-40'}`}
                  title={`${label} — click to toggle`}
                >
                  <span className="w-3 h-3 rounded-full inline-block" style={{ background: colorMap[key] || '#94a3b8' }} />
                  <span className="text-slate-200">{label}</span>
                  <span className="text-slate-400">·</span>
                  <span className="text-slate-200" style={{ minWidth: 48, textAlign: 'right' }}>
                    {(latestValue(key) * 100).toFixed(1)}%
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div style={{ height: 320 }} className="w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 18, left: 56, bottom: 28 }}>
              <CartesianGrid stroke={UI.gridStroke} strokeOpacity={UI.gridOpacity} />
              <XAxis
                dataKey="time"
                tickFormatter={formatTimeTick}
                axisLine={false}
                tick={{ fill: UI.axisTick, fontSize: 12 }}
                domain={xDomain}
                type="number"
                allowDecimals={false}
              >
                <Label value="Time (s)" position="insideBottom" offset={-6} fill={UI.axisLabel} />
              </XAxis>
              <YAxis
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                tick={{ fill: UI.axisTick2, fontSize: 12 }}
                axisLine={false}
                width={64}
                tickLine={false}
                type="number"
                allowDecimals
                tickCount={5}
                minTickGap={8}
                yAxisId="main"
              >
                <Label value="Intensity (%)" angle={-90} position="insideLeft" offset={-6} fill={UI.axisLabel} />
              </YAxis>

              <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#1f2937' }} />

              {SERIES.map(({ key, label }, idx) => (
                <Line
                  key={`${key}-${idx}`}
                  type="monotone"
                  dataKey={key}
                  name={label}
                  stroke={TEXT_COLOR_MAP[key] || '#94a3b8'}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive
                  animationDuration={900}
                  animationBegin={idx * 160}
                  activeDot={{ r: 4 }}
                  connectNulls
                  yAxisId="main"
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ---------------- TEXT: Segments Table ----------------
function SegmentsTable({ segments, onSeek }) {
  const [query, setQuery] = useState('');
  const [sortKey, setSortKey] = useState('start');
  const [sortDir, setSortDir] = useState('asc');
  const [expanded, setExpanded] = useState(null);

  const colorMap = TEXT_COLOR_MAP;

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
            onClick={() => downloadCSV(`segments.csv`, filtered)}
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

// ---------------- VIDEO: Timeline ----------------
function VideoEmotionTimelineChart({ bars, colorMap }) {
  const cleaned = useMemo(() => {
    const list = Array.isArray(bars) ? bars : [];
    return list
      .map((b) => ({
        start: Number(b.start ?? 0),
        end: Number(b.end ?? (b.start ?? 0) + 0.01),
        emo: (b.emo || 'neutral').toLowerCase(),
        conf: clamp01(Number(b.conf ?? 0)),
      }))
      .filter((b) => Number.isFinite(b.start) && Number.isFinite(b.end) && b.end > b.start)
      .sort((a, b) => a.start - b.start);
  }, [bars]);

  const xDomain = useMemo(() => {
    if (!cleaned.length) return [0, 1];
    const minStart = cleaned[0].start;
    const maxEnd = Math.max(...cleaned.map((b) => b.end));
    return [minStart, maxEnd];
  }, [cleaned]);

  const baselineData = useMemo(
    () => cleaned.flatMap((b) => [{ start: b.start, base: 0 }, { start: b.end, base: 0 }]),
    [cleaned],
  );

  return (
    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 mb-6">
      <h4 className="text-white text-lg font-semibold mb-3">Emotion Confidence Over Time</h4>

      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <ComposedChart data={baselineData} margin={{ top: 10, right: 20, left: 60, bottom: 30 }}>
            <CartesianGrid stroke={UI.gridStroke} strokeOpacity={UI.gridOpacity} />

            <XAxis
              dataKey="start"
              type="number"
              domain={xDomain}
              tickFormatter={formatTimeTick}
              tick={{ fill: UI.axisTick, fontSize: 12 }}
              label={{ value: 'Time (s)', position: 'insideBottom', offset: -5, fill: UI.axisLabel }}
            />

            <YAxis
              domain={[0, 1]}
              yAxisId="main"
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
              tick={{ fill: UI.axisTick2, fontSize: 12 }}
              label={{ value: 'Confidence', angle: -90, position: 'insideLeft', fill: UI.axisLabel, offset: -35 }}
            />

            <Tooltip
              content={({ active, label }) => {
                if (!active) return null;
                const x = Number(label);
                if (!Number.isFinite(x)) return null;
                const seg = cleaned.find((d) => x >= d.start && x <= d.end) || cleaned[0];
                if (!seg) return null;
                return (
                  <div style={{ background: UI.panelBg, color: UI.panelText, padding: 10, borderRadius: 8, border: `1px solid ${UI.border}` }}>
                    <div style={{ fontSize: 12, marginBottom: 6 }}>{`${formatTimeTick(seg.start)} — ${formatTimeTick(seg.end)}`}</div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span style={{ textTransform: 'capitalize' }}>{seg.emo}</span>
                      <span>{(seg.conf * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                );
              }}
            />

            <Area type="monotone" dataKey="base" stroke="none" fill="none" isAnimationActive={false} yAxisId="main" />

            {cleaned.map((d, i) => (
              <ReferenceArea
                key={`${d.start}-${d.end}-${i}`}
                x1={d.start}
                x2={d.end}
                y1={0}
                y2={Math.max(0.02, d.conf)}
                yAxisId="main"
                fill={colorMap[d.emo] || '#64748B'}
                fillOpacity={Math.max(0.45, Math.min(0.9, d.conf * 0.85 + 0.25))}
                stroke="none"
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2 text-sm">
        {Array.from(new Set(cleaned.map((d) => d.emo))).map((emo) => (
          <div key={emo} className="flex items-center gap-2">
            <span className="inline-block w-3 h-3 rounded" style={{ background: colorMap[emo] || '#64748B' }} />
            <span className="capitalize text-slate-200">{emo}</span>
          </div>
        ))}
      </div>

      {!cleaned.length && <div className="text-sm text-slate-400 mt-2">No video timeline data available.</div>}
    </div>
  );
}

// ---------------- MAIN: EmotionCard (accepts `data`) ----------------
export default function EmotionCard({ data = {} }) {
  const tabs = [
    { id: 'audio', label: 'Audio Analysis' },
    { id: 'video', label: 'Video Analysis' },
    { id: 'text', label: 'Text Analysis' },
  ];

  const [active, setActive] = useState('video');

  // single source of truth for seeking (used by SegmentsTable)
  const videoRef = useRef(null);
  const seekTo = (t) => {
    if (videoRef.current && Number.isFinite(Number(t))) {
      try {
        videoRef.current.currentTime = Number(t);
        videoRef.current.play?.().catch(() => {});
      } catch {
        /* no-op */
      }
    }
  };

  // Harden autoplay: attempt best-effort play
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const tryPlay = () => v.play?.().catch(() => {});
    if (active === 'video') {
      v.addEventListener('canplay', tryPlay, { once: true });
      tryPlay();
    }
    return () => {
      v?.removeEventListener?.('canplay', tryPlay);
    };
  }, [active]);

  // Use provided data (live) — keep defensive defaults
  const actual = (data && Object.keys(data).length > 0) ? data : { models: {} };

  const rafdb = actual?.models?.rafdb ?? {};
  const rav = actual?.models?.ravdess ?? {};
  const go = actual?.models?.goemotions ?? {};

  // Audio / RAVDESS
  const waveform = rav?.waveform ?? { frames: { time: [], envelope: [] }, axes: {} };
  const results = rav?.results ?? { predicted_emotion: '-', probabilities: {} };
  const predicted = results?.predicted_emotion || '-';
  const probEntries = useMemo(() => {
    const probs = results?.probabilities || {};
    return Object.entries(probs).map(([label, value]) => ({
      name: label,
      value: Number(value || 0),
    }));
  }, [results]);

  // Text / GoEmotions
  const linegraph = useMemo(() => go?.linegraph ?? { video: '', top_emotions: [], data: [] }, [go]);

  const segments = useMemo(() => {
    if (Array.isArray(go?.tabledata?.segments) && go.tabledata.segments.length) {
      return go.tabledata.segments;
    }
    const lg = go?.linegraph?.data || [];
    const topEmotions = go?.linegraph?.top_emotions || [];
    if (!lg.length) return [];

    const dedup = Array.from(new Set(topEmotions.map((e) => (e ?? '').toLowerCase().trim()).filter(Boolean)));

    return lg.map((pt, i) => {
      const start = Number(pt.time ?? 0);
      const end = Number(lg[i + 1]?.time ?? start + 1);

      const emotions = (dedup.length ? dedup : Object.keys(pt).filter((k) => k !== 'time'))
        .map((k) => k.toLowerCase().trim());
      const intensities = emotions.map((k) => Number(pt[k] ?? 0));

      const confidence =
        typeof pt.confidence === 'number' ? pt.confidence : intensities.length ? Math.max(...intensities) : 0;

      return { start, end, text: pt.text ?? '', emotions, intensities, confidence };
    });
  }, [go]);

  // Video / RAF-DB
  const videoTimeline = rafdb?.timeline || [];
  const videoDuration = Number(rafdb?.duration_s ?? 0);

  const videoBars = useMemo(() => {
    if (!Array.isArray(videoTimeline) || videoTimeline.length === 0) return [];
    const out = videoTimeline.map((pt, i) => {
      const start = Number(pt.t ?? pt.start ?? 0);
      const endRaw = Number(videoTimeline[i + 1]?.t ?? videoTimeline[i + 1]?.start ?? NaN);
      const end = Number.isFinite(endRaw)
        ? endRaw
        : Number.isFinite(videoDuration) && videoDuration > start
        ? videoDuration
        : start + 0.5;
      return {
        start,
        end,
        width: Math.max(0, end - start),
        emo: pt.emo?.toLowerCase?.() || 'neutral',
        conf: clamp01(Number(pt.conf ?? pt.confidence ?? 0)),
      };
    });
    return out.filter((d) => d.end > d.start).sort((a, b) => a.start - b.start);
  }, [videoTimeline, videoDuration]);

  // attempt to find a playable video URL in the merged JSON (optional)
  // Common patterns: rafdb.annotated_video_url or models.rafdb.annotated_video
  const videoSrc =
    (rafdb && (rafdb.annotated_video_url || rafdb.annotated_video)) ||
    (actual && actual.video_url) || // generic field if backend provides
    null; // no client-side fallback — keep it null to avoid autoplay errors

  return (
    <section className="bg-slate-800 rounded-xl p-6 w-full shadow-lg border border-slate-700">
      <div className="flex flex-wrap items-center justify-between mb-4 gap-3">
        <h3 className="text-xl font-semibold text-white">Model-wise Analysis</h3>

        <div className="flex gap-3 flex-wrap">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setActive(t.id)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-150 focus:outline-none ${
                active === t.id ? 'bg-white text-slate-900 shadow' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-slate-900 rounded-lg p-6 text-slate-200 min-h-[420px] md:min-h-[480px]">
        {active === 'audio' && (
          <div className="rounded-lg bg-slate-800 p-4 shadow border border-slate-700">
            <h4 className="text-cyan-400 text-lg font-semibold mb-3">Acoustic Waveform</h4>

            <div className="text-slate-400 text-sm mb-4">
              Sample Rate: {waveform?.meta?.sr ?? '—'} Hz · Frames: {waveform?.meta?.original_num_frames ?? '—'}
            </div>

            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart
                data={(waveform?.frames?.time || []).map((t, i) => ({
                  time: t,
                  amplitude: waveform?.frames?.envelope?.[i] ?? 0,
                }))}
                margin={{ top: 10, right: 20, left: 60, bottom: 30 }}
              >
                <defs>
                  <linearGradient id="cyanGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.8} />
                    <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
                  </linearGradient>
                </defs>

                <CartesianGrid stroke={UI.gridStroke} strokeOpacity={UI.gridOpacity} />

                <XAxis
                  dataKey="time"
                  tick={{ fill: UI.axisTick, fontSize: 12 }}
                  tickFormatter={formatTimeTick}
                  interval="preserveStartEnd"
                  label={{ value: waveform?.axes?.x_label || 'Time (s)', position: 'insideBottom', offset: -5, fill: UI.axisLabel }}
                />

                <Tooltip content={<WaveformTooltip />} />

                <YAxis
                  tickFormatter={(v) => v.toFixed(2)}
                  tick={{ fill: UI.axisTick2, fontSize: 12 }}
                  label={{
                    value: waveform?.axes?.y_label || 'Normalized Amplitude',
                    angle: -90,
                    position: 'insideLeft',
                    fill: UI.axisLabel,
                    style: { textAnchor: 'middle' },
                    offset: -35,
                  }}
                  domain={[0, 1]}
                />

                <Area type="monotone" dataKey="amplitude" stroke="none" fill="url(#cyanGradient)" fillOpacity={0.6} isAnimationActive animationDuration={1200} />
                <Line type="monotone" dataKey="amplitude" stroke="#22d3ee" strokeWidth={2.5} dot={false} isAnimationActive animationDuration={1200} />
              </ComposedChart>
            </ResponsiveContainer>

            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
              <div className="md:col-span-1">
                <div className="bg-slate-900 border border-slate-700 rounded-lg p-4">
                  <div className="text-slate-400 text-sm">Predicted Emotion</div>
                  <div className="mt-1 text-2xl font-bold text-white capitalize">{predicted}</div>
                  <div className="mt-2 text-slate-400 text-sm">
                    File: <span className="text-slate-200">{(results?.file || '').split('/').pop() || '—'}</span>
                  </div>
                </div>
              </div>

              <div className="md:col-span-2">
                <div className="bg-slate-900 border border-slate-700 rounded-lg p-4">
                  <div className="text-slate-200 font-semibold mb-2">Class Probabilities</div>
                  <div style={{ width: '100%', height: 260 }}>
                    <ResponsiveContainer>
                      <PieChart>
                        <Pie data={probEntries} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={2} isAnimationActive>
                          {probEntries.map((entry, idx) => (
                            <Cell key={`cell-${idx}`} fill={(AUDIO_COLOR_MAP[entry.name?.toLowerCase?.()] || '#94A3B8')} />
                          ))}
                        </Pie>
                        <Tooltip
                          formatter={(v, name) => [`${(Number(v) * 100).toFixed(2)}%`, name]}
                          contentStyle={{ background: 'rgba(0,0,0,0)', border: 'none', boxShadow: 'none' }}
                          labelStyle={{ display: 'none' }}
                          itemStyle={{ color: '#ffffff', fontWeight: 500, fontSize: '13px', textShadow: '0px 0px 6px rgba(0,0,0,0.6)' }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2 text-sm">
                    {probEntries.map((e) => (
                      <div key={e.name} className="flex items-center gap-2">
                        <span className="inline-block w-3 h-3 rounded" style={{ background: AUDIO_COLOR_MAP[e.name?.toLowerCase?.()] || '#94A3B8' }} />
                        <span className="capitalize text-slate-200">{e.name}</span>
                        <span className="text-slate-400">{(e.value * 100).toFixed(2)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {active === 'video' && (
          <Fragment>
            <div className="mb-4">
              {videoSrc ? (
                <video
                  ref={videoRef}
                  src={videoSrc}
                  controls
                  muted
                  playsInline
                  // do NOT autoPlay if no explicit user gesture in most browsers — best-effort
                  className="w-full h-64 object-contain bg-black"
                />
              ) : (
                <div className="w-full h-64 flex items-center justify-center bg-black text-slate-400">
                  Annotated video not available — expose `models.rafdb.annotated_video_url` from backend to enable playback.
                </div>
              )}
            </div>

            <VideoEmotionTimelineChart bars={videoBars} colorMap={VIDEO_COLOR_MAP} />
          </Fragment>
        )}

        {active === 'text' && (
          <Fragment>
            <EmotionLineGraph linegraphData={linegraph} colorMap={TEXT_COLOR_MAP} />
            <SegmentsTable segments={segments} onSeek={seekTo} />
          </Fragment>
        )}
      </div>
    </section>
  );
}
