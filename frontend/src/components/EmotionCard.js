import React, { useMemo, useState, useEffect } from 'react';
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
  ComposedChart
} from 'recharts';

import mockData from '../mock/mockData';

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

/* ------------------ Custom tooltip (dedupes payload entries by stable key) ------------------ */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  const seen = new Set();
  const items = [];

  for (const p of payload) {
    if (!p) continue;
    // Use dataKey as the stable identifier
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
    <div style={{ background: '#0f172a', color: '#e6eef8', padding: 10, borderRadius: 8, border: '1px solid #334155' }}>
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

/* ------------------ EmotionLineGraph (normalized keys, dedupe, tightened Y-axis spacing) ------------------ */
function EmotionLineGraph({ linegraphData, colorMap }) {
  const defaultShape = useMemo(() => ({ video: '', top_emotions: [], data: [] }), []);
  const payload = linegraphData || defaultShape;
  const rawData = useMemo(() => payload.data || [], [payload]);

  // Normalize top emotions: [{ key, label }]
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

  // Normalize data keys so dataKey matches normalized keys
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
  useEffect(() => setVisible(initialVis), [payload.video]); // reset on new payload

  const [isAnimating, setIsAnimating] = useState(true);
  const BASE_DURATION = 900;
  const STAGGER = 160;
  useEffect(() => {
    const total = BASE_DURATION + STAGGER * Math.max(0, topEmotions.length - 1);
    const t = setTimeout(() => setIsAnimating(false), total + 80);
    return () => clearTimeout(t);
  }, [topEmotions.length]);

  const latestValue = (emoKey) => {
    if (!data || data.length === 0) return 0;
    const last = data[data.length - 1];
    return Number(last[emoKey] ?? 0);
  };

  const toggle = (emoKey) => setVisible((v) => ({ ...v, [emoKey]: !v[emoKey] }));

  const fmtTime = (s) => {
    if (s == null || isNaN(s)) return '';
    const sec = Math.floor(s);
    if (sec < 60) return `${sec}s`;
    return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`;
  };

  // Safe X domain
  const xDomain = useMemo(() => {
    if (!data || data.length === 0) return ['dataMin', 'dataMax'];
    const arr = data.map((d) => Number(d.time ?? 0));
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    return [min, max];
  }, [data]);

  return (
    <div className="w-full">
      <div className="bg-slate-800 rounded-lg p-4 shadow-sm border border-slate-700">
        <div className="flex items-start justify-between mb-2">
          <div>
            <h4 className="text-white text-lg font-semibold">Intensity Variation</h4>
            {/* subtitle removed intentionally */}
          </div>

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
            <LineChart
              data={data}
              // tightened left margin so Y axis uses less horizontal space
              margin={{ top: 10, right: 18, left: 56, bottom: 28 }}
            >
              <CartesianGrid stroke="#0b1220" strokeOpacity={0.12} />

              <XAxis
                dataKey="time"
                tickFormatter={fmtTime}
                axisLine={false}
                tick={{ fill: '#9CA3AF', fontSize: 12 }}
                domain={xDomain}
                type="number"
                allowDecimals={false}
              >
                <Label value="Time (s)" position="insideBottom" offset={-6} fill="#94a3b8" />
              </XAxis>

              {/* tighter Y axis width to reduce wasted space */}
              <YAxis
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                tick={{ fill: '#CBD5E1', fontSize: 12 }}
                axisLine={false}
                width={64}
                tickLine={false}
                type="number"
                allowDecimals={true}
                allowDataOverflow={false}
                tickCount={5}
                minTickGap={8}
                yAxisId="main"
              >
                <Label value="Intensity (%)" angle={-90} position="insideLeft" offset={-6} fill="#94a3b8" />
              </YAxis>

              <Tooltip
                content={<CustomTooltip />}
                cursor={{ stroke: '#1f2937' }}
                formatter={null}
                labelFormatter={null}
              />

              {/* Render lines with normalized dataKey (key) and stable label (name) */}
              {topEmotions.map(({ key, label }, idx) =>
                visible[key] ? (
                  <Line
                    key={`${key}-${idx}`}
                    type="monotone"
                    dataKey={key}
                    name={label}
                    stroke={colorMap[key] || '#94a3b8'}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={isAnimating}
                    animationDuration={BASE_DURATION}
                    animationBegin={idx * STAGGER}
                    activeDot={{ r: 4 }}
                    connectNulls
                    yAxisId="main"
                  />
                ) : null
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

/* ------------------ SegmentsTable (unchanged) ------------------ */
function SegmentsTable({ segments, onSeek }) {
  const [query, setQuery] = useState('');
  const [sortKey, setSortKey] = useState('start');
  const [sortDir, setSortDir] = useState('asc');
  const [expanded, setExpanded] = useState(null);

  const colorMap = {
    neutral: '#60A5FA',     // soft cyan-blue
    admiration: '#FBBF24',  // golden amber
    annoyance: '#F87171',   // coral red
    approval: '#34D399',    // mint green
    love: '#FB7185',        // rose pink
    curiosity: '#A78BFA',   // violet
    fearful: '#F43F5E',     // vivid magenta-red
    surprised: '#38BDF8',   // bright sky blue
    sad: '#37dfb2ff',         // muted blue-gray
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

/* ------------------ EmotionCard (final) ------------------ */
export default function EmotionCard() {
  const tabs = [
    { id: 'audio', label: 'Audio Analysis' },
    { id: 'video', label: 'Video Analysis' },
    { id: 'text', label: 'Text Analysis' },
  ];
  const [active, setActive] = useState('audio');

  const linegraph = useMemo(
    () =>
      mockData?.emotion?.linegraph ||
      { video: mockData.video || '', top_emotions: [], data: mockData?.emotion?.timeline || [] },
    []
  );

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
        // use deduped topEmotions in same order
        const dedup = Array.from(new Set(topEmotions));
        dedup.forEach((emo) => {
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
  }, []);

  const colorMap = {
    neutral: '#60A5FA',     // soft cyan-blue
    admiration: '#FBBF24',  // golden amber
    annoyance: '#F87171',   // coral red
    approval: '#34D399',    // mint green
    love: '#FB7185',        // rose pink
    curiosity: '#A78BFA',   // violet
    fearful: '#F43F5E',     // vivid magenta-red
    surprised: '#38BDF8',   // bright sky blue
    sad: '#94A3B8',         // muted blue-gray
  };

  const waveform = mockData?.audioWaveform;

  // --- convert emotionIntensities -> linegraph shape ----
  const convertEmotionIntensities = (ei) => {
    if (!ei || !ei.frames) return { video: '', top_emotions: [], data: [] };
    const times = ei.frames.time || [];
    const ints = ei.frames.intensities || [];
    const emotions = ei.emotions || [];

    // build `data` array: { time: <num>, emo1: <val>, emo2: <val>, ... }
    const data = times.map((t, i) => {
      const row = { time: Number(t) };
      const vals = ints[i] || [];
      emotions.forEach((e, k) => {
        row[e.toLowerCase().trim()] = Number(vals[k] ?? 0);
      });
      return row;
    });

    // keep top_emotions normalized (lowercase keys to match conversion)
    const top_emotions = emotions.map((e) => e.toLowerCase().trim());

    return {
      video: mockData.video || '',
      top_emotions,
      data,
    };
  };

  const emotionIntensitiesLinegraph = useMemo(() => convertEmotionIntensities(mockData.emotionIntensities), []);

  // placeholder seek handler (integration point)
  const handleSeek = (t) => {
    // integrate with your video player: playerRef.current?.seek(t)
    console.log('seek to', t);
  };

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
                Sample Rate: {waveform?.meta?.sr} Hz · Frames: {waveform?.meta?.num_frames}
              </div>

              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart
                  data={(waveform?.frames?.time || []).map((t, i) => ({
                    time: t,
                    amplitude: waveform?.frames?.envelope?.[i] ?? 0,
                  }))}
                  margin={{ top: 10, right: 20, left: 60, bottom: 30 }}
                >
                  {/*  Cyan Gradient Definition */}
                  <defs>
                    <linearGradient id="cyanGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.8} />
                      <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
                    </linearGradient>
                  </defs>

                  <CartesianGrid stroke="#0b1220" strokeOpacity={0.15} />

                  <XAxis
                    dataKey="time"
                    tick={{ fill: '#9CA3AF', fontSize: 12 }}
                    label={{
                      value: waveform?.axes?.x_label || 'Time (s)',
                      position: 'insideBottom',
                      offset: -5,
                      fill: '#94a3b8',
                    }}
                  />

                  <YAxis
                    tickFormatter={(v) => v.toFixed(2)}
                    tick={{ fill: '#CBD5E1', fontSize: 12 }}
                    label={{
                      value: waveform?.axes?.y_label || 'Normalized Amplitude',
                      angle: -90,
                      position: 'insideLeft',
                      fill: '#94a3b8',
                      style: { textAnchor: 'middle' },
                      offset: -35,
                    }}
                    domain={[0, 1]}
                  />

                  <Tooltip
                    formatter={(v) => v.toFixed(2)}
                    labelFormatter={(v) => `${v}s`}
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                    }}
                  />

                  {/* ✅ AREA — Visible Cyan Gradient */}
                  <Area
                    type="monotone"
                    dataKey="amplitude"
                    stroke="none"
                    fill="url(#cyanGradient)"
                    fillOpacity={0.6}
                    isAnimationActive={true}
                    animationDuration={1400}
                    animationBegin={0}
                  />

                  {/* ✅ LINE — Animated Cyan Curve */}
                  <Line
                    type="monotone"
                    dataKey="amplitude"
                    stroke="#22d3ee"
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={true}
                    animationDuration={1400}
                    animationBegin={0}
                  />
                </ComposedChart>
              </ResponsiveContainer>
              <div className="mt-4">
                <EmotionLineGraph linegraphData={emotionIntensitiesLinegraph} colorMap={colorMap} />
              </div>
              
            </div>
          )}

        {active === 'video' && <div className="text-slate-400 text-sm">Video analysis results will appear here.</div>}

        {active === 'text' && (
          <>
            <EmotionLineGraph linegraphData={linegraph} colorMap={colorMap} />
            <SegmentsTable segments={segments} onSeek={handleSeek} />
          </>
        )}
      </div>
    </section>
  );
}
