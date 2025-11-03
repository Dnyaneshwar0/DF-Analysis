import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';

import mockData from '../mock/mockData';

// Component: EmotionIntensityChart
// - Reads mockData.emotionIntensities
// - Builds a recharts-friendly timeseries (one object per timestamp)
// - Renders a responsive line chart for the listed emotions

export default function EmotionIntensityChart({ height = 260 }) {
  const chartData = useMemo(() => {
    const ei = mockData.emotionIntensities;
    if (!ei || !ei.frames) return [];

    const times = ei.frames.time || [];
    const ints = ei.frames.intensities || [];
    const emotions = ei.emotions || [];

    // Build array of { time: <num>, emotion1: <val>, emotion2: <val>, ... }
    const out = times.map((t, i) => {
      const row = { time: Number(t) };
      const vals = ints[i] || [];
      emotions.forEach((e, k) => {
        // sanitize and clamp 0..1
        let v = vals[k] == null ? 0 : Number(vals[k]);
        if (Number.isNaN(v)) v = 0;
        row[e] = Math.max(0, Math.min(1, v));
      });
      return row;
    });

    return out;
  }, []);

  if (!chartData || chartData.length === 0)
    return (
      <div className="p-4 bg-slate-800 rounded-lg">
        <p className="text-sm text-slate-300">No emotion intensity data available.</p>
      </div>
    );

  // explicit ticks for X axis sampled from the time values (keeps things readable)
  const xTicks = chartData.map((d) => d.time);

  // map emotion -> friendly label (optional) or just use the keys
  const emotions = mockData.emotionIntensities.emotions || [];

  // visually distinct strokes (feel free to change in calling page)
  const strokes = ['#60a5fa', '#fca5a5', '#a78bfa', '#34d399', '#fbbf24'];

  return (
    <section className="bg-slate-900 p-4 rounded-lg shadow-sm">
      <h4 className="text-md font-medium text-slate-100 mb-3">Intensity Variation</h4>

      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <LineChart data={chartData} margin={{ top: 6, right: 12, left: 0, bottom: 6 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#0f172a" />

            <XAxis
              dataKey="time"
              ticks={xTicks}
              tick={{ fill: '#cbd5e1', fontSize: 12 }}
              label={{ value: mockData.emotionIntensities.axes.x_label || 'Time (s)', position: 'insideBottom', fill: '#94a3b8', offset: -4 }}
            />

            <YAxis
              domain={[0, 1]}
              tick={{ fill: '#cbd5e1', fontSize: 12 }}
              label={{ value: mockData.emotionIntensities.axes.y_label || 'Intensity', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
            />

            <Tooltip
              contentStyle={{ background: '#0b1220', border: '1px solid #111827', color: '#e2e8f0' }}
              labelFormatter={(label) => `Time: ${label}s`}
              formatter={(value, name) => [Number(value).toFixed(3), name]}
            />

            <Legend wrapperStyle={{ color: '#cbd5e1' }} />

            {emotions.map((emo, idx) => (
              <Line
                key={emo}
                type="monotone"
                dataKey={emo}
                stroke={strokes[idx % strokes.length]}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs text-slate-400 mt-3">Chart built from <span className="font-medium">mockData.emotionIntensities</span>.</p>
    </section>
  );
}
