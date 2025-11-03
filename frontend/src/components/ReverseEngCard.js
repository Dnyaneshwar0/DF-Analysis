// src/components/ReverseEngCard.js
import React, { useMemo, useEffect, useRef, useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  CartesianGrid,
  LabelList,
} from 'recharts';
import mockData from '../mock/mockData';

export default function ReverseEngCard({
  data = mockData.reverseEng,
  animationMs = 900,
  gauge: gaugeProps = {},
}) {
  if (!data) {
    return (
      <section className="bg-slate-800 rounded-xl p-6 w-full shadow-lg border border-slate-700 text-slate-300">
        <h3 className="text-lg font-semibold text-indigo-300 mb-2">Reverse Engineering</h3>
        <div>No reverse-engineering data available.</div>
      </section>
    );
  }

  const { result = {}, df_models_desc = [] } = data;
  const { all_probs = {}, predicted_label = '', confidence = 0 } = result;

  const probs = useMemo(() => {
    return Object.entries(all_probs || {})
      .map(([name, p]) => ({ name, prob: Number(p) }))
      .sort((a, b) => b.prob - a.prob);
  }, [all_probs]);

  const chartData = useMemo(
    () => probs.map((p) => ({ name: p.name, value: +(p.prob * 100).toFixed(2) })),
    [probs]
  );

  const predictedDetails = useMemo(() => {
    if (!predicted_label || !Array.isArray(df_models_desc)) return null;
    return (
      df_models_desc.find(
        (m) => String(m.name).toLowerCase() === String(predicted_label).toLowerCase()
      ) || null
    );
  }, [predicted_label, df_models_desc]);

  // ---------- Gauge styling (kept from original) ----------
  const gradientStops = ['#22d3ee', '#8b5cf6', '#a855f7'];
  const bgStroke = '#0f1724';
  const barAccent = '#7c3aed';

  const {
    width: gw = 300,
    height: gh = 150,
    strokeWidth: gStroke = 14,
    padding: gPad = 10,
    thresholds = [
      { pct: 0.5, color: '#f59e0b' },
      { pct: 0.75, color: '#10b981' },
    ],
  } = gaugeProps;

  const cx = gw / 2;
  const cy = Math.round(gh * 0.68);
  const r = Math.min(cx - gPad - gStroke / 2, cy - gPad - gStroke / 2);
  const startX = cx - r;
  const endX = cx + r;
  const arcPath = `M ${startX} ${cy} A ${r} ${r} 0 0 1 ${endX} ${cy}`;

  const targetRotation = (Number(confidence) * 180) - 90;
  const targetPercent = Math.max(0, Math.min(100, Number(confidence) * 100));

  // gauge animation state
  const [rotation, setRotation] = useState(() => targetRotation);
  const [displayPercent, setDisplayPercent] = useState(() => 0);
  const [arcProgress, setArcProgress] = useState(() => Math.max(0, Math.min(1, Number(confidence))));

  const rafRef = useRef(null);
  const fromRotationRef = useRef(targetRotation);
  const fromPercentRef = useRef(0);
  const fromArcRef = useRef(0);
  const ease = (t) => t * t * (3 - 2 * t);

  useEffect(() => {
    if (!Number.isFinite(targetRotation) || animationMs <= 0) {
      setRotation(targetRotation);
      setDisplayPercent(targetPercent);
      setArcProgress(Math.max(0, Math.min(1, Number(confidence))));
      fromRotationRef.current = targetRotation;
      fromPercentRef.current = targetPercent;
      fromArcRef.current = Math.max(0, Math.min(1, Number(confidence)));
      return;
    }

    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    const duration = Math.max(80, animationMs);
    const fromRot = fromRotationRef.current;
    const fromPct = fromPercentRef.current;
    const fromArc = fromArcRef.current;

    const toRot = targetRotation;
    const toPct = targetPercent;
    const toArc = Math.max(0, Math.min(1, Number(confidence)));

    let start = null;

    const step = (timestamp) => {
      if (!start) start = timestamp;
      const elapsed = timestamp - start;
      const t = Math.min(1, elapsed / duration);
      const e = ease(t);

      setRotation(fromRot + (toRot - fromRot) * e);
      setDisplayPercent(Number((fromPct + (toPct - fromPct) * e).toFixed(2)));
      setArcProgress(fromArc + (toArc - fromArc) * e);

      if (t < 1) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        rafRef.current = null;
        fromRotationRef.current = toRot;
        fromPercentRef.current = toPct;
        fromArcRef.current = toArc;
        setRotation(toRot);
        setDisplayPercent(Number(toPct.toFixed(2)));
        setArcProgress(toArc);
      }
    };

    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [targetRotation, targetPercent, confidence, animationMs]);

  const needleColor = (() => {
    for (let i = thresholds.length - 1; i >= 0; --i) {
      if (confidence >= thresholds[i].pct) return thresholds[i].color;
    }
    return '#ef4444';
  })();

  const needleTopY = cy - r + 8;
  const needleInnerY = cy + Math.max(6, Math.round(gStroke * 0.3));
  const needleWidth = Math.max(3, Math.round(gStroke / 3));
  const pivotRadius = Math.max(6, Math.round(gStroke * 0.55));

  // semicircle path length (used to reveal gradient arc)
  const arcLength = Math.PI * r;
  const visibleDashOffset = arcLength * (1 - Math.max(0, Math.min(1, arcProgress)));

  // ---------- Requested color palette (5 colors) ----------
  const barColors = [
    '#7753a5', // (119,83,165)
    '#7b78c9', // (123,120,201)
    '#69a2ce', // (105,162,206)
    '#6ec5b6', // (110,197,182)
    '#88dda3', // (136,221,163)
  ];

  // helper to build gradient ids
  const gradientId = (i) => `barGrad-${i}`;

  // ---------- Custom tooltip component (polished to match EmotionCard look) ----------
  const CustomTooltip = (props) => {
    const { active, payload } = props;
    if (!active || !payload || !payload.length) return null;

    const item = payload[0];
    const name = item.payload?.name ?? item.name;
    const value = Number(item.value ?? 0);
    const idx = chartData.findIndex((d) => d.name === name);
    const color = barColors[idx >= 0 ? idx % barColors.length : 0];

    const boxStyle = {
      background: 'linear-gradient(180deg, rgba(17,24,39,0.98), rgba(10,14,20,0.95))',
      border: '1px solid rgba(255,255,255,0.04)',
      borderRadius: 10,
      padding: '10px 12px',
      minWidth: 200,
      boxShadow: '0 8px 24px rgba(2,6,23,0.6)',
      color: '#bbc2cbff',
    };

    return (
      <div style={boxStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 10, height: 10, borderRadius: 6, background: color }} />
          <div style={{ fontWeight: 700, color: '#efe9e9ff' }}>{name}</div>
        </div>
        <div style={{ marginTop: 8, fontSize: 18, fontWeight: 600 }}>{value.toFixed(2)}%</div>
        <div style={{ marginTop: 6, fontSize: 12, color: 'rgba(230,238,248,0.7)' }}>probability (confidence)</div>
      </div>
    );
  };

  return (
    <section className="bg-slate-800 rounded-xl p-6 w-full shadow-lg border border-slate-700">
      <header className="mb-3">
        <h3 className="text-2xl md:text-0xl font-semibold text-white mt-1">Attribute Predictions</h3>
      </header>

      <div className="bg-gradient-to-br from-[#0b1220] to-[#0e1622] rounded-lg p-4 border border-slate-700 shadow-sm mb-4">
        <div className="flex flex-col items-center">
          <div className="text-xl font-semibold text-slate-300 mb-5">Confidence Score</div>

          <div style={{ width: gw, height: gh }} className="relative mb-2">
            <svg viewBox={`0 0 ${gw} ${gh}`} className="w-full h-full" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
              <defs>
                <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor={gradientStops[0]} />
                  <stop offset="55%" stopColor={gradientStops[1]} />
                  <stop offset="100%" stopColor={gradientStops[2]} />
                </linearGradient>
              </defs>

              <path d={arcPath} fill="none" stroke={bgStroke} strokeWidth={gStroke} strokeLinecap="round" />

              <path
                d={arcPath}
                fill="none"
                stroke="url(#gaugeGradient)"
                strokeWidth={gStroke}
                strokeLinecap="round"
                strokeDasharray={arcLength}
                strokeDashoffset={visibleDashOffset}
              />

              <g transform={`rotate(${rotation} ${cx} ${cy})`}>
                <line
                  x1={cx}
                  y1={cy + 0.6}
                  x2={cx}
                  y2={needleTopY}
                  stroke={needleColor}
                  strokeWidth={needleWidth}
                  strokeLinecap="round"
                  style={{ filter: 'drop-shadow(0 2px 2px rgba(0,0,0,0.35))' }}
                />
                <line
                  x1={cx}
                  y1={cy + 2}
                  x2={cx}
                  y2={needleInnerY}
                  stroke={needleColor}
                  strokeWidth={Math.max(1, Math.round(needleWidth / 2))}
                  strokeLinecap="round"
                />
                <circle cx={cx} cy={cy} r={pivotRadius} fill={needleColor} />
              </g>
            </svg>
          </div>

          <div className="text-3xl md:text-4xl font-bold text-white mb-1">{displayPercent}%</div>

          <div className="text-center">
            <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">Predicted Model</div>
            <div className="text-lg font-semibold text-purple-300">{predicted_label || 'Unknown'}</div>
          </div>
        </div>
      </div>

      <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 mb-4">
        <div className="flex items-center justify-between mb-3">
          <div className="text-xl font-semibold text-slate-200">Model Probabilities</div>
        </div>

        <div style={{ height: 220 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData.slice(0, 6)}
              layout="vertical"
              margin={{ top: 8, right: 8, left: 8, bottom: 4 }}
              barCategoryGap="20%"
            >
              <defs>
                {chartData.slice(0, 6).map((_, i) => (
                  <linearGradient key={i} id={gradientId(i)} x1="0" x2="1">
                    <stop offset="0%" stopColor={lightenHex(barColors[i % barColors.length], 0.16)} stopOpacity="1" />
                    <stop offset="60%" stopColor={barColors[i % barColors.length]} stopOpacity="1" />
                    <stop offset="100%" stopColor={darkenHex(barColors[i % barColors.length], 0.06)} stopOpacity="1" />
                  </linearGradient>
                ))}
              </defs>

              <CartesianGrid horizontal={false} vertical={false} />
              <XAxis
                type="number"
                domain={[0, 100]}
                tick={{ fill: '#60a5fa', fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={160}
                tick={{ fill: '#cbd5e1', fontSize: 13, fontWeight: 700 }}
                axisLine={false}
                tickLine={false}
              />

              <Tooltip content={(props) => <CustomTooltip {...props} />} cursor = {false}/>

              <Bar
                dataKey="value"
                barSize={14}
                radius={[10, 10, 10, 10]}
                isAnimationActive={true}
                animationDuration={Math.max(600, animationMs)}
                animationEasing="ease"
                activeBar={-1}

                 

              >
                {chartData.slice(0, 6).map((entry, index) => {
                  const fill = `url(#${gradientId(index)})`;
                  return (
                    <Cell
                      key={`cell-${index}`}
                      fill={fill}
                      stroke={index === 0 ? 'rgba(255,255,255,0.08)' : 'transparent'}
                      strokeWidth={index === 0 ? 1.2 : 0}
                      opacity={1}
                    />
                  );
                })}
                <LabelList
                  dataKey="value"
                  position="right"
                  formatter={(v) => `${v.toFixed(1)}%`}
                  style={{ fill: '#afc5cbff', fontSize: 12, fontWeight: 700 }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
        <div className="flex items-center justify-between mb-3">
          <div className="text-xl font-semibold text-slate-200">Model Description</div>
        </div>

        {predictedDetails ? (
          <div className="space-y-6 text-sm text-slate-200">
            <div>
              <div className="flex items-baseline justify-between">
                <div>
                 <div className="px-3 py-2 rounded-lg border border-indigo-500/40 bg-slate-900/40 shadow-sm">
  <div className="text-2xl font-semibold text-indigo-300 tracking-wide">
    {predictedDetails.name}
  </div>
  <div className="text-xl text-slate-400 italic mt-0.5">
    {predictedDetails.type}
  </div>
</div>

                </div>
                
              </div>

              {Array.isArray(predictedDetails.key_characteristics) && predictedDetails.key_characteristics.length > 0 && (
                <ul className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3 list-none">
                  {predictedDetails.key_characteristics.map((k, i) => (
                    <li
                      key={i}
                      className="flex items-start space-x-2 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 hover:bg-slate-750 transition"
                    >
                      <svg
                        className="w-4 h-4 mt-0.5 text-cyan-400 flex-shrink-0"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M4 12h16M4 6h16M4 18h16"
                        />
                      </svg>
                      <span className="text-sm font-medium leading-snug">{k}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="prose prose-sm max-w-none text-slate-300 leading-relaxed">
              {predictedDetails.description && (
                predictedDetails.description
                  .split(/(?<=\.)\s+/)
                  .reduce((chunks, sentence, idx) => {
                    if (idx % 3 === 0) chunks.push([]);
                    chunks[chunks.length - 1].push(sentence);
                    return chunks;
                  }, [])
                  .map((group, idx) => (
                    <p key={idx} className="mt-4 first:mt-0">
                      {group.join(' ')}
                    </p>
                  ))
              )}
            </div>
          </div>
        ) : (
          <div className="text-sm text-slate-400">No detailed description available for the predicted model.</div>
        )}
      </div>
    </section>
  );
}

/* -------------------- Small color helpers (unchanged) -------------------- */
function hexToRgb(hex) {
  const raw = hex.replace('#', '');
  const full = raw.length === 3 ? raw.split('').map(c => c + c).join('') : raw;
  const bigint = parseInt(full, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return { r, g, b };
}
function clamp(n, a = 0, b = 255) { return Math.max(a, Math.min(b, Math.round(n))); }
function rgbToHex({ r, g, b }) {
  return `#${((1 << 24) + (clamp(r) << 16) + (clamp(g) << 8) + clamp(b)).toString(16).slice(1)}`;
}
function lightenHex(hex, amount = 0.12) {
  const c = hexToRgb(hex);
  return rgbToHex({
    r: clamp(c.r + (255 - c.r) * amount),
    g: clamp(c.g + (255 - c.g) * amount),
    b: clamp(c.b + (255 - c.b) * amount),
  });
}
function darkenHex(hex, amount = 0.08) {
  const c = hexToRgb(hex);
  return rgbToHex({
    r: clamp(c.r * (1 - amount)),
    g: clamp(c.g * (1 - amount)),
    b: clamp(c.b * (1 - amount)),
  });
}
