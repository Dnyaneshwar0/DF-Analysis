import React, { useState } from 'react';
import { FaFileDownload, FaClipboard, FaShareAlt, FaCheckCircle, FaTimesCircle, FaExclamationTriangle } from 'react-icons/fa';
import DeepfakeCard from '../components/DeepfakeCard';
import EmotionCard from '../components/EmotionCard';
import ReverseEngCard from '../components/ReverseEngCard';

export default function ResultsPage({ data = {}, analysisOptions = {}, onReset }) {
  // prefer a tab which is enabled AND has data; otherwise fall back to option order
  const initialTab = (() => {
    if (analysisOptions.deepfake && data.deepfake) return 'deepfake';
    if (analysisOptions.emotion && data.emotion) return 'emotion';
    if (analysisOptions.reverseEng && data.reverseEng) return 'reverseEng';
    if (analysisOptions.deepfake) return 'deepfake';
    if (analysisOptions.emotion) return 'emotion';
    if (analysisOptions.reverseEng) return 'reverseEng';
    return 'deepfake';
  })();

  const [activeTab, setActiveTab] = useState(initialTab);
  const [exporting, setExporting] = useState(false);
  const [copied, setCopied] = useState(false);

  // Determine overall verdict from available real data (prefer deepfake detection)
  const verdict = (() => {
    // If deepfake detection is present use that (label/confidence)
    if (data?.deepfake && (data.deepfake.label || typeof data.deepfake.confidence === 'number')) {
      const labelRaw = (data.deepfake.label || '').toString().toUpperCase();
      const label = labelRaw === 'FAKE' ? 'Fake' : labelRaw === 'REAL' ? 'Real' : 'Suspicious';
      const confidence = typeof data.deepfake.confidence === 'number' ? data.deepfake.confidence : null;
      const text = confidence != null
        ? `Detected as ${labelRaw} with ${(confidence * 100).toFixed(1)}% confidence.`
        : `Detected as ${labelRaw}.`;
      return { label, text };
    }

    // fallback to an explicit verdict object if present in data
    if (data?.verdict && (data.verdict.label || data.verdict.text)) {
      return {
        label: data.verdict.label || 'Unknown',
        text: data.verdict.text || 'No detailed verdict provided',
      };
    }

    // ultimate fallback
    return { text: 'No verdict available', label: 'Unknown' };
  })();

  const verdictLabel = (verdict.label || '').toLowerCase();

  const doExport = (format = 'json') => {
    setExporting(true);
    try {
      if (format === 'json') {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `affectforensics_result.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      } else if (format === 'pdf') {
        // Placeholder: real implementation would generate a proper report server-side or via a library
        alert('PDF export not implemented in this demo — integrate server/pdf-generator.');
      }
    } finally {
      setExporting(false);
    }
  };

  const copyJSON = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch (e) {
      alert('Copy failed — permission denied or unsupported browser.');
    }
  };

  // small summary stats if present (keep existing behaviour)
  const confidence = data?.summary?.confidence ?? (data?.deepfake?.confidence ?? null);
  const topEmotion = data?.emotion?.topEmotion;

  return (
    <div className="max-w-6xl mx-auto px-4 lg:px-8">
      <main className="flex-1 space-y-6">
        {/* Verdict card */}
        <section className="bg-gradient-to-b from-slate-800 to-slate-900 rounded-xl p-6 shadow-md border border-slate-700">
          <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-4">
                <h2 className="text-2xl font-bold text-white">Overall Verdict</h2>
                <div>
                  {verdictLabel === 'real' && (
                    <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-green-900/40 text-green-300 text-sm font-medium border border-green-800">
                      <FaCheckCircle /> Real
                    </span>
                  )}
                  {verdictLabel === 'fake' && (
                    <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-900/30 text-rose-200 text-sm font-medium border border-rose-800/40">
                      <FaTimesCircle /> Fake
                    </span>
                  )}
                  {verdictLabel === 'suspicious' && (
                    <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-yellow-900/40 text-yellow-300 text-sm font-medium border border-yellow-800">
                      <FaExclamationTriangle /> Suspicious
                    </span>
                  )}
                  {['real', 'fake', 'suspicious'].indexOf(verdictLabel) === -1 && (
                    <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800 text-slate-300 text-sm font-medium border border-slate-700">
                      {verdict.label || 'Unknown'}
                    </span>
                  )}
                </div>
              </div>

              <p className="mt-3 text-slate-300 max-w-2xl">
                {verdict.text}
              </p>

              {/* small stats */}
              <div className="mt-4 flex flex-wrap gap-4 text-sm">
                {typeof confidence === 'number' && (
                  <div className="px-3 py-1 rounded-md bg-slate-800 border border-slate-700 text-slate-200">
                    Confidence: <span className="font-medium ml-2">{(confidence * 100).toFixed(1)}%</span>
                  </div>
                )}
                {topEmotion && (
                  <div className="px-3 py-1 rounded-md bg-slate-800 border border-slate-700 text-slate-200">
                    Top emotion: <span className="font-medium ml-2">{topEmotion}</span>
                  </div>
                )}
                {data?.deepfake?.metadata?.duration_sec && (
                  <div className="px-3 py-1 rounded-md bg-slate-800 border border-slate-700 text-slate-200">
                    Duration: <span className="font-medium ml-2">{data.deepfake.metadata.duration_sec}s</span>
                  </div>
                )}
                {data?.deepfake?.metadata?.frames_analyzed && (
                  <div className="px-3 py-1 rounded-md bg-slate-800 border border-slate-700 text-slate-200">
                    Frames analyzed: <span className="font-medium ml-2">{data.deepfake.metadata.frames_analyzed}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Side actions */}
            <div className="flex-shrink-0 flex flex-col items-end gap-3">
              <div className="flex gap-2">
                <button
                  onClick={() => doExport('json')}
                  disabled={exporting}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-gradient-to-r from-indigo-500 to-violet-500 text-white font-semibold shadow-md hover:from-indigo-600 hover:to-violet-600 transition"
                >
                  <FaFileDownload /> Export JSON
                </button>

                <button
                  onClick={copyJSON}
                  className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-slate-800 text-slate-200 border border-slate-700 hover:bg-slate-750 transition"
                >
                  <FaClipboard /> {copied ? 'Copied' : 'Copy JSON'}
                </button>
              </div>

              <button
                onClick={onReset}
                className="mt-2 w-full rounded-md bg-rose-500/90 hover:bg-rose-700 text-white font-semibold px-3 py-2 shadow-sm transition focus:outline-none focus:ring-2 focus:ring-rose-500/30"
              >
                Reset
              </button>
            </div>
          </div>
        </section>

        {/* Tabs */}
        <div>
          <nav className="relative flex space-x-3 border-b border-slate-700 px-1">
            {analysisOptions.deepfake && (
              <button
                onClick={() => setActiveTab('deepfake')}
                className={`relative py-2 px-4 rounded-t-md font-semibold transition ${
                  activeTab === 'deepfake' ? 'text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                Deepfake
                {activeTab === 'deepfake' && <span className="absolute left-2 right-2 -bottom-[1px] h-1 rounded-t-lg bg-indigo-500" />}
              </button>
            )}

            {analysisOptions.emotion && (
              <button
                onClick={() => setActiveTab('emotion')}
                className={`relative py-2 px-4 rounded-t-md font-semibold transition ${
                  activeTab === 'emotion' ? 'text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                Emotion
                {activeTab === 'emotion' && <span className="absolute left-2 right-2 -bottom-[1px] h-1 rounded-t-lg bg-indigo-500" />}
              </button>
            )}

            {analysisOptions.reverseEng && (
              <button
                onClick={() => setActiveTab('reverseEng')}
                className={`relative py-2 px-4 rounded-t-md font-semibold transition ${
                  activeTab === 'reverseEng' ? 'text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                Reverse Engineering
                {activeTab === 'reverseEng' && <span className="absolute left-2 right-2 -bottom-[1px] h-1 rounded-t-lg bg-indigo-500" />}
              </button>
            )}
          </nav>
        </div>

        {/* Tab content */}
        <section className="mt-4">
          <div className="bg-slate-900 rounded-lg p-6 border border-slate-800 min-h-[220px]">
            {/* Keep each card mounted and just hide/show via `hidden` so internal state persists */}
            {analysisOptions.deepfake && (
              <div hidden={activeTab !== 'deepfake'}>
                <DeepfakeCard data={data.deepfake || {}} />
              </div>
            )}

            {analysisOptions.emotion && (
              <div hidden={activeTab !== 'emotion'}>
                <EmotionCard data={data.emotion || {}} />
              </div>
            )}

            {analysisOptions.reverseEng && (
              <div hidden={activeTab !== 'reverseEng'}>
                <ReverseEngCard data={data.reverseEng || {}} />
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
