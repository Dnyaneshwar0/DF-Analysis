import React, { useState } from 'react';
import { FaFileDownload, FaClipboard, FaShareAlt, FaCheckCircle, FaTimesCircle, FaExclamationTriangle } from 'react-icons/fa';
import DeepfakeCard from '../components/DeepfakeCard';
import EmotionCard from '../components/EmotionCard';
import ReverseEngCard from '../components/ReverseEngCard';

export default function ResultsPage({ data = {}, analysisOptions = {}, onReset }) {
  const [activeTab, setActiveTab] = useState(
    analysisOptions.deepfake ? 'deepfake' :
    analysisOptions.emotion ? 'emotion' :
    analysisOptions.reverseEng ? 'reverseEng' :
    'deepfake'
  );

  const [exporting, setExporting] = useState(false);
  const [copied, setCopied] = useState(false);

  const verdict = data?.verdict || { text: 'No verdict available', label: 'Unknown' };
  const verdictLabel = (verdict.label || '').toLowerCase();

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'deepfake':
        return analysisOptions.deepfake && <DeepfakeCard data={data.deepfake} />;
      case 'emotion':
        return analysisOptions.emotion && <EmotionCard data={data.emotion} />;
      case 'reverseEng':
        return analysisOptions.reverseEng && <ReverseEngCard data={data.reverseEng} />;
      default:
        return null;
    }
  };

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

  // small summary stats if present
  const confidence = data?.summary?.confidence;
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
                    <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-900/40 text-red-300 text-sm font-medium border border-red-800">
                      <FaTimesCircle /> Fake
                    </span>
                  )}
                  {verdictLabel === 'suspicious' && (
                    <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-yellow-900/40 text-yellow-300 text-sm font-medium border border-yellow-800">
                      <FaExclamationTriangle /> Suspicious
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
                {data?.duration && (
                  <div className="px-3 py-1 rounded-md bg-slate-800 border border-slate-700 text-slate-200">
                    Duration: <span className="font-medium ml-2">{data.duration}</span>
                  </div>
                )}
              </div>
            </div>

            <div className="flex-shrink-0 flex flex-col items-end gap-3">
              <div className="flex gap-2">
                <button
                  onClick={() => doExport('json')}
                  disabled={exporting}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-indigo-600 text-white font-semibold hover:bg-indigo-700 transition"
                >
                  <FaFileDownload /> Export JSON
                </button>

                <button
                  onClick={copyJSON}
                  className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-slate-800 text-slate-200 border border-slate-700 hover:bg-slate-700 transition"
                >
                  <FaClipboard /> {copied ? 'Copied' : 'Copy JSON'}
                </button>
              </div>

              <button
                onClick={onReset}
                className="mt-2 w-full rounded-md bg-red-600 px-3 py-2 font-semibold hover:bg-red-700 transition"
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
            {renderActiveTab()}
          </div>
        </section>
      </main>
    </div>
  );
}
