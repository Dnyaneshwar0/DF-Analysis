import React, { useState } from 'react';
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

  return (
    <div className="max-w-8xl mx-auto px-4 lg:px-8">
      <main className="flex-1 space-y-6">
        {/* Clean single Verdict card */}
        <section className="bg-slate-800 rounded-lg p-6 shadow-sm border border-slate-700">
          <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-white">Overall Verdict</h2>
              <p className="mt-1 text-slate-400">
                {data?.verdict?.text || 'No verdict available'}
              </p>
            </div>

            <div className="flex-shrink-0 flex flex-col items-end gap-3">
              <button
                onClick={() => alert('Export functionality to be implemented')}
                className="px-4 py-2 rounded-md bg-indigo-600 text-white font-semibold hover:bg-indigo-700 transition"
              >
                Export JSON / PDF
              </button>
              <button
                onClick={onReset}
                className="rounded-md bg-red-600 px-3 py-2 font-semibold hover:bg-red-700 transition"
              >
                Reset
              </button>
            </div>
          </div>
        </section>

        {/* Tabs directly below Verdict */}
        <nav className="flex border-b border-slate-700">
          {analysisOptions.deepfake && (
            <button
              onClick={() => setActiveTab('deepfake')}
              className={`py-2 px-4 rounded-t-md font-semibold transition-colors ${
                activeTab === 'deepfake'
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Deepfake
            </button>
          )}
          {analysisOptions.emotion && (
            <button
              onClick={() => setActiveTab('emotion')}
              className={`py-2 px-4 rounded-t-md font-semibold transition-colors ${
                activeTab === 'emotion'
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Emotion
            </button>
          )}
          {analysisOptions.reverseEng && (
            <button
              onClick={() => setActiveTab('reverseEng')}
              className={`py-2 px-4 rounded-t-md font-semibold transition-colors ${
                activeTab === 'reverseEng'
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Reverse Engineering
            </button>
          )}
        </nav>

        {/* Active Tab Content */}
        <section className="mt-4">
          <div className="bg-slate-900 rounded-lg p-6 border border-slate-800 min-h-[180px]">
            {renderActiveTab()}
          </div>
        </section>
      </main>
    </div>
  );
}
