import React, { useState } from 'react';
import DeepfakeCard from '../components/DeepfakeCard';
import EmotionCard from '../components/EmotionCard';
import ReverseEngCard from '../components/ReverseEngCard';
import Sidebar from '../components/Sidebar';

export default function ResultsPage({ data, analysisOptions, onReset }) {
  const [activeTab, setActiveTab] = useState(
    analysisOptions.deepfake
      ? 'deepfake'
      : analysisOptions.emotion
      ? 'emotion'
      : 'reverseEng'
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
    <div className="max-w-8xl mx-auto flex flex-col lg:flex-row gap-6">
      <main className="flex-grow lg:max-w-[calc(100%-18rem)] space-y-8">
        {/* Overall Verdict Section */}
        <section>
          <div className="bg-slate-800 rounded-lg p-6 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold">Overall Verdict</h2>
              <p className="mt-1 text-slate-400">
                {data?.verdict?.text || 'No verdict available'}
              </p>
            </div>
            <button
              onClick={onReset}
              className="rounded-md bg-red-600 px-4 py-2 font-semibold hover:bg-red-700 transition-colors"
            >
              Reset
            </button>
          </div>
        </section>

        {/* Tabs */}
        <div className="flex border-b border-slate-700 space-x-4 px-2">
          {analysisOptions.deepfake && (
            <button
              className={`py-2 px-4 rounded-t-md font-semibold transition-colors ${
                activeTab === 'deepfake'
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
              onClick={() => setActiveTab('deepfake')}
            >
              Deepfake
            </button>
          )}
          {analysisOptions.emotion && (
            <button
              className={`py-2 px-4 rounded-t-md font-semibold transition-colors ${
                activeTab === 'emotion'
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
              onClick={() => setActiveTab('emotion')}
            >
              Emotion
            </button>
          )}
          {analysisOptions.reverseEng && (
            <button
              className={`py-2 px-4 rounded-t-md font-semibold transition-colors ${
                activeTab === 'reverseEng'
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
              onClick={() => setActiveTab('reverseEng')}
            >
              Reverse Engineering
            </button>
          )}
        </div>

        {/* Active Tab Content */}
        <div className="mt-4">{renderActiveTab()}</div>
      </main>

      {/* Sidebar only on desktop */}
      <aside className="hidden lg:flex flex-col w-72 bg-slate-800 rounded-lg p-6 sticky top-6 self-start">
        <Sidebar
          verdict={data?.verdict}
          onExport={() => alert('Export functionality to be implemented')}
        />
      </aside>
    </div>
  );
}
