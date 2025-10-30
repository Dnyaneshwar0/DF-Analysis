import React, { useState } from 'react';
import DeepfakeCard from '../components/DeepfakeCard';
import EmotionCard from '../components/EmotionCard';
import ReverseEngCard from '../components/ReverseEngCard';
import Sidebar from '../components/Sidebar';


export default function ResultsPage({ data, analysisOptions, onReset }) {
//   const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="max-w-8xl mx-auto flex flex-col lg:flex-row gap-6">
        <main className="flex-grow lg:max-w-[calc(100%-18rem)] space-y-8">
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

        {analysisOptions.deepfake && (
          <DeepfakeCard data={data.deepfake} />
        )}

        {analysisOptions.emotion && (
          <EmotionCard data={data.emotion} />
        )}

        {analysisOptions.reverseEng && (
          <ReverseEngCard data={data.reverseEng} />
        )}
      </main>

      {/* Sidebar only on desktop */}
      <aside
        className={`hidden lg:flex flex-col w-72 bg-slate-800 rounded-lg p-6 sticky top-6 self-start transition-transform duration-300 `}
      >
        <Sidebar
          verdict={data?.verdict}
          onExport={() => alert('Export functionality to be implemented')}
        />
      </aside>
    </div>
  );
}
