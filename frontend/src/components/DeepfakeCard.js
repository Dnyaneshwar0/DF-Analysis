import React, { useState } from 'react';

export default function DeepfakeCard({ data }) {
  const [activeTab, setActiveTab] = useState('overview');

  if (!data) {
    return (
      <section className="bg-slate-800 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-2">Deepfake Detection</h3>
        <p className="text-slate-400">No deepfake data available.</p>
      </section>
    );
  }

  return (
    <section className="bg-slate-800 rounded-lg p-6">
      <h3 className="text-xl font-semibold mb-4">Deepfake Detection</h3>

      {/* Tabs */}
      <div className="flex border-b border-slate-700 mb-6">
        {['overview', 'frames', 'explain'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`py-2 px-4 -mb-px border-b-2 font-semibold focus:outline-none ${
              activeTab === tab
                ? 'border-indigo-400 text-indigo-400'
                : 'border-transparent text-slate-400 hover:text-indigo-400'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div>
          <p className="mb-3 text-slate-300">{data.overview}</p>
          <p>
            <strong>Confidence Score: </strong>
            <span
              className={`font-bold ${
                data.confidence >= 0.7
                  ? 'text-red-400'
                  : 'text-green-400'
              }`}
            >
              {(data.confidence * 100).toFixed(1)}%
            </span>
          </p>
        </div>
      )}

      {activeTab === 'frames' && (
        <div className="grid grid-cols-3 gap-4 max-h-64 overflow-y-auto">
          {data.frames && data.frames.length ? (
            data.frames.map(({ id, imageUrl, score }) => (
              <div key={id} className="relative rounded overflow-hidden border border-slate-700">
                <img
                  src={imageUrl}
                  alt={`Frame ${id}`}
                  className="object-cover w-full h-24"
                  loading="lazy"
                />
                <div
                  className={`absolute bottom-0 left-0 right-0 bg-black bg-opacity-60 text-xs text-center py-1 ${
                    score >= 0.7 ? 'text-red-400' : 'text-green-400'
                  }`}
                >
                  Score: {(score * 100).toFixed(1)}%
                </div>
              </div>
            ))
          ) : (
            <p className="text-slate-400">No frames available.</p>
          )}
        </div>
      )}

      {activeTab === 'explain' && (
        <pre className="bg-slate-900 p-4 rounded text-sm overflow-x-auto max-h-48 whitespace-pre-wrap">
          {data.explanation || 'No explanation available.'}
        </pre>
      )}
    </section>
  );
}
