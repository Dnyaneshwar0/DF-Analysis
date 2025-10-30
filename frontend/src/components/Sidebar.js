import React from 'react';

export default function Sidebar({ verdict, onExport, onToggle }) {
  return (
    <div className="flex flex-col h-full justify-between">
      <div>
        <h3 className="text-xl font-bold mb-4">Summary</h3>
        <p className="mb-2">
          <strong>Status:</strong>{' '}
          <span
            className={`font-semibold ${
              verdict?.status === 'fake' ? 'text-red-400' : 'text-green-400'
            }`}
          >
            {verdict?.status?.toUpperCase() || 'Unknown'}
          </span>
        </p>
        <p className="mb-2">
          <strong>Confidence:</strong>{' '}
          <span className="font-semibold">
            {(verdict?.confidence * 100 || 0).toFixed(1)}%
          </span>
        </p>
        <p className="mb-2 text-slate-400">{verdict?.summary || 'No summary available.'}</p>
      </div>

      <div className="space-y-3 mt-6">
        <button
          onClick={onExport}
          className="w-full py-2 bg-indigo-500 rounded hover:bg-indigo-600 transition-colors font-semibold"
        >
          Export JSON / PDF
        </button>
        {/* <button
          onClick={onToggle}
          className="w-full py-2 bg-slate-700 rounded hover:bg-slate-600 transition-colors font-semibold"
        >
          Toggle Sidebar
        </button> */}
      </div>
    </div>
  );
}
