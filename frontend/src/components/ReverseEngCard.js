import React, { useState } from 'react';

export default function ReverseEngCard({ data }) {
  const [expanded, setExpanded] = useState(false);

  if (!data) {
    return (
      <section className="bg-slate-800 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-2">Reverse Engineering</h3>
        <p className="text-slate-400">No reverse engineering data available.</p>
      </section>
    );
  }

  return (
    <section className="bg-slate-800 rounded-lg p-6">
      <h3 className="text-xl font-semibold mb-4">Reverse Engineering</h3>

      <div className="space-y-3">
        <div>
          <strong>Metadata:</strong>
          <pre className="bg-slate-900 p-3 rounded max-h-48 overflow-auto text-sm whitespace-pre-wrap">
            {data.metadata || 'No metadata found.'}
          </pre>
        </div>

        <div>
          <strong>Editing Traces:</strong>
          <ul className="list-disc list-inside max-h-48 overflow-auto text-slate-300 text-sm">
            {data.editingTraces && data.editingTraces.length ? (
              data.editingTraces.map((trace, idx) => (
                <li key={idx}>{trace}</li>
              ))
            ) : (
              <li>No editing traces detected.</li>
            )}
          </ul>
        </div>

        <button
          onClick={() => setExpanded((v) => !v)}
          className="mt-4 px-4 py-2 bg-indigo-500 rounded hover:bg-indigo-600 transition-colors"
        >
          {expanded ? 'Hide Raw Data' : 'Show Raw Data'}
        </button>

        {expanded && (
          <pre className="mt-4 bg-slate-900 p-4 rounded max-h-64 overflow-auto text-xs whitespace-pre-wrap">
            {JSON.stringify(data.rawData, null, 2)}
          </pre>
        )}
      </div>
    </section>
  );
}
