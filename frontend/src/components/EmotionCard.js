import React from 'react';

export default function EmotionCard({ data }) {
  if (!data) {
    return (
      <section className="bg-slate-800 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-2">Emotion Analysis</h3>
        <p className="text-slate-400">No emotion data available.</p>
      </section>
    );
  }

  return (
    <section className="bg-slate-800 rounded-lg p-6">
      <h3 className="text-xl font-semibold mb-4">Emotion Analysis</h3>

      <div className="overflow-x-auto">
        <table className="w-full table-fixed text-sm border-collapse border border-slate-700">
          <thead>
            <tr className="bg-slate-700">
              <th className="border border-slate-600 px-2 py-1 text-left">Timestamp</th>
              <th className="border border-slate-600 px-2 py-1 text-left">Emotion</th>
              <th className="border border-slate-600 px-2 py-1 text-left">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {data.timeline && data.timeline.length ? (
              data.timeline.map(({ timestamp, emotion, confidence }) => (
                <tr key={timestamp} className="even:bg-slate-900">
                  <td className="border border-slate-700 px-2 py-1">{timestamp}</td>
                  <td className="border border-slate-700 px-2 py-1">{emotion}</td>
                  <td className="border border-slate-700 px-2 py-1">
                    {(confidence * 100).toFixed(1)}%
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="3" className="text-center text-slate-400 py-4">
                  No timeline data available.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
