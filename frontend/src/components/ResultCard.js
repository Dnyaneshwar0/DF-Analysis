import React from 'react';

export default function ResultCard({ title, children }) {
  return (
    <div className="bg-slate-800 bg-opacity-40 backdrop-blur-md border border-cyan-600 rounded-lg p-6 shadow-lg mb-6 hover:shadow-[0_0_20px_rgb(14,165,233)] transition-shadow duration-300">
      <h3 className="text-cyan-400 font-mono font-semibold text-xl mb-4 select-none drop-shadow-[0_0_8px_rgb(14,165,233)]">
        {title}
      </h3>
      <div className="text-slate-200 text-sm leading-relaxed">{children}</div>
    </div>
  );
}
