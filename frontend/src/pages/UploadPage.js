import React from 'react';

const ANALYSIS_OPTIONS = [
  { id: 'deepfake', label: 'Deepfake Detection' },
  { id: 'emotion', label: 'Emotion Analysis' },
  { id: 'reverseEng', label: 'Reverse Engineering' },
];

export default function UploadPage({
  uploadedFile,
  setUploadedFile,
  analysisOptions,
  setAnalysisOptions,
  onAnalyze,
}) {
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setUploadedFile(file || null);
  };

  const toggleOption = (id) => {
    setAnalysisOptions((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <section className="max-w-3xl mx-auto space-y-8">
      <div>
        <label
          htmlFor="file-upload"
          className="block mb-2 text-lg font-semibold"
        >
          Upload Media File
        </label>
        <input
          id="file-upload"
          type="file"
          accept="video/*,image/*,audio/*"
          onChange={handleFileChange}
          className="w-full rounded-md border border-slate-700 bg-slate-800 text-slate-100 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400"
        />
        {uploadedFile && (
          <p className="mt-2 text-slate-400 select-text">
            Selected: {uploadedFile.name} ({(uploadedFile.size / 1024 / 1024).toFixed(2)} MB)
          </p>
        )}
      </div>

      <div>
        <p className="mb-2 font-semibold">Select Analyses to Perform:</p>
        <div className="flex flex-wrap gap-6">
          {ANALYSIS_OPTIONS.map(({ id, label }) => (
            <label
              key={id}
              className="flex items-center space-x-3 cursor-pointer select-none"
            >
              <input
                type="checkbox"
                checked={analysisOptions[id]}
                onChange={() => toggleOption(id)}
                className="w-5 h-5 rounded bg-slate-700 border border-slate-600 checked:bg-indigo-400 checked:ring-2 checked:ring-indigo-400 focus:outline-none"
              />
              <span className="text-slate-200">{label}</span>
            </label>
          ))}
        </div>
      </div>

      <button
        type="button"
        onClick={onAnalyze}
        disabled={!uploadedFile || !Object.values(analysisOptions).some(Boolean)}
        className="w-full py-3 rounded-md bg-indigo-500 text-white font-semibold disabled:bg-indigo-300 disabled:cursor-not-allowed transition-colors duration-200 hover:bg-indigo-600"
      >
        Analyze
      </button>
    </section>
  );
}
