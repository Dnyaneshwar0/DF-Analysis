import React, { useRef } from 'react';
import { FaUpload, FaCheckCircle, FaRegSquare } from 'react-icons/fa';

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
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setUploadedFile(file || null);
  };

  const toggleOption = (id) => {
    setAnalysisOptions((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const openFilePicker = () => fileInputRef.current && fileInputRef.current.click();

  const formattedSize = (file) =>
    file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : '';

  return (
    <section className="max-w-3xl mx-auto p-8">
      <div className="bg-slate-900 rounded-2xl shadow-lg p-6 text-slate-100">
        <header className="mb-4 text-center">
          <h2 className="text-xl font-semibold">Upload Media</h2>
          <p className="mt-1 text-slate-400 text-sm">
            Supported Format: Video files under 200 MB
          </p>
        </header>

        <div className="grid gap-5 md:grid-cols-2">
          {/* Upload Card */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-4 flex flex-col items-center justify-center gap-3">
            <input
              ref={fileInputRef}
              id="file-upload"
              type="file"
              accept="video/*,image/*,audio/*"
              onChange={handleFileChange}
              className="hidden"
            />

            <button
              type="button"
              onClick={openFilePicker}
              className="flex flex-col items-center gap-2 w-full py-8 rounded-xl border-2 border-dashed border-slate-700 hover:border-indigo-400 transition focus:outline-none"
              aria-label="Click to upload media file"
            >
              <div className="flex items-center justify-center w-14 h-14 rounded-full bg-slate-700 text-indigo-300 shadow-md">
                <FaUpload className="text-lg" />
              </div>

              {!uploadedFile ? (
                <>
                  <div className="text-sm font-medium text-slate-100">
                    Click to select a file
                  </div>
                  <div className="text-xs text-slate-400">
                    or drag & drop into this area
                  </div>
                </>
              ) : (
                <>
                  <div className="text-sm font-medium text-slate-100 truncate max-w-full">
                    {uploadedFile.name}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {formattedSize(uploadedFile)}
                  </div>
                </>
              )}
            </button>

            {uploadedFile && (
              <div className="mt-2 w-full px-3 py-2 rounded-md bg-slate-900 border border-slate-700 text-xs text-slate-300">
                Selected:{' '}
                <span className="font-medium text-slate-100">
                  {uploadedFile.name}
                </span>{' '}
                — {formattedSize(uploadedFile)}
              </div>
            )}
          </div>

          {/* Options + CTA */}
          <aside className="flex flex-col justify-between">
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-4 space-y-3">
              <p className="mb-1 font-semibold text-slate-200 text-sm">
                Select Analyses
              </p>

              <div className="flex flex-col gap-2">
                {ANALYSIS_OPTIONS.map(({ id, label }) => {
                  const checked = !!analysisOptions[id];
                  return (
                    <button
                      key={id}
                      type="button"
                      onClick={() => toggleOption(id)}
                      className="w-full flex items-center justify-between gap-3 p-2.5 rounded-lg bg-slate-900 border border-slate-700 hover:border-indigo-400 transition"
                    >
                      <div className="flex items-center gap-2">
                        <div className="text-indigo-400">
                          {checked ? (
                            <FaCheckCircle className="text-base" />
                          ) : (
                            <FaRegSquare className="text-base text-slate-500" />
                          )}
                        </div>
                        <span className="text-slate-100 text-sm">{label}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="mt-4">
              <button
                type="button"
                onClick={onAnalyze}
                disabled={!uploadedFile || !Object.values(analysisOptions).some(Boolean)}
                className="w-full py-2.5 rounded-lg bg-gradient-to-r from-indigo-500 to-indigo-400 text-white font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-[0_6px_20px_rgba(99,102,241,0.15)] transition"
              >
                Analyze
              </button>
            </div>
          </aside>
        </div>
      </div>
    </section>
  );
}
