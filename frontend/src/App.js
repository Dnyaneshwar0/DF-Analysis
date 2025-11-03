import React, { useState } from 'react';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import ProcessingPage from './pages/ProcessingPage';
import AboutUs from './pages/AboutUs';

const APP_STATES = {
  UPLOAD: 'upload',
  PROCESSING: 'processing',
  RESULTS: 'results',
  ABOUT: 'about',
};

export default function App() {
  const [appState, setAppState] = useState(APP_STATES.UPLOAD);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [analysisOptions, setAnalysisOptions] = useState({
    deepfake: true,
    emotion: true,
    reverseEng: true,
  });

  const [resultData, setResultData] = useState(null);
  const [modelStatus, setModelStatus] = useState({
  deepfake: 'done',   // mock
  emotion: 'done',    // mock
  reverseEng: 'pending', // real model call
});


  const startAnalysis = async () => {
  if (!uploadedFile) return;
  setAppState(APP_STATES.PROCESSING);

  // Mark model states at the start
  setModelStatus({
    deepfake: 'done',   // mocked, so instantly complete
    emotion: 'done',    // mocked, so instantly complete
    reverseEng: 'running', // real analysis starts
  });

  try {
    const formData = new FormData();
    formData.append('file', uploadedFile);

    // Call your Flask backend
    const response = await fetch('http://localhost:5000/reveng/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error('Failed to analyze video');

    const reverseData = await response.json();

    // Update model state to show it’s complete
    setModelStatus((prev) => ({ ...prev, reverseEng: 'done' }));

    // Combine with mock data for now
    const mockResults = require('./mock/mockData').default;
    const finalResults = { ...mockResults, reverseEng: reverseData };

    setResultData(finalResults);

    // Move to results page
    setAppState(APP_STATES.RESULTS);
  } catch (error) {
    console.error('Error:', error);
    setModelStatus((prev) => ({ ...prev, reverseEng: 'error' }));
    alert('Something went wrong while analyzing the video.');
    setAppState(APP_STATES.UPLOAD);
  }
};

  const resetApp = () => {
    setUploadedFile(null);
    setResultData(null);
    setAppState(APP_STATES.UPLOAD);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 font-inter">
      {/* Header */}
      <header className="p-5 border-b border-slate-800 flex justify-between items-center sticky top-0 z-50 bg-slate-950 shadow-md">
        <div
          onClick={() => setAppState(APP_STATES.UPLOAD)}
          className="cursor-pointer select-none"
        >
          <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 via-purple-400 to-indigo-300 text-transparent bg-clip-text hover:opacity-90 transition">
            AffectForensics
          </h1>
        </div>

        <nav className="flex items-center space-x-8 text-sm font-medium">
          <button
            onClick={() => setAppState(APP_STATES.UPLOAD)}
            className={`transition ${
              appState === APP_STATES.UPLOAD
                ? 'text-indigo-400 border-b-2 border-indigo-400 pb-1'
                : 'text-slate-300 hover:text-indigo-400'
            }`}
          >
            Home
          </button>

          <button
            onClick={() => setAppState(APP_STATES.ABOUT)}
            className={`transition ${
              appState === APP_STATES.ABOUT
                ? 'text-indigo-400 border-b-2 border-indigo-400 pb-1'
                : 'text-slate-300 hover:text-indigo-400'
            }`}
          >
            About Us
          </button>
        </nav>
      </header>

      {/* Main */}
      <main className="flex-grow p-6 md:p-10 max-w-6xl mx-auto w-full">
        {appState === APP_STATES.UPLOAD && (
          <UploadPage
            uploadedFile={uploadedFile}
            setUploadedFile={setUploadedFile}
            analysisOptions={analysisOptions}
            setAnalysisOptions={setAnalysisOptions}
            onAnalyze={startAnalysis}
          />
        )}

        {appState === APP_STATES.PROCESSING && (
          <ProcessingPage
            fileName={uploadedFile?.name || ''}
            modelStatus={modelStatus}
          />
        )}

        {appState === APP_STATES.RESULTS && (
          <ResultsPage
            data={resultData}
            analysisOptions={analysisOptions}
            onReset={resetApp}
          />
        )}

        {appState === APP_STATES.ABOUT && <AboutUs />}
      </main>

      {/* Footer */}
      <footer className="p-5 border-t border-slate-800 text-center text-xs md:text-sm text-slate-500 select-none bg-slate-950/90 backdrop-blur">
        <p>
          &copy; 2025 <span className="text-indigo-400 font-medium">ampm</span> AffectForensics
        </p>
      </footer>
    </div>
  );
}
