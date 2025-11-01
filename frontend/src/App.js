import React, { useState } from 'react';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import ProcessingPage from './pages/ProcessingPage';
import AboutUs from './pages/AboutUs'; // <-- import the About Us page

// Constants to manage app states
const APP_STATES = {
  UPLOAD: 'upload',
  PROCESSING: 'processing',
  RESULTS: 'results',
  ABOUT: 'about',
};

/**
 * Root App component controlling main app flow
 */
export default function App() {
  const [appState, setAppState] = useState(APP_STATES.UPLOAD);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [analysisOptions, setAnalysisOptions] = useState({
    deepfake: true,
    emotion: true,
    reverseEng: true,
  });
  const [resultData, setResultData] = useState(null);

  // Handler for starting analysis
  const startAnalysis = () => {
    if (!uploadedFile) return;
    setAppState(APP_STATES.PROCESSING);

    setTimeout(() => {
      const mockResults = require('./mock/mockData').default;
      setResultData(mockResults);
      setAppState(APP_STATES.RESULTS);
    }, 3000);
  };

  // Handler for resetting app to initial state
  const resetApp = () => {
    setUploadedFile(null);
    setResultData(null);
    setAppState(APP_STATES.UPLOAD);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-900 text-slate-100">
      {/* Header */}
      <header className="p-4 border-b border-slate-700 flex justify-between items-center">
        <h1
          onClick={() => setAppState(APP_STATES.UPLOAD)}
          className="text-3xl font-extrabold glow-indigo select-none cursor-pointer"
        >
          Deepfake & Emotion Analyzer
        </h1>

        <nav className="text-sm space-x-6">
          <button
            onClick={() => setAppState(APP_STATES.UPLOAD)}
            className={`hover:text-indigo-400 transition ${
              appState === APP_STATES.UPLOAD ? 'text-indigo-400' : 'text-slate-300'
            }`}
          >
            Home
          </button>

          <button
            onClick={() => setAppState(APP_STATES.ABOUT)}
            className={`hover:text-indigo-400 transition ${
              appState === APP_STATES.ABOUT ? 'text-indigo-400' : 'text-slate-300'
            }`}
          >
            About Us
          </button>
        </nav>
      </header>

      {/* Main Content */}
      <main className="flex-grow p-6 max-w-7xl mx-auto w-full">
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
          <ProcessingPage fileName={uploadedFile?.name || ''} />
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
      <footer className="p-4 border-t border-slate-700 text-sm text-slate-500 text-center select-none">
        &copy; 2025 ampm — Deepfake Detection & Emotional Analysis
      </footer>
    </div>
  );
}
