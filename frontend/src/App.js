import React, { useState } from 'react';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import ProcessingPage from './pages/ProcessingPage';

// Constants to manage app states
const APP_STATES = {
  UPLOAD: 'upload',
  PROCESSING: 'processing',
  RESULTS: 'results',
};

/**
 * Root App component controlling main app flow
 */
export default function App() {
  // State to manage current app phase
  const [appState, setAppState] = useState(APP_STATES.UPLOAD);

  // File uploaded by user (File object)
  const [uploadedFile, setUploadedFile] = useState(null);

  // Analysis options selected by user
  const [analysisOptions, setAnalysisOptions] = useState({
    deepfake: true,
    emotion: true,
    reverseEng: true,
  });

  // Mock result data (would come from backend in real app)
  const [resultData, setResultData] = useState(null);

  // Handler for starting analysis
  const startAnalysis = () => {
    if (!uploadedFile) return;
    setAppState(APP_STATES.PROCESSING);

    // Simulate backend processing with timeout
    setTimeout(() => {
      // TODO: Replace with real API call
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
      <header className="p-4 border-b border-slate-700">
        <h1 className="text-3xl font-extrabold glow-indigo select-none">
          Deepfake & Emotion Analyzer
        </h1>
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
          <ProcessingPage
            fileName={uploadedFile?.name || ''}
          />
        )}

        {appState === APP_STATES.RESULTS && (
          <ResultsPage
            data={resultData}
            analysisOptions={analysisOptions}
            onReset={resetApp}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="p-4 border-t border-slate-700 text-sm text-slate-500 text-center select-none">
        &copy; 2025 ampm - Deepfake Detection & Emotional Analysis
      </footer>
    </div>
  );
}
