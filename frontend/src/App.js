// src/App.js
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
    extract: 'done',      // treat extraction as done by default
    deepfake: 'pending',
    emotion: 'pending',
    reverseEng: 'pending',
    // finalize will be shown/handled inside ProcessingPage visually
  });

const startAnalysis = async () => {
  if (!uploadedFile) return;
  setAppState(APP_STATES.PROCESSING);

  // Build initial status: for selected models -> running (or mocked done), else skipped
  setModelStatus((prev) => ({
    ...prev,
    deepfake: analysisOptions.deepfake ? 'running' : 'skipped',
    emotion: analysisOptions.emotion ? 'running' : 'skipped', // <-- no longer mocked as 'done'
    reverseEng: analysisOptions.reverseEng ? 'running' : 'skipped',
  }));

  // helper to run a POST and update status/results as soon as it finishes
  const runAndUpdate = async (url, statusKey, resultKeyMapper = (json) => json) => {
    try {
      // Create fresh FormData for every request to avoid subtle reuse issues
      const fd = new FormData();
      fd.append('file', uploadedFile);

      const res = await fetch(url, { method: 'POST', body: fd });
      if (!res.ok) {
        setModelStatus((prev) => ({ ...prev, [statusKey]: 'error' }));
        console.error(`${statusKey} response not ok:`, res.status);
        return null;
      }
      const json = await res.json();

      // mark done and store mapped result
      setModelStatus((prev) => ({ ...prev, [statusKey]: 'done' }));
      setResultData((prev) => ({ ...(prev || {}), [statusKey]: resultKeyMapper(json) }));
      return json;
    } catch (err) {
      setModelStatus((prev) => ({ ...prev, [statusKey]: 'error' }));
      console.error(`${statusKey} fetch error:`, err);
      return null;
    }
  };

  // Start only the selected model calls in parallel
  const promises = [];
  if (analysisOptions.reverseEng) {
    promises.push(runAndUpdate('http://localhost:5000/reveng/analyze', 'reverseEng', (j) => j));
  }
  if (analysisOptions.deepfake) {
    promises.push(
      runAndUpdate('http://localhost:5000/detect/analyze', 'deepfake', (j) => j.deepfake ?? j)
    );
  }
  if (analysisOptions.emotion) {
    // server returns {"status":"success","result": <merged_json>}
    promises.push(
      runAndUpdate('http://localhost:5000/emotion/analyze', 'emotion', (j) => j.result ?? j)
    );
  }

  // Wait for started jobs to settle; ProcessingPage will handle finalization UI and navigation
  try {
    await Promise.allSettled(promises);
    // Per-request updates already updated modelStatus/resultData in runAndUpdate
    // Do NOT change appState here — let ProcessingPage call onComplete -> parent -> navigate to results
  } catch (e) {
    console.error('Unexpected error waiting for analyses:', e);
  }
};

  const handleProcessingComplete = () => {
    // Called by ProcessingPage after finalize delay / visual completion
    setAppState(APP_STATES.RESULTS);
  };

  const resetApp = () => {
    setUploadedFile(null);
    setResultData(null);
    setModelStatus({
      extract: 'done',
      deepfake: 'pending',
      emotion: 'pending',
      reverseEng: 'pending',
    });
    setAppState(APP_STATES.UPLOAD);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 font-inter">
      {/* Header omitted for brevity — keep yours */}
      <header className="p-5 border-b border-slate-800 flex justify-between items-center sticky top-0 z-50 bg-slate-950 shadow-md">
        <div onClick={() => setAppState(APP_STATES.UPLOAD)} className="cursor-pointer select-none">
          <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 via-purple-400 to-indigo-300 text-transparent bg-clip-text hover:opacity-90 transition">
            AffectForensics
          </h1>
        </div>

        <nav className="flex items-center space-x-8 text-sm font-medium">
          <button onClick={() => setAppState(APP_STATES.UPLOAD)} className={`transition ${appState === APP_STATES.UPLOAD ? 'text-indigo-400 border-b-2 border-indigo-400 pb-1' : 'text-slate-300 hover:text-indigo-400'}`}>Home</button>
          <button onClick={() => setAppState(APP_STATES.ABOUT)} className={`transition ${appState === APP_STATES.ABOUT ? 'text-indigo-400 border-b-2 border-indigo-400 pb-1' : 'text-slate-300 hover:text-indigo-400'}`}>About Us</button>
        </nav>
      </header>

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
            analysisOptions={analysisOptions}
            onComplete={handleProcessingComplete} // <-- parent waits for this
            finalizeDelay={3500}
            completeDelay={1500}
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

      <footer className="p-5 border-t border-slate-800 text-center text-xs md:text-sm text-slate-500 select-none bg-slate-950/90 backdrop-blur">
        <p>&copy; 2025 <span className="text-indigo-400 font-medium">ampm</span> AffectForensics</p>
      </footer>
    </div>
  );
}
