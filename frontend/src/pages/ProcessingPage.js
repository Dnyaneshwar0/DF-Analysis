import React, { useEffect, useState } from 'react';
import { FaCheckCircle, FaSpinner, FaCircle } from 'react-icons/fa';

const STEPS = [
  { key: 'extract', label: 'Extracting frames & audio' },
  { key: 'deepfake', label: 'Running deepfake detection' },
  { key: 'emotion', label: 'Running emotion analysis' },
  { key: 'reverseEng', label: 'Running reverse engineering' },
  { key: 'finalize', label: 'Finalizing results' },
];

export default function ProcessingPage({ fileName = 'Unknown File', modelStatus }) {
  const [currentStep, setCurrentStep] = useState(0);

  // Controlled progress simulation — will advance until all steps complete
  useEffect(() => {
    // Automatically advance visual progress based on modelStatus
    const completedSteps = STEPS.filter((step) => {
      if (step.key === 'reverseEng') return modelStatus.reverseEng === 'done';
      if (step.key === 'deepfake') return modelStatus.deepfake === 'done';
      if (step.key === 'emotion') return modelStatus.emotion === 'done';
      if (step.key === 'extract') return true; // instant
      if (step.key === 'finalize')
        return (
          modelStatus.deepfake === 'done' &&
          modelStatus.emotion === 'done' &&
          modelStatus.reverseEng === 'done'
        );
      return false;
    }).length;

    setCurrentStep(completedSteps);

    // --- handle auto-redirect when all steps done ---
    const allDone =
      modelStatus.deepfake === 'done' &&
      modelStatus.emotion === 'done' &&
      modelStatus.reverseEng === 'done';

    if (allDone) {
      const timeout = setTimeout(() => {
        const event = new CustomEvent('processingComplete');
        window.dispatchEvent(event);
      }, 1500); // small delay for UX
      return () => clearTimeout(timeout);
    }
  }, [modelStatus]);

  const progress = ((currentStep / STEPS.length) * 100).toFixed(1);

  return (
    <section className="max-w-4xl mx-auto p-10 bg-slate-900 border border-slate-800 rounded-2xl shadow-xl text-slate-100">
      {/* Title */}
      <h2 className="text-2xl font-bold mb-6 text-center">
        Processing: <span className="text-indigo-400">{fileName}</span>
      </h2>

      {/* Progress Bar */}
      <div className="relative w-full bg-slate-800 rounded-full h-4 overflow-hidden border border-slate-700">
        <div
          className="absolute top-0 left-0 h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-400 transition-all duration-700 ease-out"
          style={{ width: `${progress}%` }}
        ></div>
      </div>

      {/* Step List */}
      <ul className="space-y-4 mt-8">
        {STEPS.map((step, idx) => {
          const completed = idx < currentStep;
          const active = idx === currentStep && currentStep < STEPS.length;
          return (
            <li
              key={step.key}
              className={`flex items-center space-x-4 p-3 rounded-lg border transition-all duration-300 ${
                completed
                  ? 'border-green-700 bg-slate-800 text-green-300'
                  : active
                  ? 'border-indigo-600 bg-slate-800 text-indigo-300 shadow-md shadow-indigo-500/20'
                  : 'border-slate-800 text-slate-500'
              }`}
            >
              <span className="w-6 h-6 flex items-center justify-center">
                {completed ? (
                  <FaCheckCircle className="text-green-400" />
                ) : active ? (
                  <FaSpinner className="animate-spin text-indigo-400" />
                ) : (
                  <FaCircle className="text-slate-600" />
                )}
              </span>
              <span
                className={`text-sm md:text-base ${
                  active ? 'font-semibold animate-pulse' : ''
                }`}
              >
                {step.label}
              </span>
            </li>
          );
        })}
      </ul>

      {/* Completion Message */}
      {currentStep >= STEPS.length && (
        <div className="mt-8 text-center text-green-400 font-semibold text-lg animate-fadeIn">
           Redirecting to Analysis Results
        </div>
      )}
    </section>
  );
}
