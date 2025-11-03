import React, { useEffect, useState } from 'react';
import { FaCheckCircle, FaSpinner, FaCircle } from 'react-icons/fa';

const STEPS = [
  'Extracting frames & audio',
  'Running deepfake detection',
  'Running emotion analysis',
  'Running reverse engineering',
  'Finalizing results',
];

export default function ProcessingPage({ fileName = 'Unknown File' }) {
  const [currentStep, setCurrentStep] = useState(0);

  // Controlled progress simulation — will advance until all steps complete
  useEffect(() => {
    if (currentStep >= STEPS.length) return;
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev < STEPS.length ? prev + 1 : prev));
    }, 1200);
    return () => clearInterval(interval);
  }, [currentStep]);

  const progress = Math.min((currentStep / STEPS.length) * 100, 100);

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
              key={step}
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
                {step}
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
