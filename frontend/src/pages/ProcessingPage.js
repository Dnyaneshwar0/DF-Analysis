import React, { useEffect, useState } from 'react';

const STEPS = [
  'Extracting frames & audio',
  'Running deepfake detection',
  'Running emotion analysis',
  'Running reverse engineering',
  'Finalizing results',
];

export default function ProcessingPage({ fileName }) {
  const [currentStep, setCurrentStep] = useState(0);

  // Progress simulation logic - can be removed when backend integrated
  useEffect(() => {
    if (currentStep >= STEPS.length) return;

    const timer = setTimeout(() => {
      setCurrentStep((prev) => prev + 1);
    }, 1000);

    return () => clearTimeout(timer);
  }, [currentStep]);

  return (
    <section className="max-w-4xl mx-auto space-y-6">
      <h2 className="text-xl font-semibold mb-4">Processing: {fileName || 'Unknown File'}</h2>

      <div className="w-full bg-slate-800 rounded-full h-4 overflow-hidden">
        <div
          className="bg-indigo-400 h-4 rounded-full transition-all duration-500"
          style={{ width: `${(currentStep / STEPS.length) * 100}%` }}
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow={(currentStep / STEPS.length) * 100}
          role="progressbar"
        ></div>
      </div>

      <ul className="space-y-3 mt-6">
        {STEPS.map((step, idx) => (
          <li
            key={step}
            className={`flex items-center space-x-3 ${
              idx < currentStep
                ? 'text-green-400'
                : idx === currentStep
                ? 'text-indigo-400 font-semibold'
                : 'text-slate-500'
            }`}
          >
            <span className="w-6 h-6 flex items-center justify-center rounded-full border-2 border-current">
              {idx < currentStep ? '✓' : idx + 1}
            </span>
            <span>{step}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
