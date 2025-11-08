// src/pages/ProcessingPage.js
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { FaCheckCircle, FaSpinner, FaCircle, FaTimesCircle } from 'react-icons/fa';

/**
 * ProcessingPage
 * Props:
 *  - fileName: string
 *  - modelStatus: { extract, deepfake, emotion, reverseEng, finalize, ... }
 *  - analysisOptions: { deepfake: bool, emotion: bool, reverseEng: bool }
 *  - onComplete: () => void
 *  - finalizeDelay: ms to visually animate finalization (default 3500)
 *  - completeDelay: ms to wait after finalize done before calling onComplete (default 1500)
 */
export default function ProcessingPage({
  fileName = 'Unknown File',
  modelStatus = {},
  analysisOptions = {},
  onComplete = null,
  finalizeDelay = 3500,
  completeDelay = 500,
}) {
  // Build dynamic steps from analysisOptions
  const STEPS = useMemo(() => {
    const base = [{ key: 'extract', label: 'Extracting frames & audio' }];

    if (analysisOptions.deepfake) base.push({ key: 'deepfake', label: 'Running Deepfake Detection' });
    if (analysisOptions.reverseEng) base.push({ key: 'reverseEng', label: 'Running Reverse Engineering' });
    if (analysisOptions.emotion) base.push({ key: 'emotion', label: 'Running Emotion Analysis' });

    base.push({ key: 'finalize', label: 'Finalizing results' });
    return base;
  }, [analysisOptions]);

  // Selected model keys (exclude extract & finalize)
  const selectedModelKeys = useMemo(
    () => STEPS.map((s) => s.key).filter((k) => k !== 'extract' && k !== 'finalize'),
    [STEPS]
  );

  // local finalize visual state: pending | running | done
  const [finalizeState, setFinalizeState] = useState('pending');

  // refs for timers so we can clear on unmount
  const finalizeTimerRef = useRef(null);
  const completeTimerRef = useRef(null);
  const sequenceStartedRef = useRef(false);

  // Helper: consider a model finished when it's 'done' OR 'error' OR 'skipped'
  const isFinishedStatus = (st) => st === 'done' || st === 'error' || st === 'skipped';

  // Determine display status for a given step key
  const statusFor = (key) => {
    if (key === 'extract') {
      const v = modelStatus.extract;
      if (v === undefined) return 'done'; // treat missing extract as done (your app marks it done by default)
      if (v === 'done' || v === 'skipped') return 'done';
      if (v === 'running') return 'running';
      if (v === 'error') return 'error';
      return 'pending';
    }

    if (key === 'finalize') {
      // if parent provided an explicit finalize status, prefer it when meaningfully set
      const parentFinalize = modelStatus.finalize;
      if (parentFinalize === 'done' || parentFinalize === 'skipped') return 'done';
      if (parentFinalize === 'running') return 'running';
      if (parentFinalize === 'error') return 'error';
      // otherwise use local visual finalize state
      return finalizeState;
    }

    const v = modelStatus[key];
    if (v === 'done' || v === 'skipped') return 'done';
    if (v === 'running') return 'running';
    if (v === 'error') return 'error';
    return 'pending';
  };

  // Count completed steps
  const doneCount = useMemo(() => {
    return STEPS.reduce((acc, s) => (statusFor(s.key) === 'done' ? acc + 1 : acc), 0);
  }, [STEPS, modelStatus, finalizeState]); // statusFor reads modelStatus & finalizeState

  const progress = ((doneCount / STEPS.length) * 100).toFixed(1);

  // Create a compact string reflecting statuses of selected models for effect deps
  const selectedStatusesKey = useMemo(() => selectedModelKeys.map((k) => modelStatus[k] ?? 'pending').join('|'), [
    selectedModelKeys,
    modelStatus,
  ]);

  // Start finalize sequence when ALL selected models are finished (or when there are none)
  useEffect(() => {
    // if we've already started sequence, ignore unless something resets
    if (sequenceStartedRef.current) return;

    const allSelectedFinished =
      selectedModelKeys.length === 0 ||
      selectedModelKeys.every((k) => isFinishedStatus(modelStatus[k]));

    if (allSelectedFinished) {
      sequenceStartedRef.current = true;
      setFinalizeState('running');

      // schedule finalize -> done
      finalizeTimerRef.current = setTimeout(() => {
        finalizeTimerRef.current = null;
        setFinalizeState('done');

        // schedule onComplete after completeDelay
        completeTimerRef.current = setTimeout(() => {
          completeTimerRef.current = null;
          if (typeof onComplete === 'function') onComplete();
        }, completeDelay);
      }, finalizeDelay);
    }

    return () => {
      if (finalizeTimerRef.current) {
        clearTimeout(finalizeTimerRef.current);
        finalizeTimerRef.current = null;
      }
      if (completeTimerRef.current) {
        clearTimeout(completeTimerRef.current);
        completeTimerRef.current = null;
      }
    };
    // depend on the compact statuses key so we re-evaluate when any selected status changes
  }, [selectedStatusesKey, finalizeDelay, completeDelay, onComplete, selectedModelKeys.length]);

  // If any selected model transitions back to non-finished while finalize running, reset the sequence
  useEffect(() => {
    // if sequence never started, nothing to do
    if (!sequenceStartedRef.current) return;

    const allStillFinished = selectedModelKeys.every((k) => isFinishedStatus(modelStatus[k]));
    if (!allStillFinished) {
      // cancel timers and reset
      if (finalizeTimerRef.current) {
        clearTimeout(finalizeTimerRef.current);
        finalizeTimerRef.current = null;
      }
      if (completeTimerRef.current) {
        clearTimeout(completeTimerRef.current);
        completeTimerRef.current = null;
      }
      sequenceStartedRef.current = false;
      setFinalizeState('pending');
    }
  }, [selectedStatusesKey, selectedModelKeys]);

  // Safety: if parent directly sets modelStatus.finalize to done (or error/skipped),
  // honor that by calling onComplete after completeDelay.
  useEffect(() => {
    const parentFinalize = modelStatus.finalize;
    if (parentFinalize === 'done' || parentFinalize === 'skipped') {
      // clear any local timers and call onComplete after completeDelay
      if (finalizeTimerRef.current) {
        clearTimeout(finalizeTimerRef.current);
        finalizeTimerRef.current = null;
      }
      if (completeTimerRef.current) {
        clearTimeout(completeTimerRef.current);
        completeTimerRef.current = null;
      }
      setFinalizeState('done');
      completeTimerRef.current = setTimeout(() => {
        completeTimerRef.current = null;
        if (typeof onComplete === 'function') onComplete();
      }, completeDelay);
    }
    if (parentFinalize === 'running') {
      // show running state
      setFinalizeState('running');
    }
  }, [modelStatus.finalize, completeDelay, onComplete]);

  // Clean up timers on unmount
  useEffect(() => {
    return () => {
      if (finalizeTimerRef.current) clearTimeout(finalizeTimerRef.current);
      if (completeTimerRef.current) clearTimeout(completeTimerRef.current);
    };
  }, []);

  return (
    <section className="max-w-4xl mx-auto p-10 bg-slate-900 border border-slate-800 rounded-2xl shadow-xl text-slate-100">
      <h2 className="text-2xl font-bold mb-6 text-center">
        Processing: <span className="text-indigo-400">{fileName}</span>
      </h2>

      <div className="relative w-full bg-slate-800 rounded-full h-4 overflow-hidden border border-slate-700">
        <div
          className="absolute top-0 left-0 h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-400 transition-all duration-700 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      <ul className="space-y-4 mt-8">
        {STEPS.map((step) => {
          const status = statusFor(step.key);
          const completed = status === 'done';
          const running = status === 'running';
          const errored = status === 'error';
          return (
            <li
              key={step.key}
              className={`flex items-center justify-between space-x-4 p-3 rounded-lg border transition-all duration-300 ${
                completed
                  ? 'border-green-700 bg-slate-800 text-green-300'
                  : running
                  ? 'border-indigo-600 bg-slate-800 text-indigo-300 shadow-md shadow-indigo-500/20'
                  : errored
                  ? 'border-rose-700 bg-slate-800 text-rose-300'
                  : 'border-slate-800 text-slate-500'
              }`}
            >
              <div className="flex items-center space-x-4">
                <span className="w-6 h-6 flex items-center justify-center">
                  {completed ? (
                    <FaCheckCircle className="text-green-400" />
                  ) : running ? (
                    <FaSpinner className="animate-spin text-indigo-400" />
                  ) : errored ? (
                    <FaTimesCircle className="text-rose-500" />
                  ) : (
                    <FaCircle className="text-slate-600" />
                  )}
                </span>

                <span className={`text-sm md:text-base ${running ? 'font-semibold animate-pulse' : ''}`}>{step.label}</span>
              </div>
            </li>
          );
        })}
      </ul>

      {doneCount >= STEPS.length && (
        <div className="mt-8 text-center text-green-400 font-semibold text-lg animate-fadeIn">Redirecting to Analysis Results</div>
      )}
    </section>
  );
}
