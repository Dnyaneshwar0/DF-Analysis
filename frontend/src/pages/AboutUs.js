import React from 'react';
import { FaLinkedin, FaGithub } from 'react-icons/fa';

// Import team images
import renukaImg from '../assests/team/renukamaam.jpg';
import parthImg from '../assests/team/parth.jpg';
import aishwaryaImg from '../assests/team/aishwarya.jpg';
import mithileshImg from '../assests/team/mithilesh.jpg';

export default function AboutUs({ siteName = 'AffectForensics' }) {
  return (
    <section className="max-w-6xl mx-auto p-10 bg-slate-900 rounded-2xl shadow-lg text-slate-100">
      <header className="mb-10 text-center">
        <h1 className="text-4xl font-extrabold bg-gradient-to-r from-indigo-400 via-purple-400 to-indigo-300 text-transparent bg-clip-text">
          {siteName}
        </h1>
      </header>

      <div className="space-y-10">
        {/* Project Overview */}
        <section className="bg-slate-800 p-8 rounded-xl border border-slate-700 shadow-md">
          <h2 className="text-2xl font-semibold mb-3 text-slate-100">Project Overview</h2>
          <p className="text-slate-300 text-lg leading-relaxed">
            AffectForensics is our third-year Mini Project at Sardar Patel Institute of Technology, Mumbai.
            It merges <span className="text-slate-100 font-medium">deepfake forensics</span>,{' '}
            <span className="text-slate-100 font-medium">reverse engineering</span>, and{' '}
            <span className="text-slate-100 font-medium">emotional manipulation analysis</span> into a unified platform.
            The system identifies synthetic media, traces its generative model and dataset characteristics, and assesses
            emotional intent using facial and linguistic cues.
          </p>
        </section>

        {/* Credits Section */}
        <section className="text-center">
          <h2 className="text-2xl font-semibold text-slate-100 mb-6">Project Credits</h2>
          <p className="text-slate-400 mb-8 italic text-lg">
            We would like to express our sincere gratitude to our mentor,{' '}
            <span className="text-slate-100 font-medium">Dr. Renuka Pawar</span>, for her invaluable guidance,
            insights, and unwavering support throughout the development of this project.
          </p>

          {/* Mentor Card */}
          <div className="flex justify-center mb-10">
            <div className="p-8 bg-slate-800 rounded-xl border border-indigo-400 shadow-lg w-80 hover:shadow-indigo-500/30 transition">
              <div className="flex flex-col items-center">
                <img
                  src={renukaImg}
                  alt="Dr. Renuka Pawar"
                  className="w-28 h-28 rounded-full object-cover mb-4 border-2 border-indigo-400 shadow-md"
                />
                <div className="font-semibold text-lg text-slate-100">Dr. Renuka Pawar</div>
                <div className="text-sm text-slate-400">Professor, CSE Department, SPIT</div>
                <div className="flex gap-4 mt-3">
                  <a
                    href="https://www.linkedin.com/in/dr-renuka-pawar-68b6ab1b/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-300 hover:text-indigo-400 text-2xl transition"
                  >
                    <FaLinkedin />
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Student Cards */}
          <div className="grid gap-8 md:grid-cols-3 justify-items-center">
            {/* Parth */}
            <div className="p-6 bg-slate-800 rounded-xl border border-slate-700 w-72 shadow-md hover:shadow-indigo-400/20 transition">
              <div className="flex flex-col items-center">
                <img
                  src={parthImg}
                  alt="Parth Gujarkar"
                  className="w-20 h-20 rounded-full object-cover mb-3 border border-slate-600 hover:border-indigo-400 transition"
                />
                <div className="font-medium text-lg text-slate-100">Parth Gujarkar</div>
                <div className="text-sm text-slate-400">Student</div>
                <div className="flex gap-4 mt-2">
                  <a
                    href="https://www.linkedin.com/in/parth-gujarkar-a0614428b/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-300 hover:text-indigo-400 text-xl transition"
                  >
                    <FaLinkedin />
                  </a>
                  <a
                    href="https://github.com/Dnyaneshwar0"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-300 hover:text-indigo-400 text-xl transition"
                  >
                    <FaGithub />
                  </a>
                </div>
              </div>
            </div>

            {/* Aishwarya */}
            <div className="p-6 bg-slate-800 rounded-xl border border-slate-700 w-72 shadow-md hover:shadow-indigo-400/20 transition">
              <div className="flex flex-col items-center">
                <img
                  src={aishwaryaImg}
                  alt="Aishwarya Mhatre"
                  className="w-20 h-20 rounded-full object-cover mb-3 border border-slate-600 hover:border-indigo-400 transition"
                />
                <div className="font-medium text-lg text-slate-100">Aishwarya Mhatre</div>
                <div className="text-sm text-slate-400">Student</div>
                <div className="flex gap-4 mt-2">
                  <a
                    href="https://www.linkedin.com/in/aishwarya-mhatre-a823802b4/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-300 hover:text-indigo-400 text-xl transition"
                  >
                    <FaLinkedin />
                  </a>
                  <a
                    href="https://github.com/ampm14"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-300 hover:text-indigo-400 text-xl transition"
                  >
                    <FaGithub />
                  </a>
                </div>
              </div>
            </div>

            {/* Mithilesh */}
            <div className="p-6 bg-slate-800 rounded-xl border border-slate-700 w-72 shadow-md hover:shadow-indigo-400/20 transition">
              <div className="flex flex-col items-center">
                <img
                  src={mithileshImg}
                  alt="Mithilesh Deshmukh"
                  className="w-20 h-20 rounded-full object-cover mb-3 border border-slate-600 hover:border-indigo-400 transition"
                />
                <div className="font-medium text-lg text-slate-100">Mithilesh Deshkmukh</div>
                <div className="text-sm text-slate-400">Student</div>
                <div className="flex gap-4 mt-2">
                  <a
                    href="https://www.linkedin.com/in/mithilesh-deshmukh-143bb7281/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-300 hover:text-indigo-400 text-xl transition"
                  >
                    <FaLinkedin />
                  </a>
                  <a
                    href="https://github.com/blast678"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-300 hover:text-indigo-400 text-xl transition"
                  >
                    <FaGithub />
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Project Repository Section */}
          <div className="mt-12 flex justify-center">
            <a
              href="https://github.com/Dnyaneshwar0/DF-Analysis"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 bg-slate-800 border border-slate-700 hover:border-indigo-500 hover:shadow-indigo-500/20 transition px-6 py-4 rounded-xl text-lg font-medium text-slate-100"
            >
              <FaGithub className="text-2xl text-indigo-400" />
              View Project Repository
            </a>
          </div>
        </section>

        <footer className="text-sm text-slate-500 text-center mt-10">
          Legal: AffectForensics estimates expressions, generative origins, and emotional intent, it does not substitute professional judgement.
        </footer>
      </div>
    </section>
  );
}
