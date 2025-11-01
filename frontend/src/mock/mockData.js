const mockData = {
  verdict: {
    status: 'fake',
    confidence: 0.82,
    summary:
      'The uploaded media is very likely to be a deepfake based on frame-level analysis and artifact detection.',
    text: 'Warning: High likelihood of deepfake content detected.',
  },

  deepfake: {
    overview:
      'The system analyzed 120 frames and detected visual artifacts consistent with deepfake manipulations.',
    confidence: 0.82,
    frames: [
      { id: 1, imageUrl: 'https://placekitten.com/200/120', score: 0.95 },
      { id: 2, imageUrl: 'https://placekitten.com/201/120', score: 0.88 },
      { id: 3, imageUrl: 'https://placekitten.com/202/120', score: 0.75 },
      { id: 4, imageUrl: 'https://placekitten.com/203/120', score: 0.15 },
      { id: 5, imageUrl: 'https://placekitten.com/204/120', score: 0.45 },
    ],
    explanation:
      `Deepfake artifacts are identified by inconsistent lighting and unnatural facial texture in multiple frames.\n` +
      `Confidence score is calculated based on a neural network trained on over 10,000 real and fake samples.\n` +
      `Frames with scores above 0.7 indicate likely manipulation.\n` +
      `Further investigation recommended for borderline cases.`,
  },

  // Emotion section reset for reinitialization
  emotion: {
    // empty timeline ready for reinitialization
    timeline: [],
    // metadata or schema hint for UI/components
    schema: {
      fields: ['timestamp', 'emotion', 'confidence'],
      timestampFormat: 'HH:mm:ss', // UI hint
    },
  },

  reverseEng: {
    metadata: `Format: MP4\nCodec: H.264\nDuration: 15 seconds\nBitrate: 2000 kbps\nDate Created: 2025-10-18T14:32:00Z\nDevice: Canon EOS R5`,
    editingTraces: [
      'Detected multiple re-encoding passes',
      'Presence of artificial noise patterns',
      'Unusual keyframe intervals suggesting frame interpolation',
    ],
    rawData: {
      hexDump: '00 01 02 03 04 05 06 07 08 09 ...',
      audioWaveformAnalysis: {
        peaks: [0.2, 0.5, 0.3, 0.9, 0.1],
        anomaliesDetected: true,
      },
      videoFrameStats: {
        totalFrames: 360,
        duplicatedFrames: 12,
        droppedFrames: 3,
      },
      suspiciousMetadataTags: ['EditingSoftware', 'Adobe After Effects'],
    },
  },
};

export default mockData;
