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

  // Emotion section now includes the provided linegraph-ready data + segments table
  emotion: {
    // timeline: points for the line graph
    timeline: [
      { time: 0.0, neutral: 0.3353, admiration: 0.1504, annoyance: 0.0 },
      { time: 2.0, neutral: 0.3201, admiration: 0.1054, annoyance: 0.0686 },
      { time: 6.0, neutral: 0.386,  admiration: 0.0974, annoyance: 0.0 },
      { time: 11.0, neutral: 0.3293, admiration: 0.0829, annoyance: 0.0732 },
      { time: 12.0, neutral: 0.4471, admiration: 0.0911, annoyance: 0.0705 },
      { time: 12.5, neutral: 0.2924, admiration: 0.0923, annoyance: 0.0605 },
      { time: 13.0, neutral: 0.3293, admiration: 0.0829, annoyance: 0.0732 },
      { time: 15.0, neutral: 0.1738, admiration: 0.0939, annoyance: 0.0 },
      { time: 20.5, neutral: 0.3371, admiration: 0.0853, annoyance: 0.0 },
    ],
    // schema hint for UI/components
    schema: {
      fields: ['timestamp', 'emotion', 'confidence'],
      timestampFormat: 'HH:mm:ss',
    },
    // linegraph payload (ready to feed to chart component)
    linegraph: {
      video: 'eddiewoo',
      top_emotions: ['neutral', 'admiration', 'annoyance'],
      data: [
        { time: 0.0,  neutral: 0.3353, admiration: 0.1504, annoyance: 0.0 },
        { time: 2.0,  neutral: 0.3201, admiration: 0.1054, annoyance: 0.0686 },
        { time: 6.0,  neutral: 0.3860, admiration: 0.0974, annoyance: 0.0 },
        { time: 11.0, neutral: 0.3293, admiration: 0.0829, annoyance: 0.0732 },
        { time: 12.0, neutral: 0.4471, admiration: 0.0911, annoyance: 0.0705 },
        { time: 12.5, neutral: 0.2924, admiration: 0.0923, annoyance: 0.0605 },
        { time: 13.0, neutral: 0.3293, admiration: 0.0829, annoyance: 0.0732 },
        { time: 15.0, neutral: 0.1738, admiration: 0.0939, annoyance: 0.0 },
        { time: 20.5, neutral: 0.3371, admiration: 0.0853, annoyance: 0.0 },
      ],
    },

    // segments: detailed table rows for the UI table (added)
    segments: [
      {
        start: 0.0,
        end: 2.0,
        text: 'one a the golden ratio is a mathematical reality',
        emotions: ['neutral', 'admiration', 'approval'],
        intensities: [0.3353, 0.1504, 0.0813],
        confidence: 0.3353,
      },
      {
        start: 2.0,
        end: 6.0,
        text: 'one a that like facts you can find everywhere',
        emotions: ['neutral', 'admiration', 'annoyance'],
        intensities: [0.3201, 0.1054, 0.0686],
        confidence: 0.3201,
      },
      {
        start: 6.0,
        end: 11.0,
        text: 'from the changes of your fingers to the pillars of the parthenon',
        emotions: ['neutral', 'admiration', 'approval'],
        intensities: [0.386, 0.0974, 0.0737],
        confidence: 0.386,
      },
      {
        start: 11.0,
        end: 12.0,
        text: "that's why even at a party of 5000 people",
        emotions: ['neutral', 'admiration', 'annoyance'],
        intensities: [0.3293, 0.0829, 0.0732],
        confidence: 0.3293,
      },
      {
        start: 12.0,
        end: 12.5,
        text: 'people',
        emotions: ['neutral', 'admiration', 'annoyance'],
        intensities: [0.4471, 0.0911, 0.0705],
        confidence: 0.4471,
      },
      {
        start: 12.5,
        end: 13.0,
        text: "that's even at a poverty of 50 5000 penplelf",
        emotions: ['neutral', 'admiration', 'annoyance'],
        intensities: [0.2924, 0.0923, 0.0605],
        confidence: 0.2924,
      },
      {
        start: 13.0,
        end: 15.0,
        text: "that's why even at a party of 5000 people",
        emotions: ['neutral', 'admiration', 'annoyance'],
        intensities: [0.3293, 0.0829, 0.0732],
        confidence: 0.3293,
      },
      {
        start: 15.0,
        end: 19.5,
        text: "i'm proud to declared love mathematics",
        emotions: ['love', 'neutral', 'admiration'],
        intensities: [0.2275, 0.1738, 0.0939],
        confidence: 0.2275,
      },
      {
        start: 20.5,
        end: 21.5,
        text: 'will seal a we',
        emotions: ['neutral', 'admiration', 'curiosity'],
        intensities: [0.3371, 0.0853, 0.0714],
        confidence: 0.3371,
      },
    ],
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
