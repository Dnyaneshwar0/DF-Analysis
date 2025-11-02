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

  reverseEng: {
    result: {
      all_probs: {
        DeepFakeDetection: 0.08096712827682495,
        Deepfakes: 0.35939571261405945,
        Face2Face: 0.2339722365140915,
        FaceSwap: 0.23401771485805511,
        NeuralTextures: 0.09164723008871078,
      },
      confidence: 0.35939571261405945,
      predicted_label: "Deepfakes",
    },
    status: "success",

    df_models_desc: [
      {
        name: "DeepFakes",
        type: "Neural face-swapping (autoencoder-based)",
        source: "Community deepfake software (early open-source implementation)",
        introduced: 2018,
        description:
          "DeepFakes is one of the earliest and most iconic deepfake generation techniques, built on a pair of autoencoder-decoder networks sharing a common encoder. The method learns to represent facial features and expressions from both source and target identities in a shared latent space. Each identity has its own decoder that reconstructs the person’s facial details from this shared representation. During inference, a source face is encoded through the common encoder and then decoded using the target’s decoder, effectively transferring the source’s expression, pose, and motion onto the target’s identity. The resulting swapped face is then composited back into the target frame using post-processing methods such as color correction and seamless blending. Despite being relatively simple, this approach marked the beginning of the modern deepfake era, enabling realistic face-swaps purely through neural network learning rather than explicit geometry. It often suffers from visible artifacts or boundary mismatches, but its simplicity and effectiveness made it the basis for many later, more advanced variants.",
        key_characteristics: [
          "Neural autoencoder-decoder based face swapping",
          "No explicit 3D modeling; purely data-driven",
          "Often leaves edge artifacts or inconsistent lighting",
          "Formed the foundation of the deepfake ecosystem",
        ],
      },
      {
        name: "Face2Face",
        type: "Classical computer graphics-based facial reenactment",
        authors: "Thies et al.",
        introduced: 2016,
        description:
          "Face2Face is a real-time facial reenactment system that transfers the expressions and mouth movements of a source actor onto a target video while preserving the target’s identity. It relies on fitting a parametric 3D morphable model (3DMM) to both the source and target faces. The system tracks facial landmarks and expression parameters for each frame, mapping the source’s expression coefficients onto the target’s model to synthesize a new target face with the desired expressions. This manipulated face is then re-rendered, seamlessly blended into the original video, and adjusted for illumination consistency. Unlike neural methods, Face2Face is purely graphics-based, depending on geometric alignment and rendering rather than deep learning. The technique is remarkably effective for real-time reenactment, but it tends to produce visible inconsistencies when faces rotate sharply, are partially occluded, or when lighting varies dramatically. It demonstrated early on that expressive face manipulation could be achieved interactively, paving the way for neural and hybrid approaches that followed.",
        key_characteristics: [
          "Graphics-based approach using 3D morphable models",
          "Real-time facial expression transfer",
          "Preserves target identity while modifying expressions",
          "May struggle with occlusion, lighting, or extreme poses",
        ],
      },
      {
        name: "FaceSwap",
        type: "Traditional graphics-based face-swapping",
        source: "Open-source faceswap tool (non-deep learning)",
        introduced: 2016,
        description:
          "FaceSwap is a classical, non-neural face-swapping approach that uses geometric alignment and image compositing to replace one person’s face with another. It begins by detecting facial landmarks in both the source and target images or frames, estimating 3D head poses to align faces accurately in orientation and scale. The source face is then warped to match the target’s pose and lighting using 3D morphable models and affine transformations. After alignment, the source texture is blended into the target frame using color correction, edge feathering, and seamless cloning techniques to ensure a natural appearance. Since FaceSwap does not rely on machine learning, it requires no training data, making it simple and efficient for direct application. However, the results can appear less realistic compared to neural methods, especially under challenging lighting conditions or complex facial motions. It serves as a foundational, graphics-based benchmark for face manipulation, highlighting the differences between traditional compositing and modern deepfake synthesis.",
        key_characteristics: [
          "Classical computer graphics method with 3D model fitting",
          "Requires no neural network training",
          "Sensitive to pose and lighting mismatches",
          "Useful as a non-neural baseline for comparison",
        ],
      },
      {
        name: "NeuralTextures",
        type: "Neural rendering and texture-based manipulation",
        authors: "Thies et al.",
        introduced: 2019,
        description:
          "NeuralTextures represents a hybrid approach that blends the principles of 3D modeling with neural rendering to achieve highly realistic and temporally coherent facial manipulations. The technique introduces the concept of 'neural textures'—learnable high-dimensional texture maps that store latent information about a person’s facial appearance, such as skin details, reflectance, and lighting characteristics. A convolutional neural network (CNN) acts as a rendering engine that synthesizes realistic output frames based on these learned textures and driving parameters such as expression and pose. During training, the neural texture and rendering network jointly learn to reproduce real video frames of a target actor, effectively building a person-specific generative model. At inference time, the system can manipulate or reenact the target face using new expressions derived from another actor or parametric model. The resulting videos are often extremely realistic and difficult to detect visually, as the neural rendering process can capture subtle appearance changes like lighting shifts and wrinkles. However, this realism comes at a computational cost, requiring substantial training data and resources. NeuralTextures stands as one of the most sophisticated and photorealistic manipulation techniques in the FaceForensics++ suite.",
        key_characteristics: [
          "Combines 3D geometry with neural rendering",
          "Learns neural textures encoding facial details and lighting",
          "Produces highly realistic and temporally stable outputs",
          "Computationally heavy and data-intensive",
        ],
      },
      {
        name: "DeepFakeDetection (DFD)",
        type: "Neural face-swapping (multiple architectures)",
        source: "Google / Jigsaw DeepFake Detection Dataset (DFD)",
        introduced: 2020,
        description:
          "The DeepFakeDetection (DFD) subset incorporated into FaceForensics++ comes from Google and Jigsaw’s DeepFake Detection Dataset, which was designed to support the DeepFake Detection Challenge (DFDC). Unlike the other FF++ methods that are tied to specific algorithms, DFD includes manipulated videos generated by multiple proprietary and undisclosed deepfake models developed by Google researchers. The dataset was produced using professional actors recorded under controlled conditions, ensuring consistent lighting, framing, and high video quality. The resulting face-swapped videos exhibit fewer visible artifacts compared to the original DeepFakes, representing a new generation of more sophisticated and less detectable manipulations. The DFD subset plays a critical role in FaceForensics++ as an 'unseen domain' for evaluating the generalization capability of deepfake detection models. By incorporating DFD, the extended version of FaceForensics++ provides a broader and more realistic benchmark for assessing how detection algorithms perform on manipulations that differ in architecture and visual style from the ones used during training.",
        key_characteristics: [
          "Integrates Google’s DFD dataset into FF++",
          "Contains videos generated by several advanced deepfake architectures",
          "High-quality, realistic manipulations with minimal artifacts",
          "Serves as an unseen test set for evaluating deepfake detectors",
        ],
      },
    ],
  },

  // existing emotion, audioWaveform, emotionIntensities remain unchanged...
   // Existing Emotion Section
  emotion: {
    timeline: [
      { time: 0.0, neutral: 0.3353, admiration: 0.1504, annoyance: 0.0 },
      { time: 2.0, neutral: 0.3201, admiration: 0.1054, annoyance: 0.0686 },
      { time: 6.0, neutral: 0.386, admiration: 0.0974, annoyance: 0.0 },
      { time: 11.0, neutral: 0.3293, admiration: 0.0829, annoyance: 0.0732 },
      { time: 12.0, neutral: 0.4471, admiration: 0.0911, annoyance: 0.0705 },
      { time: 12.5, neutral: 0.2924, admiration: 0.0923, annoyance: 0.0605 },
      { time: 13.0, neutral: 0.3293, admiration: 0.0829, annoyance: 0.0732 },
      { time: 15.0, neutral: 0.1738, admiration: 0.0939, annoyance: 0.0 },
      { time: 20.5, neutral: 0.3371, admiration: 0.0853, annoyance: 0.0 },
    ],
    schema: {
      fields: ['timestamp', 'emotion', 'confidence'],
      timestampFormat: 'HH:mm:ss',
    },
    linegraph: {
      video: 'eddiewoo',
      top_emotions: ['neutral', 'admiration', 'annoyance'],
      data: [
        { time: 0.0, neutral: 0.3353, admiration: 0.1504, annoyance: 0.0 },
        { time: 2.0, neutral: 0.3201, admiration: 0.1054, annoyance: 0.0686 },
        { time: 6.0, neutral: 0.386, admiration: 0.0974, annoyance: 0.0 },
        { time: 11.0, neutral: 0.3293, admiration: 0.0829, annoyance: 0.0732 },
        { time: 12.0, neutral: 0.4471, admiration: 0.0911, annoyance: 0.0705 },
        { time: 12.5, neutral: 0.2924, admiration: 0.0923, annoyance: 0.0605 },
        { time: 13.0, neutral: 0.3293, admiration: 0.0829, annoyance: 0.0732 },
        { time: 15.0, neutral: 0.1738, admiration: 0.0939, annoyance: 0.0 },
        { time: 20.5, neutral: 0.3371, admiration: 0.0853, annoyance: 0.0 },
      ],
    },
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

  // Acoustic waveform (existing)
  audioWaveform: {
    meta: {
      sr: 16000,
      hop_length: 512,
      frame_duration: 0.032,
      num_frames: 25,
    },
    axes: {
      x_label: 'Time (s)',
      y_label: 'Normalized Amplitude',
    },
    frames: {
      time: [
        0.0, 2.3, 4.6, 6.9, 9.2, 11.5, 13.8, 16.1, 18.4, 20.7, 23.0, 25.3, 27.6,
        29.9, 32.2, 34.5, 36.8, 39.1, 41.4, 43.7, 46.0, 48.3, 50.6, 52.9, 55.2,
      ],
      envelope: [
        0.13, 0.56, 0.48, 0.78, 0.41, 0.28, 0.17, 0.3, 0.4, 0.12, 0.18, 0.23, 0.31,
        0.46, 0.54, 0.38, 0.24, 0.15, 0.28, 0.36, 0.52, 0.44, 0.22, 0.17, 0.13,
      ],
    },
  },

  
  emotionIntensities: {
    meta: {
      sr: 16000,
      hop_length: 512,
      frame_duration: 0.032,
      num_frames: 25,
    },
    axes: {
      x_label: 'Time (s)',
      y_label: 'Intensity (0..1)',
    },
    emotions: ['fearful', 'surprised', 'sad'],
    frames: {
      time: [
        0.0, 2.3, 4.6, 6.9, 9.2, 11.5, 13.8, 16.1, 18.4, 20.7, 23.0, 25.3, 27.6,
        29.9, 32.2, 34.5, 36.8, 39.1, 41.4, 43.7, 46.0, 48.3, 50.6, 52.9, 55.2,
      ],
      intensities: [
        [0.34, 0.312, 0.134],
        [0.36, 0.3, 0.14],
        [0.38, 0.29, 0.15],
        [0.42, 0.27, 0.15],
        [0.5, 0.24, 0.16],
        [0.58, 0.2, 0.18],
        [0.62, 0.18, 0.2],
        [0.6, 0.2, 0.2],
        [0.55, 0.24, 0.21],
        [0.48, 0.3, 0.22],
        [0.44, 0.35, 0.21],
        [0.4, 0.38, 0.2],
        [0.36, 0.4, 0.21],
        [0.34, 0.42, 0.24],
        [0.32, 0.45, 0.23],
        [0.3, 0.47, 0.23],
        [0.28, 0.48, 0.24],
        [0.3, 0.46, 0.24],
        [0.33, 0.43, 0.24],
        [0.37, 0.38, 0.25],
        [0.42, 0.33, 0.25],
        [0.46, 0.3, 0.24],
        [0.44, 0.29, 0.25],
        [0.4, 0.3, 0.26],
        [0.36, 0.32, 0.27],
      ],
    },
  },
};

export default mockData;
