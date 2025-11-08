const mockData = {
  verdict: {
    status: 'fake',
    confidence: 0.82,
    summary:
      'The uploaded media is very likely to be a deepfake based on frame-level analysis and artifact detection.',
    text: 'Warning: High likelihood of deepfake content detected.',
  },

  deepfake: {
    "label": "FAKE",
  "confidence": 0.92,
  "metadata": {
    "duration_sec": 13.4,
    "frames_analyzed": 160,
    
  },
  "top_frames": [
    {
      "timestamp_sec": 2.3,
      "frame_index": 1,
      "fake_confidence": 0.97,
      "url": "/assets/frames/frame1.png"
    },
    {
      "timestamp_sec": 4.6,
      "frame_index": 2,
      "fake_confidence": 0.95,
      "url": "/assets/frames/frame2.png"
    },
    {
      "timestamp_sec": 7.1,
      "frame_index": 3,
      "fake_confidence": 0.93,
      "url": "/assets/frames/frame3.png"
    },
    {
      "timestamp_sec": 9.2,
      "frame_index": 4,
      "fake_confidence": 0.80,
      "url": "/assets/frames/frame4.png"
    },
    {
      "timestamp_sec": 12.0,
      "frame_index": 5,
      "fake_confidence": 0.88,
      "url": "/assets/frames/frame5.png"
    }
  ]
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
  file: "eddiewoo.mp4",
  models: {
    //video
    rafdb: {
      video: "eddiewoo.mp4",
      duration_s: 21.59,
      dominant_emotion: "Neutral",
      timeline: [
        { t: 0.03, emo: "Uncertain", conf: 0.0 },
        { t: 1.1, emo: "Neutral", conf: 0.559 },
        { t: 2.17, emo: "Sad", conf: 0.973 },
        { t: 3.23, emo: "Sad", conf: 0.851 },
        { t: 4.3, emo: "Neutral", conf: 0.985 },
        { t: 5.36, emo: "Happy", conf: 0.998 },
        { t: 6.43, emo: "Neutral", conf: 0.714 },
        { t: 7.5, emo: "Angry", conf: 0.753 },
        { t: 8.56, emo: "Neutral", conf: 0.693 },
        { t: 9.63, emo: "Happy", conf: 0.999 },
        { t: 10.69, emo: "Sad", conf: 0.922 },
        { t: 11.76, emo: "Happy", conf: 0.999 },
        { t: 12.83, emo: "Neutral", conf: 0.635 },
        { t: 13.89, emo: "Surprise", conf: 0.604 },
        { t: 14.96, emo: "Neutral", conf: 0.648 },
        { t: 16.02, emo: "Surprise", conf: 0.678 },
        { t: 17.09, emo: "Angry", conf: 0.962 },
        { t: 18.16, emo: "Sad", conf: 0.954 },
        { t: 19.22, emo: "Happy", conf: 0.999 },
        { t: 20.29, emo: "Neutral", conf: 0.997 }
      ]
    },
    //text
    goemotions: {
      //summary
      summary: {
        video: "eddiewoo",
        dominant_emotion: "neutral",
        emotion_distribution: {
          neutral: 0.3342773570255522,
          admiration: 0.09745002678346508,
          approval: 0.014985758624044297,
          annoyance: 0.03850042760161776,
          love: 0.020235306109691834,
          curiosity: 0.007862982991416608
        }
      },
      //graph
      linegraph: {
        video: "eddiewoo",
        top_emotions: ["neutral", "admiration", "annoyance"],
        data: [
          { time: 0.0, neutral: 0.3374, admiration: 0.1262, annoyance: 0.0 },
          { time: 2.0, neutral: 0.3537, admiration: 0.1034, annoyance: 0.064 },
          { time: 6.0, neutral: 0.3683, admiration: 0.0985, annoyance: 0.0 },
          { time: 11.0, neutral: 0.3313, admiration: 0.0867, annoyance: 0.0756 },
          { time: 12.0, neutral: 0.4471, admiration: 0.0911, annoyance: 0.0705 },
          { time: 12.5, neutral: 0.2937, admiration: 0.0968, annoyance: 0.0632 },
          { time: 13.5, neutral: 0.3399, admiration: 0.0882, annoyance: 0.0732 },
          { time: 15.0, neutral: 0.2034, admiration: 0.1003, annoyance: 0.0 },
          { time: 20.5, neutral: 0.3337, admiration: 0.0859, annoyance: 0.0 }
        ]
      },

      //table
      tabledata: {
        video: "eddiewoo",
        segments: [
          {
            start: 0.0,
            end: 2.0,
            text: "one a the golden ratio is a mathematical reality",
            emotions: ["neutral", "admiration", "approval"],
            intensities: [0.3374, 0.1262, 0.0708],
            confidence: 0.3374
          },
          {
            start: 2.0,
            end: 6.0,
            text: "one a that like facts you can find everywhere",
            emotions: ["neutral", "admiration", "annoyance"],
            intensities: [0.3537, 0.1034, 0.064],
            confidence: 0.3537
          },
          {
            start: 6.0,
            end: 11.0,
            text: "from the changes of your fingers to the pillars of the parthenon",
            emotions: ["neutral", "admiration", "approval"],
            intensities: [0.3683, 0.0985, 0.064],
            confidence: 0.3683
          },
          {
            start: 11.0,
            end: 12.0,
            text: "that's why even at a party of 5000 people",
            emotions: ["neutral", "admiration", "annoyance"],
            intensities: [0.3313, 0.0867, 0.0756],
            confidence: 0.3313
          },
          {
            start: 12.0,
            end: 12.5,
            text: "people",
            emotions: ["neutral", "admiration", "annoyance"],
            intensities: [0.4471, 0.0911, 0.0705],
            confidence: 0.4471
          },
          {
            start: 12.5,
            end: 13.0,
            text: "that's even at a poverty of 50 5000 penplelf",
            emotions: ["neutral", "admiration", "annoyance"],
            intensities: [0.2937, 0.0968, 0.0632],
            confidence: 0.2937
          },
          {
            start: 13.5,
            end: 14.5,
            text: "away that's why even at a party of 5000 people a",
            emotions: ["neutral", "admiration", "annoyance"],
            intensities: [0.3399, 0.0882, 0.0732],
            confidence: 0.3399
          },
          {
            start: 15.0,
            end: 19.5,
            text: "i'm proud to declared love mathematics",
            emotions: ["neutral", "love", "admiration"],
            intensities: [0.2034, 0.1821, 0.1003],
            confidence: 0.2034
          },
          {
            start: 20.5,
            end: 21.5,
            text: "will seal a we",
            emotions: ["neutral", "admiration", "curiosity"],
            intensities: [0.3337, 0.0859, 0.0708],
            confidence: 0.3337
          }
        ]
      }
    },

    //audio
    ravdess: {
      // final
      results: {
        file: "/home/ampm/projects/DF-Analysis/backend/data/emotion/input/processed/eddiewoo_mel.npy",
        predicted_emotion: "fearful",
        probabilities: {
          neutral: 0.01338341273367405,
          calm: 0.023433633148670197,
          happy: 0.010573669336736202,
          sad: 0.13376663625240326,
          angry: 0.08338312059640884,
          fearful: 0.33970537781715393,
          disgust: 0.08353643864393234,
          surprised: 0.31221774220466614
        }
      },
      //waveform
      waveform: {
        meta: {
          sr: 16000,
          hop_length: 512,
          frame_duration: 0.032,
          original_num_frames: 1726,
          export_num_points: 30
        },
        axes: { x_label: "Time (s)", y_label: "Normalized Amplitude" },
        frames: {
          time: [
            0.0, 1.903448275862069, 3.806896551724138, 5.710344827586207,
            7.613793103448276, 9.517241379310345, 11.420689655172414,
            13.324137931034484, 15.227586206896552, 17.131034482758622,
            19.03448275862069, 20.93793103448276, 22.841379310344827,
            24.744827586206895, 26.648275862068967, 28.551724137931036,
            30.455172413793104, 32.358620689655176, 34.262068965517244,
            36.16551724137931, 38.06896551724138, 39.97241379310345,
            41.87586206896552, 43.779310344827586, 45.682758620689654,
            47.58620689655172, 49.48965517241379, 51.393103448275866,
            53.296551724137935, 55.2
          ],
          envelope: [
            0.12832821905612946, 0.12192441266158531, 0.23499565751388152,
            0.12099218908055075, 0.12458466635695814, 0.7064206826275738,
            0.12784744388070599, 0.14732397321997015, 0.1281187380182332,
            0.2397258009376196, 0.17938262481113898, 0.12824866792251324,
            0.16708068652399738, 0.12738548116437326, 0.12820418179035187,
            0.12805371603061413, 0.12784085242912685, 0.1173230365946376,
            0.12120268159899283, 0.7070475364553592, 0.12680738033919492,
            0.5389095174855032, 0.4608906158085512, 0.12503449413283108,
            0.31190624010974066, 0.1254081171134457, 0.11782285879398217,
            0.09759997550783592, 0.24936739072716094, 0.1777241826057434
          ]
        }
      }
    }
  }
},   
};

export default mockData;