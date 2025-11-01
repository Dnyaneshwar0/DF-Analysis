# DeepFake Reverse Engineering & Emotional Analysis

### Directory Structure:
```
<to be updated>
```

****Make sure to replace placeholders and update dir struc as you commit changes***

## Backend Integration Update [1/11/25]: 

### Run Main App with:
```
python app.py
```

### Example CURL req to backend like:
```
curl -X POST -F "file=@D:/Projects/DF-Analysis/backend/data/detection/orig.mp4" http://localhost:5000/detect/analyze


curl -X POST -F "file=@D:/Projects/DF-Analysis/backend/data/detection/manip.mp4" http://localhost:5000/reveng/analyze
```


# *Imp info regarding current df_detect.keras model:

For this error:
```
ValueError: Exception encountered when calling TimeDistributed.call().

Cannot convert '15' to a shape.

Arguments received by TimeDistributed.call():
  • args=('<KerasTensor shape=(None, 15, 96, 96, 3), dtype=float32, sparse=False, ragged=False, name=input_layer>',)
  • kwargs={'mask': 'None'}
```

🧠 Why this happens

When you saved your model in Colab using:
```
model.save('deepfake_method_classifier.keras')
```

and later reloaded it in inference using:
```
tf.keras.models.load_model(path, compile=False)
```

the custom `TimeDistributed(MobileNetV2(...))` layer often fails to rebuild its input shape metadata cleanly across TF versions.

It ends up interpreting your `Input(shape=(frames, img_size, img_size, 3))` as
`Input(shape=frames)` — i.e., it lost the “tuple-ness” and treats `15` as a scalar.

This is a known TensorFlow serialization quirk for nested models with `TimeDistributed`.