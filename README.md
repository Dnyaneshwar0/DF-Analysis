# DeepFake Reverse Engineering & Emotional Analysis

### Directory Structure:
```
<to be updated>
```

****Make sure to replace placeholders and update dir struc as you commit changes***

## Backend Integration Update: 
### Run indiv services with
```
python df_detect.py     # for detection

python df_revEng.py     # for rev Eng
```

### Run Main App with:
```
python app.py
```

### CURL req to backend like:
```
curl -F "video=@D:\Projects\DF-Analysis\backend\data\detection\manip.mp4" http://localhost:5000/deepfake/analyze

curl -F "video=@D:\Projects\DF-Analysis\backend\data\detection\orig.mp4" http://localhost:5000/reverse/analyze
```
