from backend.src.utils.preprocessing import predict_video

def analyze_video(video_path):
    """
    Wrapper function to call prediction from preprocessing.
    """
    try:
        prediction = predict_video(video_path)
        # You can customize what you return here, e.g. label or score
        return prediction
    except Exception as e:
        print(f"Error processing video: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    test_video = "D:/Project/DF-Analysis/backend/data/orig.mp4"
    result = analyze_video(test_video)
    if result is not None:
        print("Prediction:", result)
    else:
        print("Failed to get prediction")
