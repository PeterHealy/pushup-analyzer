from data.data_collector import VideoCollector
from data.data_processor import DataProcessor
from models.lstm_model import PushupModel
from sklearn.model_selection import train_test_split
import config

def parse_timestamp(time_str):
    """Convert various timestamp formats to seconds
    
    Supports formats:
    - Seconds: "37"
    - MM:SS: "1:40"
    - HH:MM:SS: "1:23:45"
    
    Returns:
        Integer number of seconds
    """
    if time_str.isdigit():
        return int(time_str)
    
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS format
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS format
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("Invalid timestamp format. Use seconds (37) or MM:SS (1:40) or HH:MM:SS (1:23:45)")

def train_model(video_data):
    """Train the push-up form analysis model
    
    Args:
        video_data: Dictionary containing:
            - urls: List of YouTube URLs
            - segments: List of time segments with labels
                Each segment has start, end, and label (1 = good form, 0 = bad form)
    """
    # Convert timestamp strings to seconds
    for segments in video_data["segments"]:
        for segment in segments:
            segment["start"] = parse_timestamp(str(segment["start"]))
            segment["end"] = parse_timestamp(str(segment["end"]))
    
    # Initialize components
    collector = VideoCollector()
    processor = DataProcessor()
    model = PushupModel()

    # Download videos if URLs provided
    if "urls" in video_data:
        paths = collector.download_videos(video_data["urls"])
        video_data["videos"] = [
            {"path": path, "segments": segments}
            for path, segments in zip(paths, video_data["segments"])
        ]
    
    # Process videos and collect training data
    all_sequences = []
    all_labels = []
    
    for video in video_data["videos"]:
        sequences, labels = processor.process_video(
            video["path"],
            video["segments"]
        )
        all_sequences.extend(sequences)
        all_labels.extend(labels)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        all_sequences,
        all_labels,
        test_size=0.2  # Use 20% for validation
    )
    
    # Train and return training history
    history = model.train(X_train, y_train, X_val, y_val)
    return history

if __name__ == "__main__":
    # Example video data structure
    video_data = {
        "urls": [
            "https://youtube.com/watch?v=xxx",  # Good form example
            "https://youtube.com/watch?v=yyy"   # Bad form example
        ],
        "segments": [
            [
                {"start": "37", "end": "1:40", "label": 1}  # Good form segment
            ],
            [
                {"start": "2:15", "end": "2:45", "label": 0}  # Bad form segment
            ]
        ]
    }
    
    history = train_model(video_data)