from data.data_collector import VideoCollector
from data.data_processor import DataProcessor
from models.lstm_model import PushupModel
from sklearn.model_selection import train_test_split
import config

def parse_timestamp(time_str):
    """Convert various timestamp formats to seconds"""
    if time_str.isdigit():
        return int(time_str)
    
    parts = time_str.split(':')
    if len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("Invalid timestamp format. Use seconds (37) or MM:SS (1:40) or HH:MM:SS (1:23:45)")

def train_model(video_data):
    # Convert timestamp strings to seconds
    for segments in video_data["segments"]:
        for segment in segments:
            segment["start"] = parse_timestamp(str(segment["start"]))
            segment["end"] = parse_timestamp(str(segment["end"]))
    
    collector = VideoCollector()
    processor = DataProcessor()
    model = PushupModel()

    if "urls" in video_data:
        paths = collector.download_videos(video_data["urls"])
        video_data["videos"] = [
            {"path": path, "segments": segments}
            for path, segments in zip(paths, video_data["segments"])
        ]
    
    all_sequences = []
    all_labels = []
    
    for video in video_data["videos"]:
        sequences, labels = processor.process_video(
            video["path"],
            video["segments"]
        )
        all_sequences.extend(sequences)
        all_labels.extend(labels)
    
    X_train, X_val, y_train, y_val = train_test_split(
        all_sequences,
        all_labels,
        test_size=0.2
    )
    
    history = model.train(X_train, y_train, X_val, y_val)
    return history

if __name__ == "__main__":
    video_data = {
        "urls": [
            "https://youtube.com/watch?v=xxx",
            "https://youtube.com/watch?v=yyy"
        ],
        "segments": [
            [
                {"start": "37", "end": "1:40", "label": 1}
            ],
            [
                {"start": "2:15", "end": "2:45", "label": 0}
            ]
        ]
    }
    
    history = train_model(video_data)