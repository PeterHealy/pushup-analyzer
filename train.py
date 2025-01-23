from data.data_collector import VideoCollector
from data.data_processor import DataProcessor
from models.lstm_model import PushupModel
from sklearn.model_selection import train_test_split
import config

def train_model(video_data):
    """Train the push-up form analysis model"""
    # Initialize components
    collector = VideoCollector()
    processor = DataProcessor()
    model = PushupModel()
    
    # Download videos if URLs are provided
    if "urls" in video_data:
        paths = collector.download_videos(video_data["urls"])
        # Update video_data with downloaded paths
        video_data["videos"] = [
            {"path": path, "segments": segments}
            for path, segments in zip(paths, video_data["segments"])
        ]
    
    # Process videos and collect sequences
    all_sequences = []
    all_labels = []
    
    for video in video_data["videos"]:
        sequences, labels = processor.process_video(
            video["path"],
            video["segments"]
        )
        all_sequences.extend(sequences)
        all_labels.extend(labels)
    
    # Split data for training
    X_train, X_val, y_train, y_val = train_test_split(
        all_sequences,
        all_labels,
        test_size=0.2
    )
    
    # Train model
    history = model.train(X_train, y_train, X_val, y_val)
    return history

if __name__ == "__main__":
    # Example usage
    video_data = {
        "urls": [
            "https://youtube.com/...",  # Add your YouTube URLs
        ],
        "segments": [
            [
                {"start": 10, "end": 15, "label": 1},
                {"start": 20, "end": 25, "label": 0}
            ]
        ]
    }
    
    history = train_model(video_data)
    print("Training completed successfully!")