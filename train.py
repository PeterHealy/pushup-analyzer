from data.data_collector import VideoCollector
from data.data_processor import DataProcessor
from models.lstm_model import PushupModel
from sklearn.model_selection import train_test_split
import config
import numpy as np

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
        # Modified print statements
        print(f"Video sequences shape: {sequences.shape if isinstance(sequences, np.ndarray) else 'empty'}")
        print(f"Video labels shape: {labels.shape if isinstance(labels, np.ndarray) else 'empty'}")
        
        all_sequences.extend(sequences)
        all_labels.extend(labels)
    
    # Convert lists to numpy arrays after collecting all data
    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)
    
    # Print final shapes
    print("\nFinal shapes:")
    print("All sequences shape:", all_sequences.shape)
    print("All labels shape:", all_labels.shape)
    print("Number of sequences:", len(all_sequences))
    print("Number of labels:", len(all_labels))
    
    if len(all_sequences) > 0:
        print("Single sequence shape:", all_sequences[0].shape)

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
            "https://www.youtube.com/watch?v=WDIpL0pjun0",  
            "https://www.youtube.com/watch?v=ZYuocE5AMgU",
            "https://www.youtube.com/watch?v=-T64FLsJnAU",
            "https://www.youtube.com/watch?v=iIa2-uVHzM0",
            "https://www.youtube.com/watch?v=IODxDxX7oi4"   
        ],
        "segments": [
            [
                {"start": "00", "end": "03", "label": 1},  
                {"start": "04", "end": "06", "label": 1},
                {"start": "07", "end": "10", "label": 1},  
                {"start": "10", "end": "13", "label": 1}  
            ],
            [
                {"start": "00", "end": "02", "label": 1},
                {"start": "1:48", "end": "1:54", "label": 1},
                {"start": "1:54", "end": "1:59", "label": 1},
                {"start": "1:59", "end": "2:04", "label": 1}
            ],
            [
                {"start": "1:59", "end": "2:04", "label": 1},
                {"start": "2:04", "end": "2:10", "label": 1},
                {"start": "2:52", "end": "2:58", "label": 1},
                {"start": "3:01", "end": "3:05", "label": 1}
            ],
            [
                # Good form segments
                {"start": "31", "end": "35", "label": 1},
                {"start": "1:40", "end": "1:43", "label": 1},
                {"start": "7:30", "end": "7:35", "label": 1},
                {"start": "8:01", "end": "8:06", "label": 1},
                # Bad form segments
                {"start": "48", "end": "59", "label": 0},
                {"start": "1:00", "end": "1:05", "label": 0},
                {"start": "2:10", "end": "2:12", "label": 0},
                {"start": "7:24", "end": "7:27", "label": 0}
            ],
            [
                # Good form segments
                {"start": "32", "end": "40", "label": 1},
                {"start": "1:09", "end": "1:12", "label": 1},
                {"start": "1:55", "end": "2:03", "label": 1},
                {"start": "2:52", "end": "3:01", "label": 1},
                {"start": "3:02", "end": "3:12", "label": 1},
                {"start": "3:13", "end": "3:16", "label": 1},
                # Bad form segments
                {"start": "41", "end": "45", "label": 0},
                {"start": "1:42", "end": "1:53", "label": 0},
                {"start": "2:06", "end": "2:12", "label": 0},
                {"start": "2:14", "end": "2:20", "label": 0},
                {"start": "2:44", "end": "2:51", "label": 0}
            ]
        ]
    }
    
    history = train_model(video_data)