import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple
import config

class DataProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
    
    def extract_landmarks(self, frame) -> Tuple[List[float], mp.solutions.pose.PoseLandmark]:
        """Extract pose landmarks from a frame"""
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None, None
            
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks, results.pose_landmarks
    
    def process_video(self, video_path: str, timestamps: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Process video and extract sequences based on timestamps"""
        sequences = []
        labels = []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        current_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = frame_count / fps
            current_label = None
            
            for timestamp in timestamps:
                if timestamp["start"] <= current_time <= timestamp["end"]:
                    current_label = timestamp["label"]
                    break
            
            if current_label is not None:
                landmarks, _ = self.extract_landmarks(frame)
                if landmarks:
                    current_sequence.append(landmarks)
                    
                    if len(current_sequence) == config.SEQUENCE_LENGTH:
                        sequences.append(current_sequence)
                        labels.append(current_label)
                        current_sequence = current_sequence[1:]
            else:
                current_sequence = []
            
            frame_count += 1
        
        cap.release()
        return np.array(sequences), np.array(labels)