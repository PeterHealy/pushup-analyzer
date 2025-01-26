import cv2
from models.lstm_model import PushupModel
from data.data_processor import DataProcessor
from utils.visualization import PoseVisualizer
import config
from datetime import datetime
import os
import numpy as np
import time

class PushupAnalyzer:
    def __init__(self):
        self.model = PushupModel()
        self.processor = DataProcessor()
        self.visualizer = PoseVisualizer()
        self.frame_sequence = []
        self.is_analyzing = False
        
    def analyze_form(self, frame_sequence):
        """Analyze push-up form from a sequence of frames"""
        if len(frame_sequence) != config.SEQUENCE_LENGTH:
            return None
            
        sequence_landmarks = []
        for frame in frame_sequence:
            landmarks, _ = self.processor.extract_landmarks(frame)
            if landmarks:
                sequence_landmarks.append(landmarks)
            else:
                return None
        
        # Convert to numpy array with correct shape
        sequence_landmarks = np.array(sequence_landmarks)
        sequence_landmarks = sequence_landmarks.reshape(1, config.SEQUENCE_LENGTH, config.N_FEATURES)
        
        prediction = self.model.predict(sequence_landmarks)
        
        if prediction < 0.3:
            return "Poor form - Major corrections needed"
        elif prediction < 0.7:
            return "Fair form - Minor adjustments recommended"
        else:
            return "Good form!"

    def countdown(self, frame, count):
        """Display countdown on frame"""
        height, width = frame.shape[:2]
        font_scale = 4.0
        thickness = 4
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        text = str(count)
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate position to center text
        x = (width - text_width) // 2
        y = (height + text_height) // 2
        
        # Draw countdown number
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness)
        return frame

    def analyze_video_file(self, video_path):
        """Analyze a pre-recorded video file"""
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
            
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties for output
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create output video writer
        output_path = video_path.rsplit('.', 1)[0] + '_analyzed.mp4'
        out = cv2.VideoWriter(output_path, 
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            landmarks, pose_landmarks = self.processor.extract_landmarks(frame)
            
            if landmarks:
                self.frame_sequence.append(frame)
                if len(self.frame_sequence) > config.SEQUENCE_LENGTH:
                    self.frame_sequence.pop(0)
                
                if len(self.frame_sequence) == config.SEQUENCE_LENGTH:
                    form_feedback = self.analyze_form(self.frame_sequence)
                    if form_feedback:
                        cv2.putText(frame, form_feedback, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if pose_landmarks:
                frame = self.visualizer.draw_pose_landmarks(frame, pose_landmarks)
            
            out.write(frame)
            
            cv2.imshow('Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Analysis complete! Output saved to: {output_path}")

    def run_live(self):
        """Run real-time analysis using webcam"""
        model_exists = os.path.exists(config.MODEL_PATH)

        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        print("Camera opened successfully. Position yourself and press 'a' to start analyzing.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break
                
            # Get landmarks
            landmarks, pose_landmarks = self.processor.extract_landmarks(frame)
            
            if landmarks:
                if self.is_analyzing:
                    # Add frame to sequence
                    self.frame_sequence.append(frame)
                    if len(self.frame_sequence) > config.SEQUENCE_LENGTH:
                        self.frame_sequence.pop(0)
                    
                    # Only analyze if we have enough frames and model exists
                    if model_exists and len(self.frame_sequence) == config.SEQUENCE_LENGTH:
                        form_feedback = self.analyze_form(self.frame_sequence)
                        if form_feedback:
                            cv2.putText(frame, form_feedback, (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw landmarks using visualizer
            if pose_landmarks:
                frame = self.visualizer.draw_pose_landmarks(frame, pose_landmarks)
            
            # Add status text
            status = "Analyzing..." if self.is_analyzing else "Press 'a' to start/stop analysis"
            cv2.putText(frame, status, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            cv2.imshow('Push-up Form Analysis', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC to quit
                break
            elif key == ord('a'):  # 'a' to toggle analysis
                if not self.is_analyzing:
                    # Start countdown
                    for i in range(3, 0, -1):
                        start_time = time.time()
                        while time.time() - start_time < 1:  # Show each number for 1 second
                            ret, frame = cap.read()
                            if ret:
                                landmarks, pose_landmarks = self.processor.extract_landmarks(frame)
                                if pose_landmarks:
                                    frame = self.visualizer.draw_pose_landmarks(frame, pose_landmarks)
                                frame = self.countdown(frame, i)
                                cv2.imshow('Push-up Form Analysis', frame)
                                if cv2.waitKey(1) & 0xFF == 27:  # Allow ESC to cancel countdown
                                    break
                    
                    self.is_analyzing = True
                    self.frame_sequence = []  # Clear sequence when starting
                    print("Analysis started")
                else:
                    self.is_analyzing = False
                    self.frame_sequence = []  # Clear sequence when stopping
                    print("Analysis stopped")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = PushupAnalyzer()
    
    print("Choose analysis mode:")
    print("1. Live webcam analysis")
    print("2. Video file analysis")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        analyzer.run_live()
    elif choice == "2":
        video_path = input("Enter the path to your video file: ")
        analyzer.analyze_video_file(video_path)
    else:
        print("Invalid choice. Please run again and select 1 or 2.")