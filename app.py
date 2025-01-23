import cv2
from models.lstm_model import PushupModel
from data.data_processor import DataProcessor
from utils.visualization import PoseVisualizer
import config
from datetime import datetime
import os

class PushupAnalyzer:
    def __init__(self):
        self.model = PushupModel()
        self.processor = DataProcessor()
        self.visualizer = PoseVisualizer()
        self.frame_sequence = []
        
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
        
        prediction = self.model.predict([sequence_landmarks])
        
        if prediction < 0.3:
            return "Poor form - Major corrections needed"
        elif prediction < 0.7:
            return "Fair form - Minor adjustments recommended"
        else:
            return "Good form!"

    def run(self):
        """Run the real-time analysis"""
        # Check if model exists
        model_exists = os.path.exists(config.MODEL_PATH)

        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break
                
            # Get landmarks
            landmarks, pose_landmarks = self.processor.extract_landmarks(frame)
            
            if landmarks:
                # Add landmarks to sequence
                self.frame_sequence.append(landmarks)
                # Keep only the last SEQUENCE_LENGTH frames
                if len(self.frame_sequence) > config.SEQUENCE_LENGTH:
                    self.frame_sequence.pop(0)
                
                # Only run form analysis if model exists
                if model_exists and len(self.frame_sequence) == config.SEQUENCE_LENGTH:
                    form_feedback = self.analyze_form(self.frame_sequence)
                    if form_feedback:
                        # Draw feedback on frame
                        cv2.putText(frame, form_feedback, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw landmarks using visualizer
            if pose_landmarks:
                frame = self.visualizer.draw_pose_landmarks(frame, pose_landmarks)
                
            cv2.imshow('Push-up Form Analysis', frame)
            
            # Break loop with 'q' key or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is ESC key
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = PushupAnalyzer()
    analyzer.run()