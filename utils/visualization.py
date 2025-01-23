import cv2
import mediapipe as mp
import numpy as np

class PoseVisualizer:
    def __init__(self):
        """Initialize visualization settings"""
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Define colors (BGR format)
        self.text_color = (0, 0, 255)  # Red
        self.landmark_color = (0, 255, 0)  # Green
        self.connection_color = (255, 255, 255)  # White
        
        # Drawing specs for pose landmarks
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            color=self.landmark_color,
            thickness=2,
            circle_radius=2
        )
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=self.connection_color,
            thickness=2
        )

    def draw_pose_landmarks(self, frame, pose_landmarks):
        """Draw pose landmarks and connections on frame"""
        self.mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.landmark_drawing_spec,
            connection_drawing_spec=self.connection_drawing_spec
        )
        return frame

    def draw_feedback(self, frame, feedback_text, position=(30, 30)):
        """Draw feedback text on frame"""
        # Add background rectangle for better text visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            feedback_text, font, scale, thickness
        )
        
        # Draw background rectangle
        padding = 10
        cv2.rectangle(
            frame,
            (position[0] - padding, position[1] - text_height - padding),
            (position[0] + text_width + padding, position[1] + padding),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            feedback_text,
            position,
            font,
            scale,
            self.text_color,
            thickness
        )
        return frame

    def draw_angles(self, frame, angles_dict, start_y=100):
        """Draw angle measurements on frame"""
        y_position = start_y
        for joint, angle in angles_dict.items():
            text = f"{joint}: {angle:.1f}°"
            self.draw_feedback(frame, text, (30, y_position))
            y_position += 30
        return frame

    def draw_performance_metrics(self, frame, metrics, position=(30, 200)):
        """Draw performance metrics on frame"""
        y_position = position[1]
        for metric, value in metrics.items():
            text = f"{metric}: {value}"
            self.draw_feedback(frame, text, (position[0], y_position))
            y_position += 30
        return frame

    def create_debug_view(self, frame, pose_landmarks, angles=None, metrics=None):
        """Create a debug view with all visualization elements"""
        debug_frame = frame.copy()
        
        # Draw pose landmarks
        if pose_landmarks:
            self.draw_pose_landmarks(debug_frame, pose_landmarks)
        
        # Draw angles if provided
        if angles:
            self.draw_angles(debug_frame, angles)
        
        # Draw metrics if provided
        if metrics:
            self.draw_performance_metrics(debug_frame, metrics)
            
        return debug_frame