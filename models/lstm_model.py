try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except ImportError:
    print("Error importing TensorFlow. Installed version:", tf.__version__)
import os
import config

class PushupModel:
    def __init__(self):
        # Either load existing model or create a new one
        self.model = (
            self._create_model()  # Create new if no saved model exists
            if not os.path.exists(config.MODEL_PATH)
            else load_model(config.MODEL_PATH)  # Load existing model if available
        )
    
    def _create_model(self):
        """Create LSTM model for pose sequence analysis
        
        Architecture:
        1. LSTM layer with return sequences (processes time series data)
        2. Dropout layer to prevent overfitting
        3. Second LSTM layer
        4. Dense layer with ReLU activation
        5. Output layer with sigmoid for binary classification
        """
        model = Sequential([
            LSTM(
                config.LSTM_UNITS,
                return_sequences=True,  # Return full sequence for next LSTM layer
                input_shape=(config.SEQUENCE_LENGTH, config.N_FEATURES)  # (timesteps, features)
            ),
            Dropout(config.DROPOUT_RATE),  # Prevent overfitting
            LSTM(32),  # Second LSTM layer for deeper feature extraction
            Dense(16, activation='relu'),  # Dense layer for final feature processing
            Dense(1, activation='sigmoid')  # Output layer: 1 = good form, 0 = bad form
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',  # Binary classification loss
            metrics=['accuracy']  # Track accuracy during training
        )
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model on push-up sequences
        
        Args:
            X_train: Training sequences of pose landmarks
            y_train: Training labels (1 = good form, 0 = bad form)
            X_val: Validation sequences
            y_val: Validation labels
        """
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE
        )
        
        # Save the trained model
        self.model.save(config.MODEL_PATH)
        return history
    
    def predict(self, sequence):
        """Make prediction on a sequence of poses
        
        Args:
            sequence: List of pose landmarks representing a push-up
            
        Returns:
            Float between 0 and 1 (1 = good form, 0 = bad form)
        """
        return self.model.predict(sequence)[0][0]