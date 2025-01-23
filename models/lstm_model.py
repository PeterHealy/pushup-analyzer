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
        self.model = (
            self._create_model()
            if not os.path.exists(config.MODEL_PATH)
            else load_model(config.MODEL_PATH)
        )
    
    def _create_model(self):
        """Create LSTM model for pose sequence analysis"""
        model = Sequential([
            LSTM(
                config.LSTM_UNITS,
                return_sequences=True,
                input_shape=(config.SEQUENCE_LENGTH, config.N_FEATURES)
            ),
            Dropout(config.DROPOUT_RATE),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE
        )
        
        self.model.save(config.MODEL_PATH)
        return history
    
    def predict(self, sequence):
        """Make prediction on a sequence"""
        return self.model.predict(sequence)[0][0]