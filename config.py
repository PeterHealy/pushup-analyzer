# Configuration settings
MODEL_PATH = 'models/saved/pushup_model.h5'
DATA_DIR = 'data/training_data'
SEQUENCE_LENGTH = 30
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Model parameters
N_FEATURES = 99  # 33 landmarks * 3 coordinates
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 20