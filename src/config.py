MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "cardiffnlp/tweet_sentiment_multilingual"
DATASET_SUBSET = "english"


LEARNING_RATE = 2e-5      # Standard for BERT models
BATCH_SIZE = 32         # Batch size for training
NUM_EPOCHS = 10            # Number of training epochs
WEIGHT_DECAY = 0.01       # Regularization
MAX_LENGTH = 128          # Maximum sequence length
SEED = 42

LABEL_NAMES = ["negative", "neutral", "positive"]
NUM_LABELS = len(LABEL_NAMES)

DATA_SAVE_PATH = "./data/sentiment_dataset"

CHECKPOINT_DIR = "./models/checkpoints"

MODEL_SAVE_PATH = "./models/distilbert-sentiment"