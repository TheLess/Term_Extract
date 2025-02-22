from pathlib import Path

class Config:
    BASE_MODEL_NAME = 'bert-base-chinese'
    MAX_LENGTH = 512
    DATA_PATH = Path('data/translations.xlsx')
    TERM_DB_PATH = Path('data/term_database.json')
    TRAIN_RATIO = 0.8
    LORA_R = 8
    LORA_ALPHA = 32
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
