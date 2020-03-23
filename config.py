import os

# Mode can be either 'train' or 'infer'
# Set to 'infer' will skip the training
MODE = 'train'
URL = 'http://www.manythings.org/anki/fra-eng.zip'
FILENAME = os.path.join('/content/drive/My Drive/Neural_Machine_Translation','fra-eng.zip')
BATCH_SIZE = 64
NUM_EPOCHS = 15
MODEL_SIZE = 128
EN_TOKENIZER = os.path.join('/content/drive/My Drive/Neural_Machine_Translation', 'en_tokenizer.pkl')
FR_TOKENIZER = os.path.join('/content/drive/My Drive/Neural_Machine_Translation', 'fr_tokenizer.pkl')
H = 8
NUM_LAYERS = 4
