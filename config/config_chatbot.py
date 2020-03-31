import os
import pickle

CHATBOT_PATH = '/content/drive/My Drive/Transformer_Chatbot'
MOVIE_LINES = os.path.join(CHATBOT_PATH, 'movie_lines.txt')
MOVIE_CONVERSATIONS = os.path.join(CHATBOT_PATH, 'movie_conversations.txt')

# Maximum number of samples to preprocess
MAX_SAMPLES = 50000

# Maximum sentence length
MAX_LENGTH = 40

TOKENIZER = os.path.join(CHATBOT_PATH, 'chatbot_tokenizer.pkl')

with open(TOKENIZER, 'rb') as f:
    tokenizer = pickle.load(f)
PES = os.path.join(CHATBOT_PATH, 'pes.pkl')

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

BATCH_SIZE = 64
BUFFER_SIZE = 20000
MODEL_SIZE = 256
NUM_LAYERS = 2
NUM_HEADS = 8
#H = 8
EPOCHS = 20
DROPOUT = 0.1
UNITS = 512

ckpt_path = os.path.join(CHATBOT_PATH, 'ckpt_transformer_chatbot')
file_path = os.path.join(ckpt_path, "chatbot_transformer-epoch{epoch:03d}.hdf5")
model_weights = os.path.join(ckpt_path, "chatbot_transformer.h5")
