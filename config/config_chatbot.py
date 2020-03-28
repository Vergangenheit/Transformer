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

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

BATCH_SIZE = 64
BUFFER_SIZE = 20000
MODEL_SIZE = 256
NUM_LAYERS = 4
H = 8
EPOCHS = 20

checkpoints_en = os.path.join(CHATBOT_PATH, 'checkpoints_transformer/encoder')
checkpoints_de = os.path.join(CHATBOT_PATH, 'checkpoints_transformer/decoder')