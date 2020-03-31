import tensorflow_datasets as tfds
import tensorflow as tf
from data_processing.chatbot import dm_chatbot as dm
import config.config_chatbot as config
import pickle


def tokenize(questions: list, answers: list, save_tok=True):
    # Build tokenizer using tfds for both questions and answers
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2 ** 13)
    # save tokenizer
    if save_tok:
        with open(config.TOKENIZER, 'wb') as f:
            pickle.dump(tokenizer, f)
    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2


# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs):
    # load tokenizer
    with open(config.TOKENIZER, 'rb') as f:
        tokenizer = pickle.load(f)

    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    VOCAB_SIZE = tokenizer.vocab_size + 2
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check tokenized sentence max length
        if len(sentence1) <= config.MAX_LENGTH and len(sentence2) <= config.MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs, maxlen=config.MAX_LENGTH,
                                                                     padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs, maxlen=config.MAX_LENGTH,
                                                                      padding='post')

    return tokenized_inputs, tokenized_outputs


def create_dataset(questions, answers):
    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    dataset = tf.data.Dataset.from_tensor_slices((
        {'inputs': questions,
         'dec_inputs': answers[:, :-1]},
        {'outputs': answers[:, 1:]}
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(config.BUFFER_SIZE)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


