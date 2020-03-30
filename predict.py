import config.config_chatbot as config
import pickle
from data_processing.chatbot.dm_chatbot import preprocess_sentence
from models.trans_model import Decoder, Encoder
import tensorflow as tf


def evaluate(pes, sentence=None):
    sentence = preprocess_sentence(sentence)
    with open(config.TOKENIZER, 'rb') as f:
        tokenizer = pickle.load(f)
    sentence = tf.expand_dims(
        config.START_TOKEN + tokenizer.encode(sentence) + config.END_TOKEN, axis=0)
    de_input = tf.expand_dims(config.START_TOKEN, 0)

    # load encoder and decoder
    encoder_checkpoint = tf.train.latest_checkpoint(config.checkpoints_en)
    decoder_checkpoint = tf.train.latest_checkpoint(config.checkpoints_de)
    encoder = Encoder(config.VOCAB_SIZE, config.MODEL_SIZE, config.NUM_LAYERS, config.H)
    decoder = Decoder(config.VOCAB_SIZE, config.MODEL_SIZE, config.NUM_LAYERS, config.H)
    if encoder_checkpoint is not None and decoder_checkpoint is not None:
        encoder.load_weights(encoder_checkpoint)
        decoder.load_weights(decoder_checkpoint)
    en_output, en_alignments = encoder(pes, tf.constant(sentence, dtype=tf.int32), training=False)

    for i in range(config.MAX_LENGTH):
        de_output, de_bot_alignments, de_mid_alignments = decoder(pes, de_input, en_output, training=False)

        new_word = tf.cast(tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(new_word, config.END_TOKEN[0]):
            break
        # concatenated the predicted_id to the output which is given to the decoder as its input.
        output = tf.concat([de_input, new_word], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    with open(config.PES, 'rb') as f:
        pes = pickle.load(f)
    prediction = evaluate(pes, sentence)
    with open(config.TOKENIZER, 'rb') as f:
        tokenizer = pickle.load(f)
    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])

    return predicted_sentence
