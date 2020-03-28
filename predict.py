import numpy as np
import config.config_chatbot as config
import pickle
from data_processing.chatbot.dm_chatbot import preprocess_sentence
from model import Decoder, Encoder
import tensorflow as tf


def predict(questions: list, sentence=None):
    sentence = preprocess_sentence(sentence)
    with open(config.TOKENIZER, 'rb') as f:
        tokenizer = pickle.load(f)
    sentence = tf.expand_dims(
        config.START_TOKEN + tokenizer.encode(sentence) + config.END_TOKEN, axis=0)

    #load encoder and decoder
    encoder_checkpoint = tf.train.latest_checkpoint(config.checkpoints_en)
    decoder_checkpoint = tf.train.latest_checkpoint(config.checkpoints_de)
    encoder = Encoder(config.VOCAB_SIZE, config.MODEL_SIZE, config.NUM_LAYERS, config.H)
    decoder = Decoder(config.VOCAB_SIZE, config.MODEL_SIZE, config.NUM_LAYERS, config.H)
    if encoder_checkpoint is not None and decoder_checkpoint is not None:
        encoder.load_weights(encoder_checkpoint)
        decoder.load_weights(decoder_checkpoint)
    en_output, en_alignments = encoder(tf.constant(sentence), training=False)
