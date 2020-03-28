from config import config_trans
import model
import pickle
from data_processing.translation import pipeline as pp
import positional_embedding as pe
import tensorflow as tf


def run_test():
    data_en, data_fr_in, dataset = pp.pipeline()
    pes = pe.build_pes(data_en, data_fr_in)

    # load en_tokenizer
    with open(config_trans.EN_TOKENIZER, 'rb') as f:
        en_tokenizer = pickle.load(f)

    en_vocab_size = len(en_tokenizer.word_index) + 1
    encoder = model.Encoder(en_vocab_size, config_trans.MODEL_SIZE, config_trans.NUM_LAYERS, config_trans.H)
    sequence_in = tf.constant([[1, 2, 3, 0, 0]])
    encoder_output, _ = encoder(pes, sequence_in)

    # load fr_tokenizer
    with open(config_trans.FR_TOKENIZER, 'rb') as f:
        fr_tokenizer = pickle.load(f)
    fr_vocab_size = len(fr_tokenizer.word_index) + 1
    print(fr_vocab_size)
    decoder = model.Decoder(fr_vocab_size, config_trans.MODEL_SIZE, config_trans.NUM_LAYERS, config_trans.H)

    sequence_in = tf.constant([[14, 24, 36, 0, 0]])
    decoder_output, _, _ = decoder(pes, sequence_in, encoder_output)
    print(decoder_output.shape)


if __name__ == "__main__":
    run_test()
