from config import config_trans
from models import trans_model
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

    vocab_size = len(en_tokenizer.word_index) + 1
    encoder = trans_model.Encoder(vocab_size, config_trans.MODEL_SIZE, config_trans.NUM_LAYERS, config_trans.H)
    print(vocab_size)
    sequence_in = tf.constant([[1, 2, 3, 0, 0]])
    encoder_output, _ = encoder(pes, sequence_in)
    print(encoder_output.shape)


if __name__ == "__main__":
    run_test()
