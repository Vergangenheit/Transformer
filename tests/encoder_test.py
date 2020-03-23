import config
import model
import pickle
import pipeline as pp
import positional_embedding as pe
import tensorflow as tf


def run_test():
    data_en, data_fr_in, dataset = pp.pipeline()
    pes = pe.build_pes(data_en, data_fr_in)
    # load en_tokenizer
    with open(config.EN_TOKENIZER, 'rb') as f:
        en_tokenizer = pickle.load(f)

    vocab_size = len(en_tokenizer.word_index) + 1
    encoder = model.Encoder(vocab_size, config.MODEL_SIZE, config.NUM_LAYERS, config.H)
    print(vocab_size)
    sequence_in = tf.constant([[1, 2, 3, 0, 0]])
    encoder_output, _ = encoder(pes, sequence_in)
    print(encoder_output.shape)


if __name__ == "__main__":
    run_test()
