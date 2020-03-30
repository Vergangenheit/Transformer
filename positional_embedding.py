import numpy as np
import tensorflow as tf
import config.config_chatbot as config
import pickle


def positional_encodings(pos, model_size):
    """ Compute positional encoding for a particular position
      Args:
          pos: position of a token in the sequence
          model_size: depth size of the model

      Returns:
          The positional encoding for the given token"""

    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE


def build_pes(data_en, data_fr_in):
    max_length = max(len(data_en[0]), len(data_fr_in[0]))

    pes = []
    for i in range(max_length):
        pes.append(positional_encodings(i, config.MODEL_SIZE))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)
    # save pes
    with open(config.PES, 'wb') as f:
        pickle.dump(pes, f)
    return pes

# if __name__ == '__main__':
#     pes = build_pes(data_en, data_fr_in)
