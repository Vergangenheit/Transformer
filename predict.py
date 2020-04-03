import config_chatbot as config
import pickle
import models.chatbot_model as ch_model
from data_processing.dm_chatbot import preprocess_sentence
import tensorflow as tf
import train.train_chatbot as train


def load_model():
    tf.keras.backend.clear_session()
    learning_rate = train.CustomSchedule(config.MODEL_SIZE)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model = ch_model.transformer(
        vocab_size=config.VOCAB_SIZE,
        num_layers=config.NUM_LAYERS,
        units=config.UNITS,
        d_model=config.MODEL_SIZE,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT)

    model.compile(optimizer=optimizer, loss=train.loss_function, metrics=[tf.metrics.SparseCategoricalAccuracy()])

    # load weights
    model.load_weights(config.model_weights)

    return model


def evaluate(sentence=None):
    sentence = preprocess_sentence(sentence)
    with open(config.TOKENIZER, 'rb') as f:
        tokenizer = pickle.load(f)
    sentence = tf.expand_dims(
        config.START_TOKEN + tokenizer.encode(sentence) + config.END_TOKEN, axis=0)
    output = tf.expand_dims(config.START_TOKEN, 0)

    # load saved model
    tf.keras.backend.clear_session()
    learning_rate = train.CustomSchedule(config.MODEL_SIZE)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model = ch_model.transformer(
        vocab_size=config.VOCAB_SIZE,
        num_layers=config.NUM_LAYERS,
        units=config.UNITS,
        d_model=config.MODEL_SIZE,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT)

    model.compile(optimizer=optimizer, loss=train.loss_function, metrics=[tf.metrics.SparseCategoricalAccuracy()])

    # load weights
    model.load_weights(config.model_weights)

    for i in range(config.MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, config.END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)
    with open(config.TOKENIZER, 'rb') as f:
        tokenizer = pickle.load(f)
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence
