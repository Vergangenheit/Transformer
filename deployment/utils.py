import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
import config.config_chatbot as config
import models.chatbot_model as ch_model
from data_processing.dm_chatbot import preprocess_sentence


def load_tf_model():
    """Load in the pre-trained model"""
    global model
    model = ch_model.transformer(
        vocab_size=config.VOCAB_SIZE,
        num_layers=config.NUM_LAYERS,
        units=config.UNITS,
        d_model=config.MODEL_SIZE,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT)
    # load weights
    model.load_weights(config.model_weights)
    # global graph
    # graph = tf.get_default_graph()


def prepare_datapoint(sentence):
    sentence = preprocess_sentence(sentence)
    with open(config.TOKENIZER, 'rb') as f:
        tokenizer = pickle.load(f)
    sentence = tf.expand_dims(
        config.START_TOKEN + tokenizer.encode(sentence) + config.END_TOKEN, axis=0)
    output = tf.expand_dims(config.START_TOKEN, 0)

    return sentence, output


def generate_from_seed(model, seed):
    """Generate output from a sequence"""
    # load tokenizer
    with open(config.TOKENIZER, 'rb') as f:
        tokenizer = pickle.load(f)
    sentence = tf.expand_dims(
        config.START_TOKEN + tokenizer.encode(seed) + config.END_TOKEN, axis=0)
    output = tf.expand_dims(config.START_TOKEN, 0)
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

    prediction = tf.squeeze(output, axis=0)

    generated_answer = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    # Formatting in html
    html = ''
    html = addContent(html, header(
        'Input Seed ', color='black', gen_text='Network Output'))
    html = addContent(html, generated_answer)

    return f'<div>{html}</div>'


def header(text, color='black', gen_text=None):
    """Create an HTML header"""

    if gen_text:
        raw_html = f'<h1 style="margin-top:16px;color: {color};font-size:54px"><center>' + str(
            text) + '<span style="color: red">' + str(gen_text) + '</center></h1>'
    else:
        raw_html = f'<h1 style="margin-top:12px;color: {color};font-size:54px"><center>' + str(
            text) + '</center></h1>'
    return raw_html


def addContent(old_html, raw_html):
    """Add html content together"""

    old_html += raw_html
    return old_html


if __name__ == "__main__":
    sentence = input()
    load_tf_model()
    sentence, output = prepare_datapoint(sentence)
    print("sentence {}".format(sentence))
    print("output {}".format(output))
