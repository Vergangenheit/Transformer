import tensorflow as tf
from model import Encoder, Decoder
import config.config_chatbot as config
import os
import time
import data_processing.chatbot.pipeline as pp

def loss_function(targets, logits):
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss


class WarmupThenDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Learning schedule for training the Transformer
    Attributes:
        model_size: d_model in the paper (depth size of the model)
        warmup_steps: number of warmup steps at the beginning
    """

    def __init__(self, model_size, warmup_steps=4000):
        super(WarmupThenDecaySchedule, self).__init__()

        self.model_size = model_size
        self.model_size = tf.cast(self.model_size, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step_term = tf.math.rsqrt(step)
        warmup_term = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.model_size) * tf.math.minimum(step_term, warmup_term)


@tf.function
def train_step(encoder, decoder, source_seq, target_seq_in, target_seq_out):
    # encoder = Encoder(config.VOCAB_SIZE, config.MODEL_SIZE, config.NUM_LAYERS, config.H)
    # decoder = Decoder(config.VOCAB_SIZE, config.MODEL_SIZE, config.NUM_LAYERS, config.H)
    lr = WarmupThenDecaySchedule(config.MODEL_SIZE)
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    with tf.GradientTape() as tape:
        encoder_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
        # encoder_mask has shape (batch_size, source_len)
        # we need to add two more dimensions in between
        # to make it broadcastable when computing attention heads
        encoder_mask = tf.expand_dims(encoder_mask, axis=1)
        encoder_mask = tf.expand_dims(encoder_mask, axis=1)
        encoder_output, _ = encoder(source_seq, encoder_mask=encoder_mask)

        decoder_output, _, _ = decoder(
            target_seq_in, encoder_output, encoder_mask=encoder_mask)

        loss = loss_function(target_seq_out, decoder_output)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


def train():
    if not os.path.exists(config.checkpoints_en):
        os.makedirs(config.checkpoints_en)
    if not os.path.exists(config.checkpoints_de):
        os.makedirs(config.checkpoints_de)

    # Uncomment these lines for inference mode
    # encoder_checkpoint = tf.train.latest_checkpoint(config.checkpoints_en)
    # decoder_checkpoint = tf.train.latest_checkpoint(config.checkpoints_de)

    encoder = Encoder(config.VOCAB_SIZE, config.MODEL_SIZE, config.NUM_LAYERS, config.H)
    decoder = Decoder(config.VOCAB_SIZE, config.MODEL_SIZE, config.NUM_LAYERS, config.H)

    # if encoder_checkpoint is not None and decoder_checkpoint is not None:
    #     encoder.load_weights(encoder_checkpoint)
    #     decoder.load_weights(decoder_checkpoint)

    starttime = time.time()
    dataset = pp.dataset_pipeline()
    for e in range(config.EPOCHS):
        encoder.save_weights('checkpoints/encoder/encoder_{}.h5'.format(e + 1))
        decoder.save_weights('checkpoints/decoder/decoder_{}.h5'.format(e + 1))
        for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
            loss = train_step(encoder, decoder, source_seq, target_seq_in, target_seq_out)
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                    e + 1, batch, loss.numpy(), time.time() - starttime))
                starttime = time.time()



