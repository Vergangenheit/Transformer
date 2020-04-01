import config.config_chatbot as config
import models.chatbot_model as ch_model
import tensorflow as tf
import os


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, config.MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        })
        return config

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, config.MAX_LENGTH - 1))
    accuracy = tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)
    return accuracy


def train(dataset):
    tf.keras.backend.clear_session()
    learning_rate = CustomSchedule(config.MODEL_SIZE)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model = ch_model.transformer(
        vocab_size=config.VOCAB_SIZE,
        num_layers=config.NUM_LAYERS,
        units=config.UNITS,
        d_model=config.MODEL_SIZE,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[tf.metrics.SparseCategoricalAccuracy()])
    # instantiate checkpoint to save the models after every epoch
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    # callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(config.file_path, monitor='loss',
    #                                                          verbose=2, save_best_only=False, mode='min', period=1)
    model.fit(dataset, epochs=config.EPOCHS, verbose=2)
    model.save_weights(config.model_weights)
