import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(1)

    def call(self, inputs):
        # inputs: (batch, H, W, C)
        x = tf.reshape(inputs, (tf.shape(inputs)[0], -1, inputs.shape[-1]))

        score = self.dense1(x)
        score = self.dense2(score)

        weights = tf.nn.softmax(score, axis=1)

        context = x * weights
        context = tf.reduce_sum(context, axis=1)

        return context

    def get_config(self):
        config = super().get_config()
        return config