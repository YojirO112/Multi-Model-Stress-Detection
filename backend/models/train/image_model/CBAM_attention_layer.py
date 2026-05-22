import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class CBAM(Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]

        # Channel Attention
        self.shared_dense_1 = Dense(channels // self.reduction_ratio,
                                   activation='relu',
                                   kernel_initializer='he_normal')
        self.shared_dense_2 = Dense(channels,
                                   kernel_initializer='he_normal')

        # Spatial Attention
        self.conv_spatial = Conv2D(1, kernel_size=7,
                                  padding='same',
                                  activation='sigmoid')

    def call(self, x):
        #  Channel Attention
        avg_pool = tf.reduce_mean(x, axis=[1,2])
        max_pool = tf.reduce_max(x, axis=[1,2])

        avg_out = self.shared_dense_2(self.shared_dense_1(avg_pool))
        max_out = self.shared_dense_2(self.shared_dense_1(max_pool))

        channel_attention = tf.nn.sigmoid(avg_out + max_out)
        channel_attention = tf.reshape(channel_attention, (-1,1,1,x.shape[-1]))

        x = x * channel_attention

        #  Spatial Attention
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)

        spatial = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_attention = self.conv_spatial(spatial)

        x = x * spatial_attention

        return x

    def get_config(self):
        config = super().get_config()

        config.update({
            "reduction_ratio": self.reduction_ratio
        })

        return config