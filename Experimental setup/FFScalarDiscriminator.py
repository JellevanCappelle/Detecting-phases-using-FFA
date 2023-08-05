import tensorflow as tf
from tensorflow.keras import Model

from FFA import FFModel

class FFScalarDiscriminator(Model):
    def __init__(self, ffmodel: FFModel, bins_start: float, bins_end: float, n_bins: int, min_offset: float):
        if n_bins < 2:
            raise Exception(f"at least two bins required, got {n_bins}")

        super().__init__()
        self.ffmodel = ffmodel
        self.bins_start = float(bins_start)
        self.bins_end = float(bins_end)
        self.n_bins = n_bins
        self.bin_width = (bins_end - bins_start) / (n_bins - 1)
        self.offset_scale = min_offset if min_offset is not None else self.bin_width
        self.bins = tf.cast(tf.linspace(bins_start, bins_end, n_bins), tf.float32)
    
    def label(self, scalar):
        """returns the twohot labels for the given scalars"""

        return tf.maximum(0., 1 - tf.abs(self.bins - scalar) / self.bin_width)
    
    def neutral_label(self, n):
        """returns a neutral twohot label"""

        return tf.ones((n, self.n_bins)) / self.n_bins
    
    def offset(self, scalar):
        """returns random values at least a certain offset away from the given scalars"""

        exclusion_start = tf.maximum(self.bins_start, scalar - self.offset_scale)
        exclusion_end = tf.minimum(self.bins_end, scalar + self.offset_scale)
        exclusion_width = exclusion_end - exclusion_start
        rand = tf.random.uniform(tf.shape(scalar), self.bins_start, self.bins_end - exclusion_width, dtype = tf.float32)
        return tf.where(rand < exclusion_start, rand, rand + exclusion_width)


    @tf.function
    def call(self, data):
        if isinstance(data, tuple):
            x, scalar = data
            if x.dtype != tf.float32:
                x = tf.cast(x, tf.float32)
            if scalar.dtype != tf.float64:
                scalar = tf.cast(scalar, tf.float32)
            input = tf.concat([self.label(scalar), x], 1)
            return self.ffmodel(input)
        else:
            if data.dtype != tf.float32:
                data = tf.cast(data, tf.float32)
            input = tf.concat([self.neutral_label(tf.shape(data)[0]), data], 1)
            return self.ffmodel(input)
    
    @tf.function
    def train_step(self, data):
        x, scalar = data
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)
        if scalar.dtype != tf.float64:
            scalar = tf.cast(scalar, tf.float32)

        pos_input = tf.concat([self.label(scalar), x], 1)
        offset_scalar = self.offset(scalar)
        neg_input = tf.concat([self.label(offset_scalar), x], 1)
        return self.ffmodel.train_step((pos_input, neg_input))
    
    @tf.function
    def test_step(self, data):
        x, scalar = data
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)
        if scalar.dtype != tf.float64:
            scalar = tf.cast(scalar, tf.float32)

        pos_input = tf.concat([self.label(scalar), x], 1)
        offset_scalar = self.offset(scalar)
        neg_input = tf.concat([self.label(offset_scalar), x], 1)

        # evaluate FF-specific metrics
        self.ffmodel.update_metrics(pos_input, neg_input)
        return {metric.name: metric.result() for metric in self.ffmodel.custom_metrics}
