import tensorflow as tf
from tensorflow.keras import Model # type: ignore

from FFA_new import FFModel, FFConv

class FFScalarDiscriminator(Model):
    def __init__(self, ffmodel: FFModel, bins_start: float, bins_end: float, n_bins: int, min_offset: float):
        if n_bins < 2:
            raise Exception(f"at least two bins required, got {n_bins}")

        super().__init__()
        self.ffmodel = ffmodel
        self.convolutional = isinstance(ffmodel.fflayers[0], FFConv)
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
    
    def concat_label(self, label, data):
        """concatenates label to the data"""

        if self.convolutional:
            # shape = tf.shape(data)
            # if len(shape) != 4: # type: ignore
            #     raise Exception("expected 4D input tensor for convolutional network")
            # label = tf.tile(tf.reshape(label, (-1, 1, 1, self.n_bins)), (1, shape[1], shape[2], 1)) # type: ignore
            return (data, label)
        elif len(tf.shape(data)) != 2: # type: ignore
            raise Exception("Expected 2D input tensor for non-concolutional network")
        return tf.concat([label, data], -1)

    
    def offset(self, scalar):
        """returns random values at least a certain offset away from the given scalars"""

        exclusion_start = tf.maximum(self.bins_start, scalar - self.offset_scale)
        exclusion_end = tf.minimum(self.bins_end, scalar + self.offset_scale)
        exclusion_width = exclusion_end - exclusion_start
        rand = tf.random.uniform(tf.shape(scalar), self.bins_start, self.bins_end - exclusion_width, dtype = tf.float32) # type: ignore
        return tf.where(rand < exclusion_start, rand, rand + exclusion_width)


    @tf.function
    def call(self, data):
        if isinstance(data, tuple):
            x, scalar = data
            if x.dtype != tf.float32:
                x = tf.cast(x, tf.float32)
            if scalar.dtype != tf.float64:
                scalar = tf.cast(scalar, tf.float32)
            input = self.concat_label(self.label(scalar), x)
            return self.ffmodel(input)
        else:
            if data.dtype != tf.float32:
                data = tf.cast(data, tf.float32)
            input = self.concat_label(self.neutral_label(tf.shape(data)[0]), data) # type: ignore
            return self.ffmodel(input)
    
    @tf.function
    def train_step(self, data):
        x, scalar = data
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)
        if scalar.dtype != tf.float64:
            scalar = tf.cast(scalar, tf.float32)

        pos_input = self.concat_label(self.label(scalar), x)
        offset_scalar = self.offset(scalar)
        neg_input = self.concat_label(self.label(offset_scalar), x)
        return self.ffmodel.train_step((pos_input, neg_input))
    
    @tf.function
    def test_step(self, data):
        x, scalar = data
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)
        if scalar.dtype != tf.float64:
            scalar = tf.cast(scalar, tf.float32)

        pos_input = self.concat_label(self.label(scalar), x)
        offset_scalar = self.offset(scalar)
        neg_input = self.concat_label(self.label(offset_scalar), x)

        # evaluate FF-specific metrics
        self.ffmodel.update_metrics(pos_input, neg_input)
        return {metric.name: metric.result() for metric in self.ffmodel.custom_metrics}
