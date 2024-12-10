import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Flatten
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import metrics


class AvgMetric(metrics.Metric):
    def __init__(self, name):
        super().__init__(name = name, dtype = tf.float32)
        self.n = tf.Variable(0, dtype = tf.int32)
        self.sum = tf.Variable(0, dtype = tf.float32)
    
    def reset_state(self):
        self.n.assign(0)
        self.sum.assign(0)

    def update_state(self, x):
        self.n.assign(self.n + 1)
        self.sum.assign(self.sum + x)

    def result(self):
        return self.sum / tf.cast(self.n, tf.float32)

class FFDense(Layer):
    def __init__(self, size: int, mean_delay: float, mean_weight: float, weight_cost: float, **kwargs):
        """`mean_delay` must be in the (0, 1] range"""

        super().__init__(**kwargs)
        self.size = size
        self.means = tf.Variable(0.5 * tf.ones((size,)), trainable = False)
        self.mean_delay = mean_delay
        self.mean_weight = mean_weight
        self.weight_cost = weight_cost

    def build(self, input_shape):
        # same weight initialization as used in original code by Hinton, i.e. random normal scaled by 1/sqrt(fan_in)
        initializer = RandomNormal(stddev = 1 / np.sqrt(input_shape[-1]))
        self.w = self.add_weight(name = "weights", shape = (input_shape[-1], self.size), initializer = initializer, trainable = True)
        self.b = self.add_weight(name = "biases", shape = (self.size,), initializer = "zeros", trainable = True)

    def call(self, inputs):
        return tf.maximum(0., tf.matmul(inputs, self.w) + self.b, name = "fflayer")

    def goodness(self, state):
        """calculates goodness for a column of states"""
        return tf.reduce_sum(tf.pow(state, 2), axis = 1, keepdims = True)

    def probability(self, state):
        """calculate probability
        expects a column of state row-vectors, returns a column of scalar probabilities"""

        threshold = tf.cast(tf.shape(state)[1], tf.float32) # TODO: just use self.size?
        x = self.goodness(state) - threshold
        return 1 / (1 + tf.exp(-x))

    def normalize(self, x):
        """return `x` normalized by the mean of squared elements.
        `x` must have a float datatype"""
        return tf.math.divide_no_nan(x, tf.sqrt(tf.reduce_mean(tf.pow(x, 2), axis = 1, keepdims = True)))

    def calc_state_and_gradients(self, pos_inputs, neg_inputs):
        """calculates the forward-forward gradients.
        `labels` should be a column vector with an entry for every input instance,
        being 1 for positive and 0 for negative examples.
        """

        # calculate states and probabilities
        pos_state = self.call(pos_inputs)
        pos_goods = self.goodness(pos_state)
        pos_probs = self.probability(pos_state)
        neg_state = self.call(neg_inputs)
        neg_goods = self.goodness(neg_state)
        neg_probs = self.probability(neg_state)

        # update means
        target_means = tf.reduce_mean(pos_state, 0)
        self.means.assign(self.mean_delay * self.means + (1 - self.mean_delay) * target_means)

        # calculate gradient w.r.t. state per example
        threshold = 2 * tf.sqrt(1 / self.size) # TODO: get rid of this???
        mean_of_means = tf.reduce_mean(self.means)
        mean_of_means = tf.maximum(mean_of_means, threshold)
        pos_gradient = (1 - pos_probs) * pos_state + self.mean_weight * (mean_of_means - self.means)
        neg_gradient = - neg_probs * neg_state
        
        # calculate gradient w.r.t. weights and biases per example
        gradient_by_weight = \
            tf.tensordot(pos_inputs, pos_gradient, [[0], [0]]) / tf.cast(tf.shape(pos_inputs)[0], tf.float32) + \
            tf.tensordot(neg_inputs, neg_gradient, [[0], [0]]) / tf.cast(tf.shape(neg_inputs)[0], tf.float32)
        gradient_by_bias = tf.reduce_mean(pos_gradient, 0) + tf.reduce_mean(neg_gradient, 0)
        
        # return total gradients for this batch
        # gradients are negated for use in a minimizing optimizer
        errors = tf.reduce_sum(tf.cast(neg_goods >= pos_goods, tf.float32)) / tf.cast(tf.shape(pos_inputs)[0], tf.float32)
        return pos_state, neg_state, [-gradient_by_weight, -gradient_by_bias], errors
    
    def calc_pairwise_err(self, pos_inputs, neg_inputs):
        """calculates pairwise errors for use in validation"""

        # calculate states and goodness
        pos_state = self.call(pos_inputs)
        pos_goods = self.goodness(pos_state)
        neg_state = self.call(neg_inputs)
        neg_goods = self.goodness(neg_state)

        errors = tf.reduce_sum(tf.cast(neg_goods >= pos_goods, tf.float32)) / tf.cast(tf.shape(pos_inputs)[0], tf.float32) # TODO: just use reduce_mean()?
        return pos_state, neg_state, errors

class FFConv(Layer):
    """Performs the same operation as `FFLayer`, but as a convolution.
    Uses the shape `[batch_size, height, width, channels]` for all input and output."""

    def __init__(self, width: int, stride: int, filters: int, mean_delay: float, mean_weight: float, weight_cost: float, **kwargs):
        """`mean_delay` must be in the (0, 1] range"""

        super().__init__(**kwargs)
        self.width = width
        self.stride = stride
        self.filters = filters
        self.separate_labels = False # should be 0 if no labels are used
        self.means = tf.Variable(0.5 * tf.ones((filters,)), trainable = False)
        self.mean_delay = mean_delay
        self.mean_weight = mean_weight
        self.weight_cost = weight_cost
    
    def build(self, input_shape):
        # same weight initialization as used in original code by Hinton, i.e. random normal scaled by 1/sqrt(fan_in)
        if isinstance(input_shape[0], tuple): # infer the usage of separate labels from input_shape
            self.separate_labels = True
            input_shape, label_shape = input_shape
            stddev = 1 / np.sqrt(self.width * self.width * input_shape[-1] + label_shape[-1])
        else:
            stddev = 1 / np.sqrt(np.prod(input_shape[1:]))
        initializer = RandomNormal(stddev = stddev)
        self.w = self.add_weight(name = "weights", shape = (self.width, self.width, input_shape[-1], self.filters), initializer = initializer, trainable = True)
        self.b = self.add_weight(name = "biases", shape = (self.filters,), initializer = "zeros", trainable = True)
        if self.separate_labels:
            self.lw = self.add_weight(name = "label_weights", shape = (label_shape[-1], self.filters), initializer = initializer, trainable = True)
    
    def call(self, inputs):
        if self.separate_labels:
            inputs, labels = inputs
            bias = tf.matmul(labels, self.lw) + self.b
            bias = tf.reshape(bias, (tf.shape(inputs)[0], 1, 1, self.filters)) # type: ignore
        else:
            bias = self.b
        return tf.maximum(0., tf.nn.conv2d(inputs, self.w, self.stride, "VALID") + bias, name = "ffconv")
    
    def goodness(self, state):
        """calculates goodness per row of channels, i.e. the output shape is `[batch_size, height, width, 1]`."""
        return tf.reduce_sum(tf.pow(state, 2), axis = -1, keepdims = True)
    
    def probability(self, state):
        """Calculate probability.
        Output shape is `[batch_size, height, width, 1]`."""

        threshold = tf.cast(self.filters, tf.float32)
        x = self.goodness(state) - threshold
        return 1 / (1 + tf.exp(-x))

    def normalize(self, x):
        """Return `x` normalized by the mean of squared channels.
        `x` must have a float datatype.
        Output shape is same as input shape."""
        if isinstance(x, tuple):
            return (self.normalize(x[0]), x[1]) # don't normalize labels!
        else:
            norm = tf.sqrt(tf.reduce_mean(tf.pow(x, 2), axis = -1, keepdims = True))
            return tf.math.divide_no_nan(x, norm)
    
    def calc_state_and_gradients(self, pos_inputs, neg_inputs):
        """calculates the forward-forward gradients.
        `labels` should be a column vector with an entry for every input instance,
        being 1 for positive and 0 for negative examples.
        """

        # calculate states and probabilities
        pos_state = self.call(pos_inputs)
        pos_goods = self.goodness(pos_state)
        pos_probs = self.probability(pos_state)
        neg_state = self.call(neg_inputs)
        neg_goods = self.goodness(neg_state)
        neg_probs = self.probability(neg_state)

        # update means
        target_means = tf.reduce_mean(pos_state, (0, 1, 2))
        self.means.assign(self.mean_delay * self.means + (1 - self.mean_delay) * target_means) # type: ignore

        # calculate gradient w.r.t. state per example
        threshold = 2 * tf.sqrt(1 / self.filters) # TODO: get rid of this???
        mean_of_means = tf.reduce_mean(self.means)
        mean_of_means = tf.maximum(mean_of_means, threshold)
        pos_gradient = (1 - pos_probs) * pos_state + self.mean_weight * (mean_of_means - self.means)
        neg_gradient = -neg_probs * neg_state
        
        # calculate gradient w.r.t. weights and biases per example
        if self.separate_labels:
            pos_inputs, pos_labels = pos_inputs
            neg_inputs, neg_labels = neg_inputs
        gradient_by_weight = \
            tf.compat.v1.nn.conv2d_backprop_filter(pos_inputs, tf.shape(self.w), pos_gradient, (1, self.stride, self.stride, 1), "VALID") + \
            tf.compat.v1.nn.conv2d_backprop_filter(neg_inputs, tf.shape(self.w), neg_gradient, (1, self.stride, self.stride, 1), "VALID")
        # TODO: is averaging already done in conv2d_backprop_filter_v2??? -> no.
        gradient_by_weight /= tf.cast(tf.reduce_prod(tf.shape(pos_gradient)[:3]), tf.float32) # take average gradient from batch and convolutions # type: ignore
        gradient_by_bias = tf.reduce_mean(pos_gradient + neg_gradient, (0, 1, 2))
        if self.separate_labels:
            pos_gradient_by_label = tf.reduce_mean(pos_gradient, (1, 2))
            neg_gradient_by_label = tf.reduce_mean(neg_gradient, (1, 2))
            gradient_by_label_weights = \
                tf.tensordot(pos_labels, pos_gradient_by_label, [[0], [0]]) / tf.cast(tf.shape(pos_labels)[0], tf.float32) + \
                tf.tensordot(neg_labels, neg_gradient_by_label, [[0], [0]]) / tf.cast(tf.shape(neg_labels)[0], tf.float32)
            gradients = [-gradient_by_weight, -gradient_by_bias, -gradient_by_label_weights] # gradients are negated for use in a minimizing optimizer
        else:
            gradients = [-gradient_by_weight, -gradient_by_bias] # gradients are negated for use in a minimizing optimizer
        
        # return total gradients for this batch
        errors = tf.reduce_mean(tf.cast(neg_goods >= pos_goods, tf.float32))
        return pos_state, neg_state, gradients, errors

    def calc_pairwise_err(self, pos_inputs, neg_inputs):
        """Calculates pairwise errors for use in validation"""

        # calculate states and goodness
        pos_state = self.call(pos_inputs)
        pos_goods = self.goodness(pos_state)
        neg_state = self.call(neg_inputs)
        neg_goods = self.goodness(neg_state)

        errors = tf.reduce_mean(tf.cast(neg_goods >= pos_goods, tf.float32))
        return pos_state, neg_state, errors


class FFModel(Model):
    """base model for FFA, outputs the states of layers starting from `min_output_layer`"""#

    def __init__(self, layer_sizes: list, min_output_layer: int = 1,
                 mean_delay: float = 0.9,
                 mean_weight: float = 0.03,
                 weight_cost: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)
        
        # build model to specification
        self.fflayers = []
        for size in layer_sizes:
            if isinstance(size, tuple):
                self.fflayers.append(FFConv(size[0], size[1], size[2], mean_delay, mean_weight, weight_cost))
            else:
                self.fflayers.append(FFDense(size, mean_delay, mean_weight, weight_cost))
        
        self.min_output_layer = min_output_layer
        self.weight_cost = weight_cost
        self.pairwise_metrics = [AvgMetric(f"pairwise_{i + 1}") for i in range(len(self.fflayers))]
        self.pos_value_metrics = [AvgMetric(f"pos_value_{i + 1}") for i in range(len(self.fflayers))]
        self.neg_value_metrics = [AvgMetric(f"neg_value_{i + 1}") for i in range(len(self.fflayers))]
        self.pos_good_metrics = [AvgMetric(f"pos_good_{i + 1}") for i in range(len(self.fflayers))]
        self.neg_good_metrics = [AvgMetric(f"neg_good_{i + 1}") for i in range(len(self.fflayers))]
        self.pos_prob_metrics = [AvgMetric(f"pos_prob_{i + 1}") for i in range(len(self.fflayers))]
        self.neg_prob_metrics = [AvgMetric(f"neg_prob_{i + 1}") for i in range(len(self.fflayers))]
        
        # filter out actual FF layers for list of metrics to report
        custom_metrics = [self.pairwise_metrics, self.pos_value_metrics, self.neg_value_metrics,
                          self.pos_good_metrics, self.neg_good_metrics, self.pos_prob_metrics, self.neg_prob_metrics]
        self.custom_metrics = sum(custom_metrics, [])

    def flatten(self, x):
        return tf.reshape(x, (tf.shape(x)[0], -1)) # type: ignore

    @tf.function
    def call(self, x):
        # feed x through the network
        input = self.fflayers[0].normalize(x)
        norm_states = [input := layer.normalize(layer(self.flatten(input))) if isinstance(layer, FFDense)
                       else layer.normalize(layer(input))
                       for layer in self.fflayers]
        norm_states = [self.flatten(x) for x in norm_states]# TODO: remove?
        return tf.concat(norm_states[self.min_output_layer:], 1)
    
    @tf.function
    def train_step(self, data):
        pos_inputs, neg_inputs = data

        # collect gradients (for both weigths and biases)
        gradients = []
        pos_state, neg_state = pos_inputs, neg_inputs
        for i, layer in enumerate(self.fflayers):
            if isinstance(layer, FFDense):
                pos_state, neg_state = self.flatten(pos_state), self.flatten(neg_state)
            pos_norm, neg_norm = layer.normalize(pos_state), layer.normalize(neg_state)
            pos_state, neg_state, grad, err = layer.calc_state_and_gradients(pos_norm, neg_norm)
            self.pos_value_metrics[i].update_state(tf.reduce_mean(pos_state))
            self.neg_value_metrics[i].update_state(tf.reduce_mean(neg_state))
            self.pos_good_metrics[i].update_state(tf.reduce_mean(layer.goodness(pos_state)))
            self.neg_good_metrics[i].update_state(tf.reduce_mean(layer.goodness(neg_state)))
            self.pos_prob_metrics[i].update_state(tf.reduce_mean(layer.probability(pos_state)))
            self.neg_prob_metrics[i].update_state(tf.reduce_mean(layer.probability(neg_state)))
            self.pairwise_metrics[i].update_state(err)
            gradients += grad
        
        # apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # apply weight cost
        for layer in self.fflayers:
            if isinstance(layer, FFDense):
                layer.w.assign(layer.w - tf.cast(self.optimizer.learning_rate * layer.weight_cost, tf.float32) * layer.w)
            if isinstance(layer, FFConv):
                layer.w.assign(layer.w - tf.cast(self.optimizer.learning_rate * layer.weight_cost, tf.float32) * layer.w)
                layer.lw.assign(layer.lw - tf.cast(self.optimizer.learning_rate * layer.weight_cost, tf.float32) * layer.lw)

        # return metrics
        return {metric.name: metric.result() for metric in self.custom_metrics if metric is not None}
    
    def update_metrics(self, pos_inputs, neg_inputs):
        """computes metrics for the given inputs"""

        pos_state, neg_state = pos_inputs, neg_inputs
        for i, layer in enumerate(self.fflayers):
            if isinstance(layer, FFDense):
                pos_state, neg_state = self.flatten(pos_state), self.flatten(neg_state)
            pos_state, neg_state, err = layer.calc_pairwise_err(layer.normalize(pos_state), layer.normalize(neg_state))
            self.pos_value_metrics[i].update_state(tf.reduce_mean(pos_state))
            self.neg_value_metrics[i].update_state(tf.reduce_mean(neg_state))
            self.pos_good_metrics[i].update_state(tf.reduce_mean(layer.goodness(pos_state)))
            self.neg_good_metrics[i].update_state(tf.reduce_mean(layer.goodness(neg_state)))
            self.pos_prob_metrics[i].update_state(tf.reduce_mean(layer.probability(pos_state)))
            self.neg_prob_metrics[i].update_state(tf.reduce_mean(layer.probability(neg_state)))
            self.pairwise_metrics[i].update_state(err)
