import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import metrics


class AvgMetric(metrics.Metric):
    def __init__(self, name):
        super().__init__(name)
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

class FFLayer(Layer):
    def __init__(self, size: int, mean_delay: float, mean_weight: float, weight_cost: float, **kwargs):
        """`mean_delay` must be in the (0, 1] range"""

        super().__init__(**kwargs)
        self.size = size
        self.means = tf.Variable(0.5 * tf.ones((size,)), trainable = False)
        self.mean_delay = mean_delay
        self.mean_weight = mean_weight
        self.weight_cost = weight_cost

    def build(self, input_shape):
        # NOTE: sligtly different from the initialization used by Hinton
        self.w = self.add_weight(name = "weights", shape = (input_shape[-1], self.size), initializer = "he_normal", trainable = True)
        self.b = self.add_weight(name = "biases", shape = (self.size,), initializer = "zeros", trainable = True)

    def call(self, inputs):
        return tf.maximum(0., tf.matmul(inputs, self.w) + self.b, name = "fflayer")

    def goodness(self, state):
        """calculates goodness for a column of states"""

        return tf.reduce_sum(tf.pow(state, 2), axis = 1, keepdims = True)

    def probability(self, state):
        """calculate probability
        expects a column of state row-vectors, returns a column of scalar probabilities"""

        threshold = tf.cast(tf.shape(state)[1], tf.float32)
        x = self.goodness(state) - threshold
        return 1 / (1 + tf.exp(-x))

    @staticmethod
    def normalize(x):
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
        threshold = 2 * tf.sqrt(1 / self.size)
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
        return pos_state, neg_state, -gradient_by_weight, -gradient_by_bias, errors
    
    def calc_pairwise_err(self, pos_inputs, neg_inputs):
        """calculates pairwise errors for use in validation"""

        # calculate states and goodness
        pos_state = self.call(pos_inputs)
        pos_goods = self.goodness(pos_state)
        neg_state = self.call(neg_inputs)
        neg_goods = self.goodness(neg_state)

        errors = tf.reduce_sum(tf.cast(neg_goods >= pos_goods, tf.float32)) / tf.cast(tf.shape(pos_inputs)[0], tf.float32)
        return pos_state, neg_state, errors

class FFModel(Model):
    """base model for FFA, outputs the states of layers starting from `min_output_layer`"""#

    def __init__(self, layer_sizes: list, min_output_layer: int = 1,
                 mean_delay: float = 0.9,
                 mean_weight: float = 0.03,
                 weight_cost: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)
        self.min_output_layer = min_output_layer
        self.fflayers = [FFLayer(size, mean_delay, mean_weight, weight_cost)
                         for size in layer_sizes]
        self.weight_cost = weight_cost
        self.pairwise_metrics = [AvgMetric(f"pairwise_{i + 1}") for i in range(len(self.fflayers))]
        self.pos_value_metrics = [AvgMetric(f"pos_value_{i + 1}") for i in range(len(self.fflayers))]
        self.neg_value_metrics = [AvgMetric(f"neg_value_{i + 1}") for i in range(len(self.fflayers))]
        self.pos_good_metrics = [AvgMetric(f"pos_good_{i + 1}") for i in range(len(self.fflayers))]
        self.neg_good_metrics = [AvgMetric(f"neg_good_{i + 1}") for i in range(len(self.fflayers))]
        self.pos_prob_metrics = [AvgMetric(f"pos_prob_{i + 1}") for i in range(len(self.fflayers))]
        self.neg_prob_metrics = [AvgMetric(f"neg_prob_{i + 1}") for i in range(len(self.fflayers))]
        self.custom_metrics = self.pairwise_metrics \
            + self.pos_value_metrics + self.neg_value_metrics \
            + self.pos_good_metrics + self.neg_good_metrics \
            + self.pos_prob_metrics + self.neg_prob_metrics

    @tf.function
    def call(self, x):
        # feed x through the network
        input = FFLayer.normalize(x)
        norm_states = [input := FFLayer.normalize(layer(input)) for layer in self.fflayers]
        return tf.concat(norm_states[self.min_output_layer:], 1)
    
    @tf.function
    def train_step(self, data):
        pos_inputs, neg_inputs = data

        # collect gradients (for both weigths and biases)
        gradients = []
        pos_state, neg_state = pos_inputs, neg_inputs
        for i, layer in enumerate(self.fflayers):
            pos_norm, neg_norm = FFLayer.normalize(pos_state), FFLayer.normalize(neg_state)
            pos_state, neg_state, grad_w, grad_b, err = layer.calc_state_and_gradients(pos_norm, neg_norm)
            self.pos_value_metrics[i].update_state(tf.reduce_mean(pos_state))
            self.neg_value_metrics[i].update_state(tf.reduce_mean(neg_state))
            self.pos_good_metrics[i].update_state(tf.reduce_mean(layer.goodness(pos_state)))
            self.neg_good_metrics[i].update_state(tf.reduce_mean(layer.goodness(neg_state)))
            self.pos_prob_metrics[i].update_state(tf.reduce_mean(layer.probability(pos_state)))
            self.neg_prob_metrics[i].update_state(tf.reduce_mean(layer.probability(neg_state)))
            self.pairwise_metrics[i].update_state(err)
            gradients.append(grad_w)
            gradients.append(grad_b)
        
        # apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # apply weight cost
        for layer in self.fflayers:
            layer.w.assign(layer.w - tf.cast(self.optimizer.learning_rate * layer.weight_cost, tf.float32) * layer.w)

        # return metrics
        return {metric.name: metric.result() for metric in self.custom_metrics}
    
    def update_metrics(self, pos_inputs, neg_inputs):
        """computes metrics for the given inputs"""

        pos_state, neg_state = pos_inputs, neg_inputs
        for i, layer in enumerate(self.fflayers):
            pos_state, neg_state, err = layer.calc_pairwise_err(FFLayer.normalize(pos_state), FFLayer.normalize(neg_state))
            self.pos_value_metrics[i].update_state(tf.reduce_mean(pos_state))
            self.neg_value_metrics[i].update_state(tf.reduce_mean(neg_state))
            self.pos_good_metrics[i].update_state(tf.reduce_mean(layer.goodness(pos_state)))
            self.neg_good_metrics[i].update_state(tf.reduce_mean(layer.goodness(neg_state)))
            self.pos_prob_metrics[i].update_state(tf.reduce_mean(layer.probability(pos_state)))
            self.neg_prob_metrics[i].update_state(tf.reduce_mean(layer.probability(neg_state)))
            self.pairwise_metrics[i].update_state(err)
