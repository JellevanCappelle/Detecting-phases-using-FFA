import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import metrics

from FFA import FFModel

class FFClassifier(Model):
    """simple supervised classifier model with an `FFModel` under the hood"""

    def __init__(self, ffmodel: FFModel, classifier: Model, n_classes: int, big_mult: float = 1000, classifier_weight_cost: float = 0.03, metrics = [metrics.CategoricalAccuracy(name = "label_accuracy")]):
        """`classifier` will be trained to predict logit probabilities.
        `big_mult` is used as a penalty multiplier for the true label when choosing a false label."""
        super().__init__()
        self.ffmodel = ffmodel
        self.classifier = classifier
        self.n_classes = n_classes
        self.big_mult = big_mult
        self.classifier_weight_cost = classifier_weight_cost
        self.user_metrics = metrics
    
    @tf.function
    def call(self, x):
        """return classifier output"""
        neutral_labels = tf.cast(tf.fill((tf.shape(x)[0], self.n_classes), 1 / self.n_classes), tf.float32)
        x = tf.concat([neutral_labels, x], 1)
        norm_state = self.ffmodel(x)
        return self.classifier(norm_state)

    @tf.function
    def train_step(self, data):
        """add true/false labels to the input data to generate positive/negative training data"""
        examples, class_labels = data
        examples = tf.cast(examples, tf.float32)
        class_labels = tf.cast(class_labels, tf.float32)
        
        batch_size = tf.shape(examples)[0]
        
        # forward pass with neutral examples
        neutral_labels = tf.cast(tf.fill((batch_size, self.n_classes), 1 / self.n_classes), tf.float32)
        neutral_examples = tf.concat([neutral_labels, examples], 1)
        norm_state = self.ffmodel(neutral_examples)
        predicted_logits = self.classifier(norm_state)

        # generate positive and negative examples
        pos_examples = tf.concat([class_labels, examples], 1)
        false_labels = predicted_logits - self.big_mult * class_labels
        false_labels = tf.reshape(tf.random.categorical(false_labels, 1), (batch_size,))
        false_labels = tf.one_hot(false_labels, self.n_classes)
        neg_examples = tf.concat([false_labels, examples], 1)

        # train models
        ff_metrics = self.ffmodel.train_step((pos_examples, neg_examples))
        classifier_metrics = self.classifier.train_step((norm_state, class_labels))

        # hack: apply weight cost to classifier
        w = self.classifier.trainable_variables[0]
        w.assign(w - tf.cast(self.classifier.optimizer.learning_rate * self.classifier_weight_cost, tf.float32) * w)

        # calculate debug metrics
        state_metrics = {
            "norm_state_stddev": tf.math.reduce_std(norm_state),
            "norm": tf.reduce_mean(tf.pow(norm_state, 2)),
            }
        
        # return combined metrics
        for m in self.user_metrics:
            m.update_state(class_labels, predicted_logits)
        user_metrics = {metric.name: metric.result() for metric in self.user_metrics}
        ff_metrics = {f"ffmodel_{key}": value for key, value in ff_metrics.items()}
        classifier_metrics = {f"classifier_{key}": value for key, value in classifier_metrics.items()}
        return {**user_metrics, **state_metrics, **ff_metrics, **classifier_metrics}

    def test_step(self, data):
        examples, class_labels = data
        examples = tf.cast(examples, tf.float32)
        class_labels = tf.cast(class_labels, tf.float32)

        batch_size = tf.shape(examples)[0]

        # generate positive and negative examples
        predicted_logits = self(examples)
        pos_examples = tf.concat([class_labels, examples], 1)
        false_labels = predicted_logits - self.big_mult * class_labels
        false_labels = tf.reshape(tf.random.categorical(false_labels, 1), (batch_size,))
        false_labels = tf.one_hot(false_labels, self.n_classes)
        neg_examples = tf.concat([false_labels, examples], 1)

        # evaluate FF-specific metrics
        self.ffmodel.update_metrics(pos_examples, neg_examples)
        custom_metrics = {metric.name: metric.result() for metric in self.ffmodel.custom_metrics}
        
        # evaluate other metrics
        for m in self.user_metrics:
            m.update_state(class_labels, predicted_logits)
        user_metrics = {metric.name: metric.result() for metric in self.user_metrics}
        return  {**user_metrics, **custom_metrics}
