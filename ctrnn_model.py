import tensorflow as tf
import numpy as np
import os

class CTRNN(tf.keras.layers.Layer):
    def __init__(self, num_units, cell_clip=-1, global_feedback=False, fix_tau=True, **kwargs):
        super(CTRNN, self).__init__(**kwargs)
        self._num_units = num_units
        self._unfolds = 6
        self._delta_t = 0.1
        self.global_feedback = global_feedback
        self.fix_tau = fix_tau
        self.tau = 1
        self.cell_clip = cell_clip

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        self._input_size = int(input_shape[-1])
        self.W = self.add_weight(name='W', shape=[self._input_size, self._num_units],
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='b', shape=[self._num_units],
                                 initializer=tf.constant_initializer(0.0), trainable=True)
        if not self.fix_tau:
            self._tau_var = self.add_weight(name='tau', shape=[], initializer=tf.constant_initializer(self.tau),
                                            trainable=True)
        self.built = True

    def call(self, inputs, states):
        state = states[0]
        tau = tf.nn.softplus(self._tau_var) if not self.fix_tau else self.tau

        if not self.global_feedback:
            input_f_prime = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)

        for i in range(self._unfolds):
            if self.global_feedback:
                fused_input = tf.concat([inputs, state], axis=-1)
                input_f_prime = tf.nn.tanh(tf.matmul(fused_input, self.W) + self.b)

            f_prime = -state / tau + input_f_prime
            state = state + self._delta_t * f_prime

            if self.cell_clip > 0:
                state = tf.clip_by_value(state, -self.cell_clip, self.cell_clip)

        return state, [state]


class NODE(tf.keras.layers.Layer):
    def __init__(self, num_units, cell_clip=-1, **kwargs):
        super(NODE, self).__init__(**kwargs)
        self._num_units = num_units
        self._unfolds = 6
        self._delta_t = 0.1
        self.cell_clip = cell_clip

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        self._input_size = int(input_shape[-1])
        self.W = self.add_weight(name='W', shape=[self._input_size + self._num_units, self._num_units],
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='b', shape=[self._num_units],
                                 initializer=tf.constant_initializer(0.0), trainable=True)
        self.built = True

    def call(self, inputs, states):
        state = states[0]
        for i in range(self._unfolds):
            k1 = self._delta_t * self._f_prime(inputs, state)
            k2 = self._delta_t * self._f_prime(inputs, state + k1 * 0.5)
            k3 = self._delta_t * self._f_prime(inputs, state + k2 * 0.5)
            k4 = self._delta_t * self._f_prime(inputs, state + k3)

            state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

            if self.cell_clip > 0:
                state = tf.clip_by_value(state, -self.cell_clip, self.cell_clip)

        return state, [state]

    def _f_prime(self, inputs, state):
        fused_input = tf.concat([inputs, state], axis=-1)
        input_f_prime = tf.nn.tanh(tf.matmul(fused_input, self.W) + self.b)
        return input_f_prime


class CTGRU(tf.keras.layers.Layer):
    def __init__(self, num_units, M=8, cell_clip=-1, **kwargs):
        super(CTGRU, self).__init__(**kwargs)
        self._num_units = num_units
        self.M = M
        self.cell_clip = cell_clip
        self.ln_tau_table = np.logspace(0, M - 1, num=M, base=10**0.5)

    @property
    def state_size(self):
        return self._num_units * self.M

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        self._input_size = int(input_shape[-1])
        self.tau_r_dense = tf.keras.layers.Dense(self._num_units * self.M, activation=None, name="tau_r")
        self.tau_s_dense = tf.keras.layers.Dense(self._num_units * self.M, activation=None, name="tau_s")
        self.detect_signal_dense = tf.keras.layers.Dense(self._num_units, activation=tf.nn.tanh, name="detect_signal")
        self.built = True

    def call(self, inputs, states):
        h_hat = tf.reshape(states[0], [-1, self._num_units, self.M])
        h = tf.reduce_sum(h_hat, axis=2)

        fused_input = tf.concat([inputs, h], axis=-1)

        ln_tau_r = self.tau_r_dense(fused_input)
        ln_tau_r = tf.reshape(ln_tau_r, shape=[-1, self._num_units, self.M])
        sf_input_r = -tf.square(ln_tau_r - self.ln_tau_table)
        rki = tf.nn.softmax(sf_input_r, axis=2)

        q_input = tf.reduce_sum(rki * h_hat, axis=2)
        reset_value = tf.concat([inputs, q_input], axis=1)
        qk = self.detect_signal_dense(reset_value)
        qk = tf.reshape(qk, [-1, self._num_units, 1])

        ln_tau_s = self.tau_s_dense(fused_input)
        ln_tau_s = tf.reshape(ln_tau_s, shape=[-1, self._num_units, self.M])
        sf_input_s = -tf.square(ln_tau_s - self.ln_tau_table)
        ski = tf.nn.softmax(sf_input_s, axis=2)

        h_hat_next = ((1 - ski) * h_hat + ski * qk) * np.exp(-1.0 / self.ln_tau_table)

        if self.cell_clip > 0:
            h_hat_next = tf.clip_by_value(h_hat_next, -self.cell_clip, self.cell_clip)

        h_next = tf.reduce_sum(h_hat_next, axis=2)
        h_hat_next_flat = tf.reshape(h_hat_next, shape=[-1, self._num_units * self.M])

        return h_next, [h_hat_next_flat]


# Example usage:
ctrnn_cell = CTRNN(num_units=128)
rnn_layer = tf.keras.layers.RNN(ctrnn_cell)
inputs = tf.random.normal([32, 10, 16])  # Example inputs: batch_size=32, time_steps=10, features=16
outputs = rnn_layer(inputs)
