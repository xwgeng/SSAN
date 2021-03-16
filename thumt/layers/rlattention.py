# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

RelaxedBernoulli=tf.contrib.distributions.RelaxedBernoulli
RelaxedOneHotCategorical=tf.contrib.distributions.RelaxedOneHotCategorical

from thumt.layers.attention import split_heads, combine_heads
from thumt.layers.attention import multiplicative_attention

from thumt.layers.nn import linear
from thumt.layers.gumbel import gumbel_softmax


def policy_network(queries, memories, size, num_heads, mode, soft_select=False, keep_prob=None, scope=None):
    with tf.variable_scope(scope, default_name="policy_network", 
                            values=[queries, memories]):
        with tf.variable_scope("input_layer"):
            if memories is None:
                combined = linear(queries, 2 * size, True, True, scope="qk_transform")
                q, k = tf.split(combined, [size, size], axis=-1) 
            else:
                q = linear(queries, size, True, True, scope="q_transform")
                k = linear(memories, size, True, scope="k_transform")
            
            #q = tf.tanh(q)
            #k = tf.tanh(k)

            q = split_heads(q, num_heads)
            k = split_heads(k, num_heads)

            #q = linear(q, size, True, True, scope="q2_transform")
            #k = linear(k, size, True, True, scope="k2_transform")
            
            logits = tf.matmul(q, k, transpose_b=True)
            prob = tf.nn.sigmoid(logits, name="policy")

        with tf.variable_scope("output_layer"):
            if mode == 'train':
                if soft_select == False:
                    sample_prob = RelaxedBernoulli(0.5, logits=logits)
                    y = sample_prob.sample()
                    y_hard = tf.cast(tf.greater(y, 0.5), y.dtype)
                    y = tf.stop_gradient(y_hard - y) + y
                else:
                    y = prob
            else:
                #prob = tf.Print(prob, [tf.reduce_mean(prob), prob], summarize=100000)
                y = tf.cast(tf.greater(prob, 0.5), prob.dtype)
        nce = -tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=prob)
        return y, prob, nce


def multiplicative_rlattention(queries, keys, values, bias, sample, keep_prob=None, name=None, epsilon=1e-6):
    with tf.name_scope(name, default_name="multiplicative_rlattention",
                        values=[queries, keys, values, bias]):
        logits = tf.matmul(queries, keys, transpose_b=True)
        if bias is not None:
            logits += bias

        #weights = tf.nn.softmax(logits, name="attention_weights")
        #weights *= sample
        logits_exp = tf.exp(logits)
        logits_exp *= sample
        logits_sum = tf.reduce_sum(logits_exp, axis=-1, keepdims=True)
        weights = logits_exp / (logits_sum + epsilon) 
        
        weights = tf.Print(weights, [weights], summarize=10000000)
        if keep_prob is not None and keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)

        outputs = tf.matmul(weights, values)

        return {"weights": weights, "outputs": outputs}


def multihead_rlattention(queries, memories, bias, num_heads, key_size,
                        value_size, output_size, mode, rlsan, 
                        keep_prob=None, output=True, state=None, dtype=None,
                        scope=None, soft_select=False):
    """ Multi-head scaled-dot-product attention with input/output
        transformations via RL.

    :param queries: A tensor with shape [batch, length_q, depth_q]
    :param memories: A tensor with shape [batch, length_m, depth_m]
    :param bias: A tensor (see attention_bias)
    :param num_heads: An integer dividing key_size and value_size
    :param key_size: An integer
    :param value_size: An integer
    :param output_size: An integer
    :param keep_prob: A floating point number in (0, 1]
    :param output: Whether to use output transformation
    :param state: An optional dictionary used for incremental decoding
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, heads, length_q, length_kv]
        outputs: A tensor with shape [batch, length_q, depth_v]
    """

    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    with tf.variable_scope(scope, default_name="multihead_rlattention",
                           values=[queries, memories], dtype=dtype):
        next_state = {}

        if memories is None:
            # self attention
            size = key_size * 2 + value_size
            combined = linear(queries, size, True, True, scope="qkv_transform")
            q, k, v = tf.split(combined, [key_size, key_size, value_size], 
                                    axis=-1)

            if state is not None:
                k = tf.concat([state["key"], k], axis=1)
                v = tf.concat([state["value"], v], axis=1)
                next_state["key"] = k
                next_state["value"] = v
        else:
            q = linear(queries, key_size, True, True, scope="q_transform")
            combined = linear(memories, key_size + value_size, True,
                                scope="kv_transform")
            k, v = tf.split(combined, [key_size, value_size], axis=-1)

        # split heads
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)

        key_depth_per_head = key_size // num_heads
        q *= key_depth_per_head ** -0.5

        # attention
        if rlsan:
            sample, prob, nce = policy_network(queries, memories, key_size, num_heads, mode, soft_select, keep_prob)
            results = multiplicative_rlattention(q, k, v, bias, sample, keep_prob)
        else:
            # scale query
            results = multiplicative_attention(q, k, v, bias, keep_prob)

        # combine heads
        weights = results["weights"]
        x = combine_heads(results["outputs"])

        if output:
            outputs = linear(x, output_size, True, True,
                             scope="output_transform")
        else:
            outputs = x

        outputs = {"weights": weights, "outputs": outputs}

        if state is not None:
            outputs["state"] = next_state

        if rlsan:
            outputs["sample"] = sample
            outputs["prob"] = prob
            outputs["nce"] = nce

        return outputs
