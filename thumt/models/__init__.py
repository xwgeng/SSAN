# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.seq2seq
import thumt.models.rnnsearch
import thumt.models.rnnsearch_l2
import thumt.models.rnnsearch_l2_res
import thumt.models.transformer
import thumt.models.gated_transformer
import thumt.models.gated_gru_transformer
import thumt.models.rlsan


def get_model(name):
    name = name.lower()

    if name == "rnnsearch":
        return thumt.models.rnnsearch.RNNsearch
    elif name == "rnnsearch_l2":
        return thumt.models.rnnsearch_l2.RNNsearch
    elif name == "rnnsearch_l2_res":
        return thumt.models.rnnsearch_l2_res.RNNsearch
    elif name == "seq2seq":
        return thumt.models.seq2seq.Seq2Seq
    elif name == "transformer":
        return thumt.models.transformer.Transformer
    elif name == "gated_transformer":
        return thumt.models.gated_transformer.Transformer
    elif name == "gated_gru_transformer":
        return thumt.models.gated_gru_transformer.Transformer
    elif name == "rlsan":
        return thumt.models.rlsan.RLSAN
    else:
        raise LookupError("Unknown model %s" % name)
