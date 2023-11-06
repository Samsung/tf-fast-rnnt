# Copyright (c)  2023  Samsung Electronics Co., Ltd. All Rights Reserved
# Copyright      2021  Xiaomi Corp.       (author: Daniel Povey, Wei Kang)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import imp
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.python.framework import ops

from .rnnt_loss import do_rnnt_pruning
from .rnnt_loss import get_rnnt_logprobs
from .rnnt_loss import get_rnnt_logprobs_joint
from .rnnt_loss import get_rnnt_logprobs_pruned
from .rnnt_loss import get_rnnt_logprobs_smoothed
from .rnnt_loss import get_rnnt_prune_ranges
from .rnnt_loss import rnnt_loss
from .rnnt_loss import rnnt_loss_pruned
from .rnnt_loss import rnnt_loss_simple
from .rnnt_loss import rnnt_loss_smoothed
from pathlib import Path

__version__ = '1.2'

path = Path(__path__[0])
lib_file = imp.find_module('_tf_fast_rnnt', [path.parent.absolute()])[1]
_tf_fast_rnnt = tf.load_op_library(lib_file)

def mutual_information_recursion(
    px: tf.Tensor,
    py: tf.Tensor,
    boundary: tf.Tensor,
    calc_gradients: bool = False,
) -> Union[Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], tf.Tensor]:
    """A recursion that is useful in computing mutual information between two
    sequences of real vectors, but may be useful more generally in
    sequence-to-sequence tasks where monotonic alignment between pairs of
    sequences is desired.  The definitions of the arguments are definitions that
    would be used when computing this type of mutual information, but you can
    also view them as arbitrary quantities and just make use of the formula
    computed by this function.

    Args:
      px:
        A torch.tf.Tensor of some floating point type, with shape ``[B][S][T+1]``,
        where ``B`` is the batch size, ``S`` is the length of the ``x`` sequence
        (including representations of ``EOS`` symbols but not ``BOS`` symbols),
        and ``T`` is the length of the ``y`` sequence (including representations
        of ``EOS`` symbols but not ``BOS`` symbols).  In the mutual information
        application, ``px[b][s][t]`` would represent the following log odds
        ratio; ignoring the b index on the right to make the notation more
        compact::

          px[b][s][t] =  log [ p(x_s | x_{0..s-1}, y_{0..t-1}) / p(x_s) ]

        This expression also implicitly includes the log-probability of
        choosing to generate an ``x`` value as opposed to a ``y`` value.  In
        practice it might be computed as ``a + b``, where ``a`` is the log
        probability of choosing to extend the sequence of length ``(s,t)``
        with an ``x`` as opposed to a ``y`` value; and ``b`` might in practice
        be of the form::

            log(N exp f(x_s, y_{t-1}) / sum_t'  exp f(x_s, y_t'))

        where ``N`` is the number of terms that the sum over ``t'`` included,
        which might include some or all of the other sequences as well as this
        one.

        Note:
          we don't require ``px`` and py to be contiguous, but the
          code assumes for optimization purposes that the ``T`` axis has
          stride 1.

      py:
        A torch.tf.Tensor of the same dtype as ``px``, with shape ``[B][S+1][T]``,
        representing::

          py[b][s][t] =  log [ p(y_t | x_{0..s-1}, y_{0..t-1}) / p(y_t) ]

        This function does not treat ``x`` and ``y`` differently; the only
        difference is that for optimization purposes we assume the last axis
        (the ``t`` axis) has stride of 1; this is true if ``px`` and ``py`` are
        contiguous.

      boundary:
        If supplied, a torch.LongTensor of shape ``[B][4]``, where each
        row contains ``[s_begin, t_begin, s_end, t_end]``,
        with ``0 <= s_begin <= s_end < S`` and ``0 <= t_begin <= t_end < T``
        (this implies that empty sequences are allowed).
        If not supplied, the values ``[0, 0, S, T]`` will be assumed.
        These are the beginning and one-past-the-last positions in the ``x`` and
        ``y`` sequences respectively, and can be used if not all sequences are
        of the same length.

      calc_gradients:
        Whether to return grads of ``px`` and ``py``, this grad standing for the
        occupation probability is the output of the backward with a
        ``fake gradient`` the ``fake gradient`` is the same as the gradient
        you'd get if you did ``torch.autograd.grad((scores.sum()), [px, py])``.
        This is useful to implement the pruned version of rnnt loss.

    Returns:
      Returns a torch.tf.Tensor of shape ``[B]``, containing the log of the mutual
      information between the b'th pair of sequences.  This is defined by
      the following recursion on ``p[b,s,t]`` (where ``p`` is of shape
      ``[B,S+1,T+1]``), representing a mutual information between sub-sequences
      of lengths ``s`` and ``t``::

             p[b,0,0] = 0.0
        if !modified:
             p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                p[b,s,t-1] + py[b,s,t-1])
        if modified:
             p[b,s,t] = log_add(p[b,s-1,t-1] + px[b,s-1,t-1],
                                p[b,s,t-1] + py[b,s,t-1])

      where we handle edge cases by treating quantities with negative indexes
      as **-infinity**.  The extension to cases where the boundaries are
      specified should be obvious; it just works on shorter sequences with
      offsets into ``px`` and ``py``.
    """
    #assert px.ndim == 3
    B, S, T1 = px.shape
    T = py.shape[-1]
    #assert px.shape[-1] in [T, T + 1]  # if T, then "modified".
    #assert py.shape == (B, S + 1, T)
    #assert px.dtype == py.dtype
    #if boundary is not None:
        #assert boundary.dtype == tf.int32
        #assert boundary.shape == (B, 4)
        #for s_begin, t_begin, s_end, t_end in boundary.numpy().tolist():
            #assert 0 <= s_begin <= s_end <= S
            #assert 0 <= t_begin <= t_end <= T
    
    ans, px_grad, py_grad = _tf_fast_rnnt.fast_rnnt_loss(px, py, boundary, calc_gradients)
    return (ans, (px_grad, py_grad)) if calc_gradients else ans

def cummin(x):
    return _tf_fast_rnnt.cummin(x)

@ops.RegisterGradient("FastRNNTLoss")
def _RNNTLossGrad(op, *grads):
    # tf.assert_equal causes 'Skipping loop optimization for Merge node with control input' warning
    # tf.assert_equal(op.inputs[3], True)  # set calc_gradients or calc_gradients to True if you want to get gradients
    gradpx = op.outputs[1]
    gradpy = op.outputs[2]
    # NOTE since here we are batch first, cannot use _BroadcastMul
    ans_grad = tf.reshape(grads[0], (-1, 1, 1))
    return [ans_grad * gradpx, ans_grad * gradpy, None, None]
