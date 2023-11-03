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

import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import tensorflow as tf
import tf_fast_rnnt

from typing import Optional, Tuple, Union

@tf.function
def fix_for_boundary(px: tf.Tensor, boundary: Optional[tf.Tensor] = None) -> tf.Tensor:
    """
    Insert -inf's into `px` in appropriate places if `boundary` is not
    None.  If boundary == None and rnnt_type == "regular", px[:,:,-1] will
    be -infinity, but if boundary is specified, we need px[b,:,boundary[b,3]]
    to be -infinity.

     Args:
          px: a tf.Tensor of of shape [B][S][T+1] (this function is only
              called if rnnt_type == "regular", see other docs for `rnnt_type`)
              px is modified in-place and returned.
           boundary: None, or a tf.Tensor of shape [B][3] containing
              [s_begin, t_begin, s_end, t_end]; we need only t_end.
    """
    if boundary is None:
        return px
    
    shape = tf.shape(px)
    B = shape[0]
    S = shape[1]
    T1 = shape[2]

    bound = boundary[:, 3]
    bound = bound[:, tf.newaxis]
    t = tf.broadcast_to(bound, [B, S])
    a, b = tf.meshgrid(
      tf.range(B), 
      tf.range(S),
      indexing='ij')
    indices = tf.stack([a, b, t], axis=-1)
    updates = tf.broadcast_to(float("-inf"), [B, S])
    px = tf.tensor_scatter_nd_update(px, indices, updates)
    return px

@tf.function
def get_rnnt_logprobs(
    lm: tf.Tensor,
    am: tf.Tensor,
    symbols: tf.Tensor,
    termination_symbol: int,
    rnnt_type: str = "regular",
    boundary: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Reduces RNN-T problem (the simple case, where joiner network is just
    addition), to a compact, standard form that can then be given
    (with boundaries) to mutual_information_recursion().
    This function is called from rnnt_loss_simple(), but may be useful for
    other purposes.

    Args:
      lm:
        Language model part of un-normalized logprobs of symbols, to be added to
        acoustic model part before normalizing.  Of shape::

           [B][S+1][C]

        where B is the batch size, S is the maximum sequence length of
        the symbol sequence, possibly including the EOS symbol; and
        C is size of the symbol vocabulary, including the termination/next-frame
        symbol.
        Conceptually, lm[b][s] is a vector of length [C] representing the
        "language model" part of the un-normalized logprobs of symbols,
        given all symbols *earlier than* s in the sequence.  The reason
        we still need this for position S is that we may still be emitting
        the termination/next-frame symbol at this point.
      am:
        Acoustic-model part of un-normalized logprobs of symbols, to be added
        to language-model part before normalizing.  Of shape::

           [B][T][C]

        where B is the batch size, T is the maximum sequence length of
        the acoustic sequences (in frames); and C is size of the symbol
        vocabulary, including the termination/next-frame symbol.  It reflects
        the "acoustic" part of the probability of any given symbol appearing
        next on this frame.
      symbols:
        A LongTensor of shape [B][S], containing the symbols at each position
        of the sequence.
      termination_symbol:
        The identity of the termination symbol, must be in {0..C-1}
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      rnnt_type:
        Specifies the type of rnnt paths: `regular`, `modified` or `constrained`.
        `regular`: The regular rnnt that taking you to the next frame only if
                   emitting a blank (i.e., emitting a symbol does not take you
                   to the next frame).
        `modified`: A modified version of rnnt that will take you to the next
                    frame either emitting a blank or a non-blank symbol.
        `constrained`: A version likes the modified one that will go to the next
                       frame when you emit a non-blank symbol, but this is done
                       by "forcing" you to take the blank transition from the
                       *next* context on the *current* frame, e.g. if we emit
                       c given "a b" context, we are forced to emit "blank"
                       given "b c" context on the current frame.
    Returns:
        (px, py) (the names are quite arbitrary).
           px: logprobs, of shape [B][S][T+1] if rnnt_type is regular,
                                  [B][S][T] if rnnt_type is not regular.
           py: logprobs, of shape [B][S+1][T]

      in the recursion::

          p[b,0,0] = 0.0
          if rnnt_type == "regular":
             p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                p[b,s,t-1] + py[b,s,t-1])
          if rnnt_type != "regular":
             p[b,s,t] = log_add(p[b,s-1,t-1] + px[b,s-1,t-1],
                                p[b,s,t-1] + py[b,s,t-1])
          .. where p[b][s][t] is the "joint score" of the pair of subsequences
          of length s and t respectively.  px[b][s][t] represents the
          probability of extending the subsequences of length (s,t) by one in
          the s direction, given the particular symbol, and py[b][s][t]
          represents the probability of extending the subsequences of length
          (s,t) by one in the t direction,
          i.e. of emitting the termination/next-frame symbol.

          if rnnt_type == "regular", px[:,:,T] equals -infinity, meaning on the
          "one-past-the-last" frame we cannot emit any symbols.
          This is simply a way of incorporating
          the probability of the termination symbol on the last frame.
    """
    #assert lm.ndim == 3, lm.ndim
    #assert am.ndim == 3, am.ndim
    #assert lm.shape[0] == am.shape[0], (lm.shape[0], am.shape[0])
    #assert lm.shape[2] == am.shape[2], (lm.shape[2], am.shape[2])

    shape = tf.shape(am)
    B = shape[0]
    T = shape[1]
    C = shape[2]
    S = tf.shape(lm)[1] - 1
    #assert symbols.shape == (B, S), symbols.shape
    #assert S >= 1, S
    #assert T >= S, (T, S)
    #assert rnnt_type in ["regular", "modified", "constrained"], rnnt_type

    # subtracting am_max and lm_max is to ensure the probs are in a good range
    # to do exp() without causing underflow or overflow.
    am_max = tf.math.reduce_max(am, axis=2, keepdims=True)  # am_max: [B][T][1]
    lm_max = tf.math.reduce_max(lm, axis=2, keepdims=True)  # lm_max: [B][S+1][1]
    am_probs = tf.math.exp(am - am_max)
    lm_probs = tf.math.exp(lm - lm_max)
    # normalizers: [B][S+1][T]
    normalizers = tf.math.log(
        tf.matmul(lm_probs, am_probs, transpose_b=True) + tf.math.nextafter(0., 1.)
    )

    # add lm_max and am_max to normalizers, to make it as if we had not
    # subtracted am_max and lm_max above.
    normalizers = normalizers + lm_max + tf.transpose(am_max, perm=[0, 2, 1])  # [B][S+1][T]
    index = tf.reshape(symbols, [B, S, 1])
    # px is the probs of the actual symbols..
    px_am = tf.gather_nd(tf.transpose(am, perm=[0, 2, 1]),  # (B, C, T)
                        index,
                        batch_dims=1
                        )
    if rnnt_type == "regular":
        px_am = tf.concat(
            (
                px_am,
                tf.fill(
                    [B, S, 1],
                    float("-inf")
                ),
            ),
            axis=2,
        )  # now: [B][S][T+1], index [:,:,T] has -inf..
    px_lm = tf.expand_dims(tf.gather_nd(
        lm[:, :S], tf.expand_dims(symbols, -1),
        batch_dims=2
    ), -1)
    px = px_am + px_lm  # [B][S][T+1], last slice with indexes out of
    # boundary is  -inf
    # normalizers [B][S+1][T]
    px -= tf.concat([normalizers, tf.zeros([B,S+1,1], dtype=tf.float32)], axis=2)[:, :S, :]

    # py is the probs of termination symbols, of shape [B][S+1][T]
    py_am = tf.expand_dims(am[:, :, termination_symbol], 1)  # [B][1][T]
    py_lm = tf.expand_dims(lm[:, :, termination_symbol], 2)  # [B][S+1][1]
    py = py_am + py_lm - normalizers

    if rnnt_type == "regular":
        px = fix_for_boundary(px, boundary)
    elif rnnt_type == "constrained":
        px += py[:, 1:, :]

    return (px, py)

@tf.function
def rnnt_loss_simple(
    lm: tf.Tensor,
    am: tf.Tensor,
    symbols: tf.Tensor,
    termination_symbol: int,
    boundary: Optional[tf.Tensor] = None,
    rnnt_type: str = "regular",
    delay_penalty: float = 0.0,
    reduction: Optional[str] = "mean",
    return_grad: bool = False,
) -> Union[tf.Tensor, Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]]:
    """A simple case of the RNN-T loss, where the 'joiner' network is just
    addition.

    Args:
      lm:
        language-model part of unnormalized log-probs of symbols, with shape
        (B, S+1, C), i.e. batch, symbol_seq_len+1, num_classes
      am:
        acoustic-model part of unnormalized log-probs of symbols, with shape
        (B, T, C), i.e. batch, frame, num_classes
      symbols:
        the symbol sequences, a LongTensor of shape [B][S], and elements in
        {0..C-1}.
      termination_symbol:
        the termination symbol, with 0 <= termination_symbol < C
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      rnnt_type:
        Specifies the type of rnnt paths: `regular`, `modified` or `constrained`.
        `regular`: The regular rnnt that taking you to the next frame only if
                   emitting a blank (i.e., emitting a symbol does not take you
                   to the next frame).
        `modified`: A modified version of rnnt that will take you to the next
                    frame either emitting a blank or a non-blank symbol.
        `constrained`: A version likes the modified one that will go to the next
                       frame when you emit a non-blank symbol, but this is done
                       by "forcing" you to take the blank transition from the
                       *next* context on the *current* frame, e.g. if we emit
                       c given "a b" context, we are forced to emit "blank"
                       given "b c" context on the current frame.
      delay_penalty: A constant value to penalize symbol delay, this may be
         needed when training with time masking, to avoid the time-masking
         encouraging the network to delay symbols.
         See https://github.com/k2-fsa/k2/issues/955 for more details.
      reduction:
        Specifies the reduction to apply to the output: `none`, `mean` or `sum`.
        `none`: no reduction will be applied.
        `mean`: apply `torch.mean` over the batches.
        `sum`: the output will be summed.
        Default: `mean`
      return_grad:
        Whether to return grads of px and py, this grad standing for the
        occupation probability is the output of the backward with a
        `fake gradient`, the `fake gradient` is the same as the gradient you'd
        get if you did `torch.autograd.grad((-loss.sum()), [px, py])`, note, the
        loss here is the loss with reduction "none".
        This is useful to implement the pruned version of rnnt loss.
    Returns:
       If return_grad is False, returns a tensor of shape (B,), containing the
       total RNN-T loss values for each element of the batch if reduction equals
       to "none", otherwise a scalar with the reduction applied.
       If return_grad is True, the grads of px and py, which is the output of
       backward with a `fake gradient`(see above), will be returned too. And the
       returned value will be a tuple like (loss, (px_grad, py_grad)).
    """
    px, py = get_rnnt_logprobs(
        lm=lm,
        am=am,
        symbols=symbols,
        termination_symbol=termination_symbol,
        boundary=boundary,
        rnnt_type=rnnt_type,
    )

    if delay_penalty > 0.0:
        shape = tf.shape(px)
        B = shape[0]
        S = shape[1]
        T0 = shape[2]
        T = T0 if rnnt_type != "regular" else T0 - 1
        if boundary is None:
            offset = tf.broadcast_to(tf.Tensor(
                (T - 1) / 2, dtype=px.dtype
            ), [B, 1, 1])
        else:
            offset = (boundary[:, 3] - 1) / 2
        penalty = tf.reshape(offset, [B, 1, 1]) - tf.reshape(tf.range(
            T0, dtype=offset.dtype
        ), [1, 1, T0])
        penalty = penalty * delay_penalty
        px += tf.cast(penalty, px.dtype)

    scores_and_grads = tf_fast_rnnt.mutual_information_recursion(
        px=px, py=py, boundary=boundary, return_grad=return_grad
    )
   
    negated_loss = scores_and_grads[0] if return_grad else scores_and_grads
    if reduction == "none":
        loss = -negated_loss
    elif reduction == "mean":
        loss = -torch.mean(negated_loss)
    elif reduction == "sum":
        loss = -tf.reduce_sum(negated_loss)
    else:
        raise ValueError(
            f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"
        )
    return (loss, scores_and_grads[1]) if return_grad else loss

@tf.function
def get_rnnt_logprobs_joint(
    logits: tf.Tensor,
    symbols: tf.Tensor,
    termination_symbol: int,
    boundary: Optional[tf.Tensor] = None,
    rnnt_type: str = "regular",
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Reduces RNN-T problem to a compact, standard form that can then be given
    (with boundaries) to mutual_information_recursion().
    This function is called from rnnt_loss().

    Args:
      logits:
        The output of joiner network, with shape (B, T, S + 1, C),
        i.e. batch, time_seq_len, symbol_seq_len+1, num_classes
      symbols:
        A LongTensor of shape [B][S], containing the symbols at each position
        of the sequence.
      termination_symbol:
        The identity of the termination symbol, must be in {0..C-1}
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      rnnt_type:
        Specifies the type of rnnt paths: `regular`, `modified` or `constrained`.
        `regular`: The regular rnnt that taking you to the next frame only if
                   emitting a blank (i.e., emitting a symbol does not take you
                   to the next frame).
        `modified`: A modified version of rnnt that will take you to the next
                    frame either emitting a blank or a non-blank symbol.
        `constrained`: A version likes the modified one that will go to the next
                       frame when you emit a non-blank symbol, but this is done
                       by "forcing" you to take the blank transition from the
                       *next* context on the *current* frame, e.g. if we emit
                       c given "a b" context, we are forced to emit "blank"
                       given "b c" context on the current frame.
    Returns:
      (px, py) (the names are quite arbitrary)::

          px: logprobs, of shape [B][S][T+1] if rnnt_type is regular,
                                 [B][S][T] if rnnt_type is not regular.
          py: logprobs, of shape [B][S+1][T]

      in the recursion::

         p[b,0,0] = 0.0
         if rnnt_type == "regular":
            p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                               p[b,s,t-1] + py[b,s,t-1])
         if rnnt_type != "regular":
            p[b,s,t] = log_add(p[b,s-1,t-1] + px[b,s-1,t-1],
                               p[b,s,t-1] + py[b,s,t-1])
      .. where p[b][s][t] is the "joint score" of the pair of subsequences of
      length s and t respectively.  px[b][s][t] represents the probability of
      extending the subsequences of length (s,t) by one in the s direction,
      given the particular symbol, and py[b][s][t] represents the probability
      of extending the subsequences of length (s,t) by one in the t direction,
      i.e. of emitting the termination/next-frame symbol.

      if rnnt_type == "regular", px[:,:,T] equals -infinity, meaning on the
      "one-past-the-last" frame we cannot emit any symbols.
      This is simply a way of incorporating
      the probability of the termination symbol on the last frame.
    """
    #assert logits.ndim == 4, logits.ndim
    shape = tf.shape(logits)
    B = shape[0]
    T = shape[1]
    S1 = shape[2]
    C = shape[3]
    S = S1 - 1
    #assert symbols.shape == (B, S), symbols.shape
    #assert S >= 1, S
    #assert T >= S, (T, S)
    #assert rnnt_type in ["regular", "modified", "constrained"], rnnt_type

    normalizers = tf.math.reduce_logsumexp(logits, axis=3)
    normalizers = tf.transpose(normalizers, (0, 2, 1))
    # (B, T, S)
    index = tf.broadcast_to(tf.reshape(symbols, [B, 1, S, 1]), [B, T, S, 1])
    px = tf.gather_nd(
        logits[:, :, :S, :], index, batch_dims=3
    )
    px = tf.transpose(px, (0, 2, 1))

    if rnnt_type == "regular":
        px = tf.concat(
            (
                px,
                tf.fill(
                    (B, S, 1), float("-inf")
                ),
            ),
            axis=2,
        )  # now: [B][S][T+1], index [:,:,T] has -inf..

    px -= tf.concat([normalizers, tf.zeros([B,S+1,1], dtype=tf.float32)], axis=2)[:, :S, :]

    py = (
        tf.transpose(logits[:, :, :, termination_symbol], (0, 2, 1))
    )  # [B][S+1][T]
    py -= normalizers

    if rnnt_type == "regular":
        px = fix_for_boundary(px, boundary)
    elif rnnt_type == "constrained":
        px += py[:, 1:, :]

    return (px, py)

@tf.function
def rnnt_loss(
    logits: tf.Tensor,
    symbols: tf.Tensor,
    termination_symbol: int,
    boundary: Optional[tf.Tensor] = None,
    rnnt_type: str = "regular",
    delay_penalty: float = 0.0,
    reduction: Optional[str] = "mean",
    return_grad: bool = False,
) -> tf.Tensor:
    """A normal RNN-T loss, which uses a 'joiner' network output as input,
    i.e. a 4 dimensions tensor.

    Args:
      logits:
        The output of joiner network, with shape (B, T, S + 1, C),
        i.e. batch, time_seq_len, symbol_seq_len+1, num_classes
      symbols:
        The symbol sequences, a LongTensor of shape [B][S], and elements
        in {0..C-1}.
      termination_symbol:
        the termination symbol, with 0 <= termination_symbol < C
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T] if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      rnnt_type:
        Specifies the type of rnnt paths: `regular`, `modified` or `constrained`.
        `regular`: The regular rnnt that taking you to the next frame only if
                   emitting a blank (i.e., emitting a symbol does not take you
                   to the next frame).
        `modified`: A modified version of rnnt that will take you to the next
                    frame either emitting a blank or a non-blank symbol.
        `constrained`: A version likes the modified one that will go to the next
                       frame when you emit a non-blank symbol, but this is done
                       by "forcing" you to take the blank transition from the
                       *next* context on the *current* frame, e.g. if we emit
                       c given "a b" context, we are forced to emit "blank"
                       given "b c" context on the current frame.
      delay_penalty: A constant value to penalize symbol delay, this may be
         needed when training with time masking, to avoid the time-masking
         encouraging the network to delay symbols.
         See https://github.com/k2-fsa/k2/issues/955 for more details.
      reduction:
        Specifies the reduction to apply to the output: `none`, `mean` or `sum`.
        `none`: no reduction will be applied.
        `mean`: apply `torch.mean` over the batches.
        `sum`: the output will be summed.
        Default: `mean`

    Returns:
      If recursion is `none`, returns a tensor of shape (B,), containing the
      total RNN-T loss values for each element of the batch, otherwise a scalar
      with the reduction applied.
    """
    px, py = get_rnnt_logprobs_joint(
        logits=logits,
        symbols=symbols,
        termination_symbol=termination_symbol,
        boundary=boundary,
        rnnt_type=rnnt_type,
    )
    if delay_penalty > 0.0:
        shape = tf.shape(px)
        B = shape[0]
        S = shape[1]
        T0 = shape[2]
        T = T0 if rnnt_type != "regular" else T0 - 1
        if boundary is None:
            offset = tf.broadcast_to(tf.Tensor(
                (T - 1) / 2, dtype=px.dtype
            ), [B, 1, 1])
        else:
            offset = (boundary[:, 3] - 1) / 2
        penalty = tf.reshape(offset, [B, 1, 1]) - tf.reshape(tf.range(
            T0, dtype=offset.dtype
        ), [1, 1, T0])
        penalty = penalty * delay_penalty
        px += tf.cast(penalty, px.dtype)
   
    scores_and_grads = tf_fast_rnnt.mutual_information_recursion(
        px=px, py=py, boundary=boundary, return_grad=return_grad
    )
    negated_loss = scores_and_grads[0] if return_grad else scores_and_grads
	
    if reduction == "none":
        loss = -negated_loss
    elif reduction == "mean":
        loss = -tf.reduce_mean(negated_loss)
    elif reduction == "sum":
        loss = -tf.reduce_sum(negated_loss)
    else:
        raise ValueError(
            f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"
        )
    return (loss, scores_and_grads[1]) if return_grad else loss

@tf.function
def _monotonic_lower_bound(x: tf.Tensor) -> tf.Tensor:
    """Compute a monotonically increasing lower bound of the tensor `x` on the
    last dimension. The basic idea is: we traverse the tensor in reverse order,
    and update current element with the following statement,
        min_value = min(x[i], min_value)
        x[i] = min_value
    >>> import torch
    >>> x = tf.Tensor([0, 2, 1, 3, 6, 5, 8], dtype=torch.int64)
    >>> _monotonic_lower_bound(x)
    tensor([0, 1, 1, 3, 5, 5, 8], dtype=torch.int64)
    >>> x
    tensor([0, 2, 1, 3, 6, 5, 8], dtype=torch.int64)
    >>> x = torch.randint(20, (3, 6), dtype=torch.int64)
    >>> x
    tensor([[12, 18,  5,  4, 18, 17],
            [11, 14, 14,  3, 10,  4],
            [19,  3,  8, 13,  7, 19]], dtype=torch.int64)
    >>> _monotonic_lower_bound(x)
    tensor([[ 4,  4,  4,  4, 17, 17],
            [ 3,  3,  3,  3,  4,  4],
            [ 3,  3,  7,  7,  7, 19]], dtype=torch.int64)
    Args:
      x:
        The source tensor.
    Returns:
      Returns a tensor which is monotonic on the last dimension
      (i.e. satisfiy `x[i] <= x[i+1]`).
    """
    x = tf.reverse(x, axis=(-1,))
    x = tf_fast_rnnt.cummin(x)
    x = tf.reverse(x, axis=(-1,))
    return x

@tf.function
def _adjust_pruning_lower_bound(
    s_begin: tf.Tensor, s_range: tf.int32
) -> tf.Tensor:
    """Adjust s_begin (pruning lower bounds) to make it satisfy the following
    constraints

      - monotonic increasing, i.e. s_begin[i] <= s_begin[i + 1]
      - start with symbol 0 at first frame.
      - s_begin[i + 1] - s_begin[i] < s_range, which means that we can't skip
        any symbols.

    To make it monotonic increasing, we can use `_monotonic_lower_bound` above,
    which guarantees `s_begin[i] <= s_begin[i + 1]`. The main idea is:
    traverse the array in reverse order and update the elements by
    `min_value = min(a_begin[i], min_value)`.

    The method we used to realize `s_begin[i + 1] - s_begin[i] < s_range`
    constraint is a little tricky. We first transform `s_begin` with
    `s_begin = -(s_begin - (s_range - 1) * tf.range(0,T))`
    then we make the transformed `s_begin` monotonic increasing, after that,
    we transform back `s_begin` with the same formula as the previous
    transformation. The idea is: if we want to make
    `s_begin[i + 1] - s_begin[i] < s_range` we only need to make
    `-(s_begin[i] - i * (s_range - 1))` a non-decreasing array. Proof:

      -(s_begin[i] - i * (s_range - 1)) <= -(s_begin[i + 1] - (i + 1) * (s_range - 1))
                            -s_begin[i] <= -s_begin[i + 1] + (i + 1) * (s_range - 1) - i * (s_range - 1)
                            -s_begin[i] <= -s_begin[i + 1] + s_range - 1
            s_begin[i + 1] - s_begin[i] <= s_range - 1
            s_begin[i + 1] - s_begin[i] < s_range

    The above transformation can not guarantee the start symbol to be 0, so we
    have to make all the elements that less than 0 to be 0 before transforming
    back the `s_begin`.
    """
    # s_begin (B, T)
    shape = tf.shape(s_begin)
    B = shape[0]
    T = shape[1]

    s_begin = _monotonic_lower_bound(s_begin)
    # do the magic transformation
    s_begin = -(
        s_begin - (s_range - 1) * tf.range(0, T)
    )
    # make the transformed tensor to be non-decreasing
    s_begin = _monotonic_lower_bound(s_begin)
    # make start symbol to be zero.
    s_begin = tf.clip_by_value(s_begin, clip_value_min=0, clip_value_max=tf.int32.max)
    # do the magic transformation again to recover s_begin
    s_begin = -(
        s_begin - (s_range - 1) * tf.range(0, T)
    )
    return s_begin


# To get more insight of how we calculate pruning bounds, please read
# chapter 3.2 (Pruning bounds) of our Pruned RNN-T paper
# (https://arxiv.org/pdf/2206.13236.pdf)
@tf.function
def get_rnnt_prune_ranges(
    px_grad: tf.Tensor,
    py_grad: tf.Tensor,
    boundary: tf.Tensor,
    s_range: tf.int32,
) -> tf.Tensor:
    """Get the pruning ranges of normal rnnt loss according to the grads
    of px and py returned by mutual_information_recursion.

    For each sequence with T frames, we will generate a tensor with the shape of
    (T, s_range) containing the information that which symbols will be token
    into consideration for each frame. For example, here is a sequence with 10
    frames and the corresponding symbols are `[A B C D E F]`, if the s_range
    equals 3, one possible ranges tensor will be::

      [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [1, 2, 3],
       [1, 2, 3], [1, 2, 3], [3, 4, 5], [3, 4, 5], [3, 4, 5]]

    which means we only consider `[A B C]` at frame 0, 1, 2, 3, and `[B C D]`
    at frame 4, 5, 6, `[D E F]` at frame 7, 8, 9.

    We can only consider limited number of symbols because frames and symbols
    are monotonic aligned, theoretically it can only generate particular range
    of symbols given a particular frame.

    Note:
      For the generated tensor ranges(assuming batch size is 1), ranges[:, 0]
      is a monotonic increasing tensor from 0 to `len(symbols) - s_range` and
      it satisfies `ranges[t+1, 0] - ranges[t, 0] < s_range` which means we
      won't skip any symbols.

    Args:
      px_grad:
        The gradient of px, see docs in `mutual_information_recursion` for more
        details of px.
      py_grad:
        The gradient of py, see docs in `mutual_information_recursion` for more
        details of py.
      boundary:
        a LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame]
      s_range:
        How many symbols to keep for each frame.
    Returns:
      A tensor with the shape of (B, T, s_range) containing the indexes of the
      kept symbols for each frame.
    """
    shape = tf.shape(px_grad)
    B = shape[0]
    S = shape[1]
    T1 = shape[2]
    T = tf.shape(py_grad)[-1]
    #assert T1 in [T, T + 1], T1
    S1 = S + 1
    #assert py_grad.shape == (B, S + 1, T), py_grad.shape
    #assert boundary.shape == (B, 4), boundary.shape

    #assert S >= 1, S
    #assert T >= S, (T, S)

    # s_range > S means we won't prune out any symbols. To make indexing with
    # ranges run normally, s_range should be equal to or less than ``S + 1``.
    if s_range > S:
        s_range = S + 1

#    if T1 == T:
#        assert (
#            s_range >= 1
#        ), "Pruning range for modified RNN-T should be equal to or greater than 1, or no valid paths could survive pruning."
#    else:
#        assert (
#            s_range >= 2
#        ), "Pruning range for standard RNN-T should be equal to or greater than 2, or no valid paths could survive pruning."

    cumsum = tf.cumsum(py_grad, 1)                                            # (B, S1, T)
    cumsum_pad = tf.zeros((B, 1, T), dtype=py_grad.dtype)
    cumsum = tf.concat((cumsum_pad, cumsum), axis=1)                          # (B, S1 + 1, T)
    blk_sum_grad = cumsum[:, s_range:, :] - cumsum[:, :S1 - s_range + 1, :]   # (B, S1 - s_range + 1, T)
    px_pad = tf.zeros((B, 1, T1), dtype=px_grad.dtype)
    px_grad_pad = tf.concat((px_pad, px_grad), axis=1)                        # (B, S1, T)
    final_grad = blk_sum_grad - px_grad_pad[:, : S1 - s_range + 1, :T]        # (B, S1 - s_range + 1, T)
    s_begin = tf.math.argmax(final_grad, axis=1, output_type=tf.int32)        # (B, T)

    # Handle the values of s_begin in padding positions.
    # -1 here means we fill the position of the last frame (before padding) with
    # padding value which is `len(symbols) - s_range + 1`.
    # This is to guarantee that we reach the last symbol at last frame (before
    # padding).
    # The shape of the mask is (B, T), for example, we have a batch containing
    # 3 sequences, their lengths are 3, 5, 6 (i.e. B = 3, T = 6), so the mask is
    # [[True, True, False, False, False, False],
    #  [True, True, True,  True,  False, False],
    #  [True, True, True,  True,  True,  False]]
    mask = tf.broadcast_to(tf.reshape(tf.range(0, T), [1, T]), [B, T])
    mask = mask < tf.reshape(boundary[:, 3], [B, 1]) - 1

    s_begin_padding = tf.reshape(boundary[:, 2], [B, 1]) - s_range + 1
    # handle the cases where `len(symbols) < s_range`
    s_begin_padding = tf.clip_by_value(s_begin_padding, clip_value_min=0, clip_value_max=tf.int32.max)

    s_begin = tf.where(mask, s_begin, s_begin_padding)

    # adjusting lower bound to make it satisfy some constraints, see docs in
    # `_adjust_pruning_lower_bound` for more details of these constraints.
    # T1 == T here means we are using the non-regular(i.e. modified rnnt or
    # constrained rnnt) version of transducer, the third constraint becomes
    # `s_begin[i + 1] - s_begin[i] < 2`, because it only emits one symbol per
    # frame.
    s_begin = _adjust_pruning_lower_bound(s_begin, 2 if T1 == T else s_range)

    ranges = tf.broadcast_to(tf.reshape(s_begin, [B, T, 1]), [B, T, s_range]) + tf.range(
        s_range)

    return ranges

@tf.function
def do_rnnt_pruning(
    am: tf.Tensor, lm: tf.Tensor, ranges: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Prune the output of encoder(am) and prediction network(lm) with ranges
    generated by `get_rnnt_prune_ranges`.

    Args:
      am:
        The encoder output, with shape (B, T, C)
      lm:
        The prediction network output, with shape (B, S + 1, C)
      ranges:
        A tensor containing the symbol indexes for each frame that we want to
        keep. Its shape is (B, T, s_range), see the docs in
        `get_rnnt_prune_ranges` for more details of this tensor.

    Returns:
      Return the pruned am and lm with shape (B, T, s_range, C)
    """
    # am (B, T, C)
    # lm (B, S + 1, C)
    # ranges (B, T, s_range)
    #assert ranges.shape[0] == am.shape[0], (ranges.shape[0], am.shape[0])
    #assert ranges.shape[0] == lm.shape[0], (ranges.shape[0], lm.shape[0])
    #assert am.shape[1] == ranges.shape[1], (am.shape[1], ranges.shape[1])
    shape = tf.shape(ranges)
    B = shape[0]
    T = shape[1]
    s_range = shape[2]

    shape = tf.shape(lm)
    B = shape[0]
    S1 = shape[1]
    C = shape[2]

    S = S1 - 1

    # (B, T, s_range, C)
    am_pruning = tf.broadcast_to(tf.expand_dims(am, 2), [B, T, s_range, C])
    ranges = tf.reshape(ranges, [B, T*s_range])
    # (B, T, s_range, C)
    lm_pruning = tf.gather(
        lm,
        ranges,
        batch_dims=1,
        axis=1
    )
    lm_pruning = tf.reshape(lm_pruning, [B, T, s_range, C])
    return am_pruning, lm_pruning

@tf.function
def _roll_by_shifts(src: tf.Tensor, shifts: tf.Tensor):
    """Roll tensor with different shifts for each row.

    Note:
      We assume the src is a 3 dimensions tensor and roll the last dimension.

    Example:

      >>> src = tf.range(15).reshape((1,3,5))
      >>> src
      tensor([[[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14]]])
      >>> shift = tf.Tensor([[1, 2, 3]])
      >>> shift
      tensor([[1, 2, 3]])
      >>> _roll_by_shifts(src, shift)
      tensor([[[ 4,  0,  1,  2,  3],
               [ 8,  9,  5,  6,  7],
               [12, 13, 14, 10, 11]]])
    """
    #assert tf.rank(src) == 3, tf.rank(src)
    shape = tf.shape(src)
    B = shape[0]
    T = shape[1]
    S = shape[2]

    #assert shifts.shape == (B, T), shifts.shape

    index = tf.range(S)
    index = tf.reshape(index, [1, S])
    index = tf.tile(index, [T, 1])
    index = index[tf.newaxis,:]
    index = tf.tile(index, [B, 1, 1])
    index = (index - tf.reshape(shifts, [B, T, 1])) % S
    index = tf.expand_dims(index, -1)
    return tf.gather_nd(src, index, batch_dims=2)

@tf.function
def get_rnnt_logprobs_pruned(
    logits: tf.Tensor,
    symbols: tf.Tensor,
    ranges: tf.Tensor,
    termination_symbol: int,
    boundary: tf.Tensor,
    rnnt_type: str = "regular",
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Construct px, py for mutual_information_recursion with pruned output.

    Args:
      logits:
        The pruned output of joiner network, with shape (B, T, s_range, C)
      symbols:
        The symbol sequences, a LongTensor of shape [B][S], and elements in
        {0..C-1}.
      ranges:
        A tensor containing the symbol ids for each frame that we want to keep.
        It is a LongTensor of shape ``[B][T][s_range]``, where ``ranges[b,t,0]``
        contains the begin symbol ``0 <= s <= S - s_range + 1``, such that
        ``logits[b,t,:,:]`` represents the logits with positions
        ``s, s + 1, ... s + s_range - 1``.
        See docs in :func:`get_rnnt_prune_ranges` for more details of what
        ranges contains.
      termination_symbol:
        the termination symbol, with 0 <= termination_symbol < C
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      rnnt_type:
        Specifies the type of rnnt paths: `regular`, `modified` or `constrained`.
        `regular`: The regular rnnt that taking you to the next frame only if
                   emitting a blank (i.e., emitting a symbol does not take you
                   to the next frame).
        `modified`: A modified version of rnnt that will take you to the next
                    frame either emitting a blank or a non-blank symbol.
        `constrained`: A version likes the modified one that will go to the next
                       frame when you emit a non-blank symbol, but this is done
                       by "forcing" you to take the blank transition from the
                       *next* context on the *current* frame, e.g. if we emit
                       c given "a b" context, we are forced to emit "blank"
                       given "b c" context on the current frame.
    Returns:
      (px, py) (the names are quite arbitrary)::
          px: logprobs, of shape [B][S][T+1] if rnnt_type is regular,
                                 [B][S][T] if rnnt_type is not regular.
          py: logprobs, of shape [B][S+1][T]
      in the recursion::
         p[b,0,0] = 0.0
         if rnnt_type == "regular":
            p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                               p[b,s,t-1] + py[b,s,t-1])
         if rnnt_type != "regular":
            p[b,s,t] = log_add(p[b,s-1,t-1] + px[b,s-1,t-1],
                               p[b,s,t-1] + py[b,s,t-1])
      .. where p[b][s][t] is the "joint score" of the pair of subsequences of
      length s and t respectively.  px[b][s][t] represents the probability of
      extending the subsequences of length (s,t) by one in the s direction,
      given the particular symbol, and py[b][s][t] represents the probability
      of extending the subsequences of length (s,t) by one in the t direction,
      i.e. of emitting the termination/next-frame symbol.
      if `rnnt_type == "regular"`, px[:,:,T] equals -infinity, meaning on the
      "one-past-the-last" frame we cannot emit any symbols.
      This is simply a way of incorporating
      the probability of the termination symbol on the last frame.
    """
    # logits (B, T, s_range, C)
    # symbols (B, S)
    # ranges (B, T, s_range)
    #assert logits.ndim == 4, logits.ndim
    shape = tf.shape(logits)
    B = shape[0]
    T = shape[1]
    s_range = shape[2]
    C = shape[3]

    #assert ranges.shape == (B, T, s_range), ranges.shape
    shape = tf.shape(symbols)
    B = shape[0]
    S = shape[1]

    #assert S >= 1, S
    #assert T >= S, (T, S)
    #assert rnnt_type in ["regular", "modified", "constrained"], rnnt_type

    normalizers = tf.math.reduce_logsumexp(logits, axis=3)
    symbols_with_terminal = tf.concat(
        (
            symbols,
            tf.reshape(tf.tile(
                [termination_symbol], [B]
                ), [B, 1]),
        ),
        axis=1,
    )

    # (B, T, s_range)
    pruned_symbols = tf.gather_nd(
        tf.broadcast_to(tf.expand_dims(symbols_with_terminal, 1), [B, T, S + 1]),
        ranges[:, :, :, tf.newaxis],
        batch_dims=2
    )

    # (B, T, s_range)
    px = tf.squeeze(tf.gather_nd(
        logits, tf.reshape(pruned_symbols, [B, T, s_range, 1, 1]), batch_dims=3
    ), -1)

    px = px - normalizers

    # (B, T, S) with index larger than s_range in axis 2 fill with -inf
    px = tf.concat(
        (
            px,
            tf.fill(
                (B, T, S + 1 - s_range),
                float("-inf")
            ),
        ),
        axis=2,
    )

    # (B, T, S) with index out of s_range in axis 2 fill with -inf
    px = _roll_by_shifts(px, ranges[:, :, 0])[:, :, :S]

    px = tf.transpose(px, (0, 2, 1))

    if rnnt_type == "regular":
        px = tf.concat(
            (
                px,
                tf.fill(
                    (B, S, 1), float("-inf")
                ),
            ),
            axis=2,
        )  # now: [B][S][T+1], index [:,:,T] has -inf..

    py = logits[:, :, :, termination_symbol]  # (B, T, s_range)
    py = py - normalizers

    # (B, T, S + 1) with index larger than s_range in axis 2 filled with -inf
    py = tf.concat(
        (
            py,
            tf.fill(
                (B, T, S + 1 - s_range),
                float("-inf")
            ),
        ),
        axis=2,
    )

    # (B, T, S + 1) with index out of s_range in axis 2 fill with -inf
    py = _roll_by_shifts(py, ranges[:, :, 0])
    # (B, S + 1, T)
    py = tf.transpose(py, (0, 2, 1))

    if rnnt_type == "regular":
        px = fix_for_boundary(px, boundary)
    elif rnnt_type == "constrained":
        px += py[:, 1:, :]

    return (px, py)

@tf.function
def rnnt_loss_pruned(
    logits: tf.Tensor,
    symbols: tf.Tensor,
    ranges: tf.Tensor,
    termination_symbol: int,
    boundary: tf.Tensor = None,
    rnnt_type: str = "regular",
    delay_penalty: float = 0.0,
    reduction: Optional[str] = "mean",
    training: bool = False,
) -> tf.Tensor:
    """A RNN-T loss with pruning, which uses the output of a pruned 'joiner'
    network as input, i.e. a 4 dimensions tensor with shape (B, T, s_range, C),
    s_range means the number of symbols kept for each frame.

    Args:
      logits:
        The pruned output of joiner network, with shape (B, T, s_range, C),
        i.e. batch, time_seq_len, prune_range, num_classes
      symbols:
        A LongTensor of shape [B][S], containing the symbols at each position
        of the sequence.
      ranges:
        A tensor containing the symbol ids for each frame that we want to keep.
        It is a LongTensor of shape ``[B][T][s_range]``, where ``ranges[b,t,0]``
        contains the begin symbol ``0 <= s <= S - s_range + 1``, such that
        ``logits[b,t,:,:]`` represents the logits with positions
        ``s, s + 1, ... s + s_range - 1``.
        See docs in :func:`get_rnnt_prune_ranges` for more details of what
        ranges contains.
      termination_symbol:
        The identity of the termination symbol, must be in {0..C-1}
      boundary:
        a LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T] if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      rnnt_type:
        Specifies the type of rnnt paths: `regular`, `modified` or `constrained`.
        `regular`: The regular rnnt that taking you to the next frame only if
                   emitting a blank (i.e., emitting a symbol does not take you
                   to the next frame).
        `modified`: A modified version of rnnt that will take you to the next
                    frame either emitting a blank or a non-blank symbol.
        `constrained`: A version likes the modified one that will go to the next
                       frame when you emit a non-blank symbol, but this is done
                       by "forcing" you to take the blank transition from the
                       *next* context on the *current* frame, e.g. if we emit
                       c given "a b" context, we are forced to emit "blank"
                       given "b c" context on the current frame.
      delay_penalty: A constant value to penalize symbol delay, this may be
         needed when training with time masking, to avoid the time-masking
         encouraging the network to delay symbols.
         See https://github.com/k2-fsa/k2/issues/955 for more details.
      reduction:
        Specifies the reduction to apply to the output: `none`, `mean` or `sum`.
        `none`: no reduction will be applied.
        `mean`: apply `torch.mean` over the batches.
        `sum`: the output will be summed.
        Default: `mean`
    Returns:
      If reduction is `none`, returns a tensor of shape (B,), containing the
      total RNN-T loss values for each sequence of the batch, otherwise a scalar
      with the reduction applied.
    """
    px, py = get_rnnt_logprobs_pruned(
        logits=logits,
        symbols=symbols,
        ranges=ranges,
        termination_symbol=termination_symbol,
        boundary=boundary,
        rnnt_type=rnnt_type,
    )
 
    if delay_penalty > 0.0:
        shape = tf.shape(px)
        B = shape[0]
        S = shape[1]
        T0 = shape[2]

        T = T0 if rnnt_type != "regular" else T0 - 1
        if boundary is None:
            offset = tf.broadcast_to(tf.Tensor(
                (T - 1) / 2, dtype=px.dtype 
            ), [B, 1, 1])
        else:
            offset = (boundary[:, 3] - 1) / 2
        penalty = tf.reshape(offset, [B, 1, 1]) - tf.reshape(tf.range(
            T0, dtype=offset.dtype
        ), [1, 1, T0])
        penalty = penalty * delay_penalty
        px += tf.cast(penalty, px.dtype)
    
    if training:
      negated_loss, _ = tf_fast_rnnt.mutual_information_recursion(px=px, py=py, boundary=boundary, return_grad=training)
    else:
      negated_loss = tf_fast_rnnt.mutual_information_recursion(px=px, py=py, boundary=boundary, return_grad=training)

    if reduction == "none":
        return -negated_loss
    elif reduction == "mean":
        return -tf.reduce_mean(negated_loss)
    elif reduction == "sum":
        return -tf.reduce_sum(negated_loss)
    else:
        raise ValueError(
            f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"
        )

@tf.function
def get_rnnt_logprobs_smoothed(
    lm: tf.Tensor,
    am: tf.Tensor,
    symbols: tf.Tensor,
    termination_symbol: int,
    lm_only_scale: float = 0.1,
    am_only_scale: float = 0.1,
    boundary: Optional[tf.Tensor] = None,
    rnnt_type: str = "regular",
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Reduces RNN-T problem (the simple case, where joiner network is just
    addition), to a compact, standard form that can then be given
    (with boundaries) to mutual_information_recursion().
    This version allows you to make the loss-function one of the form::

          lm_only_scale * lm_probs +
          am_only_scale * am_probs +
          (1-lm_only_scale-am_only_scale) * combined_probs

    where lm_probs and am_probs are the probabilities given the lm and acoustic
    model independently.

    This function is called from
    :func:`rnnt_loss_smoothed`, but may be useful for other purposes.

    Args:
      lm:
        Language model part of un-normalized logprobs of symbols, to be added to
        acoustic model part before normalizing.  Of shape::

           [B][S+1][C]

        where B is the batch size, S is the maximum sequence length of
        the symbol sequence, possibly including the EOS symbol; and
        C is size of the symbol vocabulary, including the termination/next-frame
        symbol.
        Conceptually, lm[b][s] is a vector of length [C] representing the
        "language model" part of the un-normalized logprobs of symbols,
        given all symbols *earlier than* s in the sequence.  The reason
        we still need this for position S is that we may still be emitting
        the termination/next-frame symbol at this point.
      am:
        Acoustic-model part of un-normalized logprobs of symbols, to be added
        to language-model part before normalizing.  Of shape::

           [B][T][C]

        where B is the batch size, T is the maximum sequence length of
        the acoustic sequences (in frames); and C is size of the symbol
        vocabulary, including the termination/next-frame symbol.  It reflects
        the "acoustic" part of the probability of any given symbol appearing
        next on this frame.
      symbols:
        A LongTensor of shape [B][S], containing the symbols at each position
        of the sequence.
      termination_symbol:
        The identity of the termination symbol, must be in {0..C-1}
      lm_only_scale:
        the scale on the "LM-only" part of the loss.
      am_only_scale:
        the scale on the "AM-only" part of the loss, for which we use
        an "averaged" LM (averaged over all histories, so effectively unigram).
      boundary:
        a optional LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      modified: if True, each time a real symbol is consumed a frame will
        also be consumed, so at most 1 symbol can appear per frame.
      rnnt_type:
        Specifies the type of rnnt paths: `regular`, `modified` or `constrained`.
        `regular`: The regular rnnt that taking you to the next frame only if
                   emitting a blank (i.e., emitting a symbol does not take you
                   to the next frame).
        `modified`: A modified version of rnnt that will take you to the next
                    frame either emitting a blank or a non-blank symbol.
        `constrained`: A version likes the modified one that will go to the next
                       frame when you emit a non-blank symbol, but this is done
                       by "forcing" you to take the blank transition from the
                       *next* context on the *current* frame, e.g. if we emit
                       c given "a b" context, we are forced to emit "blank"
                       given "b c" context on the current frame.
    Returns:
        (px, py) (the names are quite arbitrary).
           px: logprobs, of shape [B][S][T+1] if rnnt_type == "regular",
                                  [B][S][T] if rnnt_type != "regular".
           py: logprobs, of shape [B][S+1][T]

        in the recursion::

          p[b,0,0] = 0.0
          if rnnt_type == "regular":
             p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                p[b,s,t-1] + py[b,s,t-1])
          if rnnt_type != "regular":
             p[b,s,t] = log_add(p[b,s-1,t-1] + px[b,s-1,t-1],
                                p[b,s,t-1] + py[b,s,t-1])
          .. where p[b][s][t] is the "joint score" of the pair of subsequences
          of length s and t respectively.  px[b][s][t] represents the
          probability of extending the subsequences of length (s,t) by one in
          the s direction, given the particular symbol, and py[b][s][t]
          represents the probability of extending the subsequences of length
          (s,t) by one in the t direction,
          i.e. of emitting the termination/next-frame symbol.

          px[:,:,T] equals -infinity, meaning on the "one-past-the-last" frame
          we cannot emit any symbols.  This is simply a way of incorporating
          the probability of the termination symbol on the last frame.
    """
    #assert lm.ndim == 3, lm.ndim
    #assert am.ndim == 3, am.ndim
    #assert lm.shape[0] == am.shape[0], (lm.shape[0], am.shape[0])
    #assert lm.shape[2] == am.shape[2], (lm.shape[2], am.shape[2])
    shape = tf.shape(am)
    B = shape[0]
    T = shape[1]
    C = shape[2]

    S = tf.shape(lm)[1] - 1
    #assert symbols.shape == (B, S), symbols.shape
    #assert S >= 1, S
    #assert T >= S, (T, S)
    #assert rnnt_type in ["regular", "modified", "constrained"], rnnt_type

    # Caution: some parts of this code are a little less clear than they could
    # be due to optimizations.  In particular it may not be totally obvious that
    # all of the logprobs here are properly normalized.  We test that
    # this code is invariant to adding constants in the appropriate ways.

    # subtracting am_max and lm_max is to ensure the probs are in a good range
    # to do exp() without causing underflow or overflow.
    am_max = tf.math.reduce_max(am, axis=2, keepdims=True)  # am_max: [B][T][1]
    lm_max = tf.math.reduce_max(lm, axis=2, keepdims=True)  # lm_max: [B][S+1][1]
    am_probs = tf.math.exp(am - am_max)
    lm_probs = tf.math.exp(lm - lm_max)
    # normalizers: [B][S+1][T]
    normalizers = tf.math.log(
        tf.matmul(lm_probs, am_probs, transpose_b=True) + tf.math.nextafter(0., 1.)
    )

    # normalizer per frame, if we take only the LM probs by themselves
    lmonly_normalizers = tf.math.reduce_sum(lm_probs, 
        axis=2, keepdims=True
    )  # lmonly_normalizers: [B][S+1][1]
    unigram_lm = tf.math.reduce_mean(lm_probs / lmonly_normalizers,
        axis=(0, 1), keepdims=True) + tf.math.nextafter(0., 1.)  # [1][1][C]
    amonly_normalizers = (
        tf.math.log(tf.reshape(tf.linalg.matvec(tf.reshape(am_probs, [-1, C]), tf.reshape(unigram_lm, [C]))
        , [B, T, 1]))
        + am_max
    )  # [B][T][1]
    amonly_normalizers = tf.transpose(amonly_normalizers, perm=[0, 2, 1])  # [B][1][T]
    unigram_lm = tf.math.log(unigram_lm)
    lmonly_normalizers = (
        tf.math.log(lmonly_normalizers) + lm_max
    )  # [B][S+1][1], log-normalizer, used for LM-only part of prob.

    # add lm_max and am_max to normalizers, to make it as if we had not
    # subtracted am_max and lm_max above.
    normalizers = normalizers + lm_max + tf.transpose(am_max, perm=[0, 2, 1])  # [B][S+1][T]

    # px is the probs of the actual symbols (not yet normalized)..
    index = tf.reshape(symbols, [B, S, 1])
    # px is the probs of the actual symbols..
    px_am = tf.gather_nd(tf.transpose(am, perm=[0, 2, 1]),  # (B, C, T)
                        index,
                        batch_dims=1
                        )
    if rnnt_type == "regular":
        px_am = tf.concat(
            (
                px_am,
                tf.fill(
                    (B, S, 1),
                    float("-inf")                    
                ),
            ),
            axis=2,
        )  # now: [B][S][T+1], index [:,:,T] has -inf..
    px_lm = tf.expand_dims(tf.gather_nd(
        lm[:, :S], tf.expand_dims(symbols, -1),
        batch_dims=2
    ), -1)

    px_lm_unigram = tf.gather(tf.reshape(unigram_lm, [-1]),
        tf.expand_dims(symbols, -1)
    )  # [B][S][1]

    px = px_am + px_lm  # [B][S][T+1] if not modified, [B][S][T] if modified
    px -= tf.concat([normalizers, tf.zeros([B,S+1,1], dtype=tf.float32)], axis=2)[:, :S, :]

    px_amonly = (
        px_am + px_lm_unigram
    )  # [B][S][T+1] if !modified; [B][S][T] if modified.

    px_amonly -= tf.concat([amonly_normalizers, tf.zeros([B,1,1], dtype=tf.float32)], axis=2)
    px_lmonly = px_lm - lmonly_normalizers[:, :S, :]

    # py is the probs of termination symbols, of shape [B][S+1][T]
    py_am = tf.expand_dims(am[:, :, termination_symbol], 1)  # [B][1][T]
    py_lm = tf.expand_dims(lm[:, :, termination_symbol], 2)  # [B][S+1][1]
    py = py_am + py_lm - normalizers

    py_lm_unigram = unigram_lm[0][0][termination_symbol]  # scalar, normalized..
    py_amonly = py_am + py_lm_unigram - amonly_normalizers  # [B][1][T]
    py_lmonly = py_lm - lmonly_normalizers  # [B][S+1][1]

    combined_scale = 1.0 - lm_only_scale - am_only_scale

    # We need to avoid exact zeros in the scales because otherwise multiplying
    # -inf by zero generates nan.
    if lm_only_scale == 0.0:
        lm_only_scale = 1.0e-20
    if am_only_scale == 0.0:
        am_only_scale = 1.0e-20

    px_interp = (
        px * combined_scale
        + px_lmonly * lm_only_scale
        + px_amonly * am_only_scale
    )
    py_interp = (
        py * combined_scale
        + py_lmonly * lm_only_scale
        + py_amonly * am_only_scale
    )

    if rnnt_type == "regular":
        px_interp = fix_for_boundary(px_interp, boundary)
    elif rnnt_type == "constrained":
        px_interp += py_interp[:, 1:, :]

    return (px_interp, py_interp)

@tf.function
def rnnt_loss_smoothed(
    lm: tf.Tensor,
    am: tf.Tensor,
    symbols: tf.Tensor,
    termination_symbol: int,
    lm_only_scale: float = 0.1,
    am_only_scale: float = 0.1,
    boundary: Optional[tf.Tensor] = None,
    rnnt_type: str = "regular",
    delay_penalty: float = 0.0,
    reduction: Optional[str] = "mean",
    return_grad: bool = False,
) -> Union[Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], tf.Tensor]:
    """A simple case of the RNN-T loss, where the 'joiner' network is just
    addition.

    Args:
      lm:
        language-model part of unnormalized log-probs of symbols, with shape
        (B, S+1, C), i.e. batch, symbol_seq_len+1, num_classes.
        These are assumed to be well-normalized, in the sense that we could
        use them as probabilities separately from the am scores
      am:
        acoustic-model part of unnormalized log-probs of symbols, with shape
        (B, T, C), i.e. batch, frame, num_classes
      symbols:
        the symbol sequences, a LongTensor of shape [B][S], and elements in
        {0..C-1}.
      termination_symbol:
        the termination symbol, with 0 <= termination_symbol < C
      lm_only_scale:
        the scale on the "LM-only" part of the loss.
      am_only_scale:
        the scale on the "AM-only" part of the loss, for which we use
        an "averaged" LM (averaged over all histories, so effectively unigram).
      boundary:
        a LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as
        [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
      rnnt_type:
        Specifies the type of rnnt paths: `regular`, `modified` or `constrained`.
        `regular`: The regular rnnt that taking you to the next frame only if
                   emitting a blank (i.e., emitting a symbol does not take you
                   to the next frame).
        `modified`: A modified version of rnnt that will take you to the next
                    frame either emitting a blank or a non-blank symbol.
        `constrained`: A version likes the modified one that will go to the next
                       frame when you emit a non-blank symbol, but this is done
                       by "forcing" you to take the blank transition from the
                       *next* context on the *current* frame, e.g. if we emit
                       c given "a b" context, we are forced to emit "blank"
                       given "b c" context on the current frame.
      delay_penalty: A constant value to penalize symbol delay, this may be
         needed when training with time masking, to avoid the time-masking
         encouraging the network to delay symbols.
         See https://github.com/k2-fsa/k2/issues/955 for more details.
      reduction:
        Specifies the reduction to apply to the output: `none`, `mean` or `sum`.
        `none`: no reduction will be applied.
        `mean`: apply `torch.mean` over the batches.
        `sum`: the output will be summed.
        Default: `mean`
      return_grad:
        Whether to return grads of px and py, this grad standing for the
        occupation probability is the output of the backward with a
        `fake gradient`, the `fake gradient` is the same as the gradient you'd
        get if you did `torch.autograd.grad((-loss.sum()), [px, py])`, note, the
        loss here is the loss with reduction "none".
        This is useful to implement the pruned version of rnnt loss.

    Returns:
       If return_grad is False, returns a tensor of shape (B,), containing the
       total RNN-T loss values for each element of the batch if reduction equals
       to "none", otherwise a scalar with the reduction applied.
       If return_grad is True, the grads of px and py, which is the output of
       backward with a `fake gradient`(see above), will be returned too. And the
       returned value will be a tuple like (loss, (px_grad, py_grad)).
    """
    px, py = get_rnnt_logprobs_smoothed(
        lm=lm,
        am=am,
        symbols=symbols,
        termination_symbol=termination_symbol,
        lm_only_scale=lm_only_scale,
        am_only_scale=am_only_scale,
        boundary=boundary,
        rnnt_type=rnnt_type,
    )

    if delay_penalty > 0.0:
        shape = tf.shape(px)
        B = shape[0]
        S = shape[1]
        T0 = shape[2]

        T = T0 if rnnt_type != "regular" else T0 - 1
        if boundary is None:
            offset = tf.broadcast_to(tf.Tensor(
                (T - 1) / 2, dtype=px.dtype 
            ), [B, 1, 1])
        else:
            offset = (boundary[:, 3] - 1) / 2
        penalty = tf.reshape(offset, [B, 1, 1]) - tf.reshape(tf.range(
            T0, dtype=offset.dtype
        ), [1, 1, T0])
        penalty = penalty * delay_penalty
        px += tf.cast(penalty, px.dtype)

    scores_and_grads = tf_fast_rnnt.mutual_information_recursion(
        px=px, py=py, boundary=boundary, return_grad=return_grad
    )
    negated_loss = scores_and_grads[0] if return_grad else scores_and_grads
    if reduction == "none":
        loss = -negated_loss
    elif reduction == "mean":
        loss = -tf.reduce_mean(negated_loss)
    elif reduction == "sum":
        loss = -tf.reduce_sum(negated_loss)
    else:
        raise ValueError(
            f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"
        )
    return (loss, scores_and_grads[1]) if return_grad else loss
