/**
 * @copyright
 * Copyright (c)  2023  Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright      2021  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
**/

#include <cooperative_groups.h>
#include <cuda/std/limits>

#include "tf_fast_rnnt/csrc/mutual_information.h"

#define GET_MACRO(_0, _1, _2, NAME, ...) NAME
#define Assert(...) GET_MACRO(_0, ##__VA_ARGS__, Assert2, Assert1, Assert0)(__VA_ARGS__)
#define Assert1(expression) assert((expression))
#define Assert2(expression, msg) assert((expression) && (msg))

namespace tf_fast_rnnt {
/*
  Forward of mutual_information.  Each thread block computes blocks of the 'p'
  array of (s, t) shape equal to (BLOCK_SIZE, BLOCK_SIZE), e.g. (32, 32).
  Thread-blocks loop over such blocks, but they might loop only once if there is
  not that much data to process.  We sequentially launch thread groups in
  such a way that thread-blocks within a group do not depend on each other
  (see the "iter" parameter).  The blocks of the 'image' (i.e. of the p matrix)
  that each group handles are arranged in a diagonal.

  Template args:
      scalar_t: the floating-point type, e.g. float, double; maybe eventually
                half, although I think we don't support LogAdd for half yet.
      BLOCK_SIZE: an integer power of two no greater than 32 (this limitation
                is because we assume BLOCK_SIZE + 1 <= 64 in some data-loading
                code).
  Args:
      px:    Tensor of shape [B][S][T + 1], if !modified; [B][S][T] if modified;
             may be interpreted as the log-odds ratio of
             generating the next x in the sequence, i.e.
             px(b, s, t) is the log of
                p(x_s | x_0..x_{s-1}, y_0..y_{s-1}) / p(x_s),
             i.e. the log-prob of generating x_s given subsequences of lengths
             (s, t), divided by the prior probability of generating x_s.  (See
             mutual_information.py for more info).
      py:     The log-odds ratio of generating the next y in the sequence.
              Shape [B][S + 1][T]
      p:      This function writes to p(b, s, t) the mutual information between
              sub-sequences of x and y of length s and t respectively, from the
              b'th sequences in the batch.  Its shape is [B][S + 1][T + 1].
              Concretely, this function implements the following recursion,
              in the case where s_begin == t_begin == 0:

               p[b,0,0] = 0.0
             if not `modified`:
               p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                  p[b,s,t-1] + py[b,s,t-1])          (eq. 0)
             if `modified`:
               p[b,s,t] = log_add(p[b,s-1,t-t] + px[b,s-1,t-1],
                                  p[b,s,t-1] + py[b,s,t-1])          (eq. 0)

             treating values with any -1 index as -infinity.
              .. if `boundary` is set, we start from p[b,s_begin,t_begin]=0.0.
   boundary:  If set, a tensor of shape [B][4] of type int64_t, which
              contains, where for each batch element b, boundary[b] equals
              [s_begin, t_begin, s_end, t_end]
              which are the beginning and end (i.e. one-past-the-last) of the
              x and y sequences that we should process.  Otherwise, must be
              a tensor of shape [0][0] of type int64_t; the values will
              default to (0, 0, S, T).
     ans: a tensor `ans` of shape [B], where this function will set
            ans(b) = p[b][s_end][t_end],
            with s_end and t_end being (S, T) if `boundary` was specified,
            and (boundary[b][2], boundary[b][3]) otherwise.
            `ans` represents the mutual information between each pair of
            sequences (i.e. x[b] and y[b], although the sequences are not
            supplied directly to this function).

   The block-dim and grid-dim must both be 1-dimensional, and the block-dim must
   be at least 128.
*/

// do not use call by reference here! use call by value to copy parameters from host to device.
template <typename scalar_t,
          int BLOCK_SIZE> // e.g. BLOCK_SIZE == 16 or 32.
__global__ void mutual_information_kernel(
    // B, S, T + 1, i.e. batch, x_seq_length, y_seq_length + 1
    typename tf::TTypes<scalar_t, 3>::ConstTensor px,
    typename tf::TTypes<scalar_t, 3>::ConstTensor py, // B, S + 1, T.
    // B, S + 1, T + 1.  This is an output.
    typename tf::TTypes<scalar_t, 3>::Tensor p,
    // B, 4;  or 0, 0 if boundaries are the defaults (0, 0, S, T)
    typename tf::TTypes<int64_t>::ConstMatrix boundary,
    typename tf::TTypes<scalar_t>::Vec ans, // [B]
    int iter) { // This kernel is sequentially called with 'iter' = 0, 1, 2 and
                // so on, up to num_iters - 1 where num_iters = num_s_blocks +
                // num_t_blocks - 1 num_s_blocks = S / BLOCK_SIZE + 1
                // num_t_blocks = T / BLOCK_SIZE + 1
                // so that each group depends on the previous group...
  const int B = px.dimension(0), S = px.dimension(1), T = py.dimension(2);
  const bool modified = (px.dimension(2) == T);
  const int t_offset = (modified ? -1 : 0); // see CPU code to understand.

  // num_s_blocks and num_t_blocks are the number of blocks we need to cover the
  // array of dimension (S, T) with blocks of this dimension, in the s and t directions
  // respectively.
  // You can read the following expressions as simplifications of, for example,
  // num_s_blocks = ((S + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE,
  // i.e. rounding-up division of (S + 1) by BLOCK_SIZE, and the same for (T +
  // 1).
  const int num_s_blocks = S / BLOCK_SIZE + 1;
  //, num_t_blocks = T / BLOCK_SIZE + 1;

  // num_blocks_this_iter is an upper bound on the number of blocks of dimension
  // (BLOCK_SIZE by BLOCK_SIZE) that might be active on this iteration (`iter`).
  // These iterations start from the bottom left of the image so that on iter ==
  // 0 we process only one block with block-index (0, 0) then on iter == 1 we
  // process block-indexes (1, 0) and (0, 1); and then on iter==2 we process (2,
  // 0), (1, 1) and (0, 2); and so on.  We also will never have more than
  // `num_s_blocks` blocks (We'll never have more than num_t_blocks either, but
  // the numbering we use corresponds to s and not t, so when we hit the
  // num_t_blocks limit, the blocks with the lowest s indexes would just not be
  // active and we'll 'continue' in the loop below).
  int num_blocks_this_iter = min(iter + 1, num_s_blocks);

  // For the block with s_block_begin == 0 and t_block_begin == 0 (for
  // easy illustration), px_buf[s][t] will contain px[s - 1][t + t_offset]; or
  // -infinity. for out-of-range indexes into px. Likewise, py_buf[s][t] will
  // contain (py[s][t - 1]).
  __shared__ scalar_t px_buf[BLOCK_SIZE][BLOCK_SIZE],
      py_buf[BLOCK_SIZE][BLOCK_SIZE];

  // p_buf[s][t] == p[s+s_block_begin-1][t+t_block_begin-1]
  // 1st row/col of p_buf correspond to the previously computed blocks (lower
  // `iter`), or to negative indexes into p.  So, for the origin block,
  // p_buf[s][t] corresponds to p[s - 1][t - 1]; or -inf for
  // out-of-range values.
  __shared__ scalar_t p_buf[BLOCK_SIZE + 1][BLOCK_SIZE + 1];

  // boundary_buf will be used to store the b'th row of `boundary` if we have
  // boundary information supplied; or (0, 0, S, T) otherwise.
  __shared__ int64_t boundary_buf[4];

  if (threadIdx.x == 0) {
    boundary_buf[0] = 0;
    boundary_buf[1] = 0;
    boundary_buf[2] = S;
    boundary_buf[3] = T;
  }

  // batch_block_iter iterates over batch elements (index b) and block
  // indexes in the range [0..num_blocks_this_iter-1], combining both
  // batch and block indexes.
  for (int batch_block_iter = blockIdx.x;
       batch_block_iter < B * num_blocks_this_iter;
       batch_block_iter += gridDim.x) {
    int block = batch_block_iter / B,
        b = batch_block_iter % B; // b is the index into the batch

    // Note: `block` can be no greater than `iter` because num_blocks_this_iter
    // <= iter + 1, i.e. iter >= num_blocks_this_iter - 1; and
    // block < num_blocks_this_iter, so iter - block >= 0.
    int s_block_begin = block * BLOCK_SIZE,
        t_block_begin = (iter - block) * BLOCK_SIZE;
    bool is_origin_block = (s_block_begin + t_block_begin == 0);

    __syncthreads();

    if (threadIdx.x < 4)
      boundary_buf[threadIdx.x] = boundary(b, threadIdx.x);

    __syncthreads();

    int s_begin = boundary_buf[0], t_begin = boundary_buf[1],
        s_end = boundary_buf[2], t_end = boundary_buf[3];

    s_block_begin += s_begin;
    t_block_begin += t_begin;

    // block_S and block_T are the actual sizes of this block (the block of `p`
    // that we will write), no greater than (BLOCK_SIZE, BLOCK_SIZE) but
    // possibly less than that if we are towards the end of the sequence.  The
    // last element in the output matrix p that we need to write is (s_end,
    // t_end), i.e. the one-past-the-end index is (s_end + 1, t_end + 1).
    int block_S = min(BLOCK_SIZE, s_end + 1 - s_block_begin),
        block_T = min(BLOCK_SIZE, t_end + 1 - t_block_begin);

    if (block_S <= 0 || block_T <= 0)
      continue;

    // Load px_buf and py_buf.
    for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int s_in_block = i / BLOCK_SIZE, t_in_block = i % BLOCK_SIZE,
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin,
          t_off = t + t_offset;
      // comparing as unsigned int makes sure the index is nonnegative.
      // Caution: if s_begin > 0 or t_begin > 0 we may end up loading some px
      // and py values that are outside the proper boundaries that we need, but
      // the corresponding p_buf values will end up being 0 so this won't
      // matter.
      scalar_t this_px = -INFINITY;
      // Below, "&& t <= t_end" can be interpreted as:
      //  "&& (modified ? t_off < t_end : t_off <= t_end)
      // [since px's last valid index is t_end - 1 if modified, else t_end.
      if (s > s_begin && s <= s_end && t_off >= t_begin && t <= t_end)
        this_px = px(b, s - 1, t_off);

      px_buf[s_in_block][t_in_block] = this_px;

      scalar_t this_py = -INFINITY;
      if (t > t_begin && t <= t_end && s <= s_end)
        this_py = py(b, s, t - 1);
      py_buf[s_in_block][t_in_block] = this_py;
    }

    // Load the 1st row and 1st column of p_buf.
    // This is the context from previously computed blocks of the
    // image.  Remember: p_buf[s][t] will correspond to p[s + s_block_begin -
    // 1][t + t_block_begin - 1]
    if (threadIdx.x <= BLOCK_SIZE) {
      // s_in_p_buf and t_in_pbuf are simply the indexes into p_buf
      int s_in_p_buf = threadIdx.x, t_in_p_buf = 0,
          s = s_in_p_buf + s_block_begin - 1,
          t = t_in_p_buf + t_block_begin - 1;

      scalar_t this_p = -INFINITY;
      if (s >= s_begin && s <= s_end && t >= t_begin && t <= t_end)
        this_p = p(b, s, t);
      p_buf[s_in_p_buf][t_in_p_buf] = this_p;
    } else if (static_cast<unsigned int>(static_cast<int>(threadIdx.x) - 64) <=
               static_cast<unsigned int>(BLOCK_SIZE)) {
      // Another warp handles the other leg.  Checking as unsigned
      // tests that threadIdx.x - 64 is both >= 0 and <= BLOCK_SIZE
      int s_in_p_buf = 0, t_in_p_buf = static_cast<int>(threadIdx.x) - 64,
          s = s_in_p_buf + s_block_begin - 1,
          t = t_in_p_buf + t_block_begin - 1;

      scalar_t this_p = -INFINITY;
      if (s >= s_begin && s <= s_end && t >= t_begin && t <= t_end)
        this_p = p(b, s, t);
      p_buf[s_in_p_buf][t_in_p_buf] = this_p;
    }

    __syncthreads();

    // from here to the next __syncthreads(), only the 1st warp should be active
    // so we shouldn't need to synchronize.  (implicit within-warp
    // synchronization).

    if (threadIdx.x == 0) {
      // This if-statement is an optimization and modification of the loop below
      // for the value i == 0, i.e. inner-iteration == 0.  The modification is
      // to set p_buf to 1.0 = exp(0.0) if this is the "origin block",
      // i.e. s == s_begin, t == t_begin.  This corresponds to the
      // probability of the pair of sequences of length (0, 0).
      p_buf[1][1] =
          (is_origin_block ? 0.0
                           : LogAdd(
                                 // px_buf has t_offset applied.
                                 p_buf[0][1 + t_offset] + px_buf[0][0],
                                 p_buf[1][0] + py_buf[0][0]));
    }

    int s = threadIdx.x;
    for (int i = 1; i < block_S + block_T - 1; ++i) {
      __syncwarp();
      // i is the inner iteration, which corresponds to the (s + t) indexes of
      // the elements within the block that we write.  So i == 0 writes
      // positions (s, t) == (0, 0) (but we treated i == 0 as a special case
      // above); i == 1 writes (0, 1) and (1, 0); i == 2 writes (0, 2), (1, 1)
      // and (2, 1); and so on.  Note: not many threads participate in this
      // part, only up to BLOCK_SIZE at most.  Unfortunately we couldn't figure
      // out a very meaningful way for more threads to do work, that looked like
      // it would really speed things up.
      // So this kernel does (2 * BLOCK_SIZE) iterations, which may seem a lot,
      // but we do at least do the I/O in an efficient way and keep the
      // inner loop simple and fast (e.g. no exp() or log()).
      int t = i - s;
      if (s < block_S &&
          static_cast<unsigned int>(t) < static_cast<unsigned int>(block_T)) {
        // p_buf is indexed by s + 1 and t + 1 because it has an extra initial
        // row and column for context from previous blocks.  Taking into account
        // the way these buffers relate to the tensors p, px and py,
        // can be interpreted as follows,
        // writing sbb for s_block_begin and tbb for t_block_begin:
        //
        //   p[b][s+sbb][t+tbb] = LogAdd(p[b][s+sbb-1][t+tbb] +
        //   px[s+sbb-1][t+tbb],
        //                               p[b][s+sbb][t+tbb-1] +
        //                               py[s+sbb][t+tbb-1]
        //
        // where you can see that apart from the offsets of tbb and sbb, this is
        // the same as the recursion defined for p in
        // mutual_information.py:mutual_information_recursion(); and (eq. 0)
        // above.

        // note: px_buf has t_offset applied..
        p_buf[s + 1][t + 1] = LogAdd(p_buf[s][t + 1 + t_offset] + px_buf[s][t],
                                     p_buf[s + 1][t] + py_buf[s][t]);
        // We don't need to do __syncthreads() in this loop because all the
        // threads that are active are in the same warp.  (However, in future,
        // if NVidia changes some things, we might need to sync here).
      }
    }
    __syncthreads();

    // Write out the data to p;
    for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int s_in_block = i / BLOCK_SIZE, t_in_block = i % BLOCK_SIZE,
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      if (s_in_block < block_S && t_in_block < block_T) {
        scalar_t this_p = p_buf[s_in_block + 1][t_in_block + 1];
        p(b, s, t) = this_p;
      }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      // Write `ans`, if this is the final (top-right) block in its sequence
      // Logically, the following equation corresponds to:
      //   ans(b) = p[b][s_end][t_end]
      if (s_block_begin + block_S - 1 == s_end &&
          t_block_begin + block_T - 1 == t_end) {
        // you could read block_S below as block_S - 1 + 1, meaning,
        // it's the last index in a block of dimension block_S, but the indexes into
        // p_buf have a "+ 1".  Likewise for block_T.
        ans(b) = p_buf[block_S][block_T];
      }
    }
  }
}

// like exp(), but returns 0 if arg is inf/nan, or if result would be
// infinity or nan (note: this can happen for out-of-range elements
// when setting px_buf and py_buf is block_S != BLOCK_SIZE or
// block_T != BLOCK_SIZE, and it's a problem because even though
// out-of-range gradients are zero, if we multiply them by infinity
// we get NaN.
template <typename Real> __forceinline__ __device__ Real safe_exp(Real x) {
  if (x - x != 0)
    return 0;
  else {
    Real ans = exp(x);
    if (ans - ans != 0.0)
      return 0;
    return ans;
  }
}

/*
  Backward of mutual_information.

  The forward pass is:

               p[b,s,t] = log_add(p[b,s-1,t+t_offset] + px[b,s-1,t+t_offset],
                                  p[b,s,t-1] + py[b,s,t-1])          (eq. 0)

  where t_offset = (modified ? -1 : 0)

  The backprop for the above, implemented in the obvious way, would be as
  follows (note, we define term1 and term2 with offsets in the indexes, which
  will be convenient later..):

     term1(b,s-1,t+t_offset) =
       exp(p[b,s-1,t+t_offset] + px[b,s-1,t+t_offset] - p[b,s,t])       (0a)
     term2(b,s,t-1) = exp(p[b,s,t-1] + py[b,s,t-1] - p[b,s,t])          (0b)

  p_grad[b,s-1,t+t_offset] += p_grad[b,s,t] * term1(b,s-1,t+t_offset)      (1a)
 px_grad[b,s-1,t+t_offset] += p_grad[b,s,t] * term1(b,s-1,t+t_offset)      (1b)
           p_grad[b,s,t-1] += p_grad[b,s,t] * term2(b,s,t-1)       (1c)
          py_grad[b,s,t-1] += p_grad[b,s,t] * term2(b,s,t-1)       (1d)

  Adding 1 and -t_offset to the s and t indexes of (1a) an (1b), and
  1 to the t index of (1c) and (1d), the equations become:

      p_grad[b,s,t] += p_grad[b,s+1,t-t_offset] * term1(b,s,t)      (2a)
     px_grad[b,s,t] += p_grad[b,s+1,t-t_offset] * term1(b,s,t)       (2b)
      p_grad[b,s,t] += p_grad[b,s,t+1] * term2(b,s,t)       (2c)
     py_grad[b,s,t] += p_grad[b,s,t+1] * term2(b,s,t)       (2d)

   .. and replacing "+=" with "=", we can write:

      p_grad[b,s,t]  = p_grad[b,s+1,t-t_offset] * term1(b,s,t)  +    (3a)
                       p_grad[b,s,t+1] * term2(b,s,t)
      px_grad[b,s,t] = p_grad[b,s+1,t-t_offset] * term1(b,s,t)       (3b)
      py_grad[b,s,t] = p_grad[b,s,t+1] * term2(b,s,t)                (3c)

   Writing the definitions of term1 and term2 in a more convenient way:
     term1(b,s,t) = exp(p[b,s,t] + px[b,s,t] - p[b,s+1,t-t_offset])     (4a)
     term2(b,s,t) = exp(p[b,s,t] + py[b,s,t] - p[b,s,t+1])                 (4b)

  The backward pass will be slightly different from the forward pass in terms of
  how we store and index p (and p_grad), because for writing a particular block
  of p_grad, we need context on the top and right instead of the bottom and
  left.  So there are offsets of 1.
 */

// do not use call by reference here! use call by value to copy parameters from host to device.
template <typename scalar_t, int BLOCK_SIZE>
__global__ void mutual_information_backward_kernel(
    typename tf::TTypes<scalar_t, 3>::ConstTensor
        px, // B, S, T + 1 if !modified; B, S, T if modified.
    typename tf::TTypes<scalar_t, 3>::ConstTensor py, // B, S + 1, T.
    // B, S + 1, T + 1.  Produced in forward pass.
    typename tf::TTypes<scalar_t, 3>::Tensor p,
    // [B].  This is an input.
    typename tf::TTypes<scalar_t>::Vec ans_grad,
    typename tf::TTypes<scalar_t, 3>::Tensor
        p_grad, // B, S + 1, T + 1 if !modified; B, S, T if modified.
    typename tf::TTypes<scalar_t, 3>::Tensor px_grad, // B, S, T + 1.
    typename tf::TTypes<scalar_t, 3>::Tensor py_grad, // B, S + 1, T.
    // B, 4;  or 0, 0 if boundaries are the defaults (0, 0, S, T)
    typename tf::TTypes<int64_t>::ConstMatrix boundary,
    int iter, // This kernel is sequentially called with 'iter' = num_iters
              // - 1, num_iters - 2, .. 0, where num_iters can be taken to
              // be any sufficiently large number but will actually be:
              // num_s_blocks + num_t_blocks - 1 where num_s_blocks = S /
              // BLOCK_SIZE + 1 and num_t_blocks = T / BLOCK_SIZE + 1
    bool overwrite_ans_grad) { // If overwrite_ans_grad == true, this function
                               // will overwrite ans_grad with a value which,
                               // if everything is working correctly, should be
                               // identical or very close to the value of
                               // ans_grad that was passed in.
  const int B = px.dimension(0), S = px.dimension(1), T = py.dimension(2);
  const bool modified = (px.dimension(2) == T);
  const int neg_t_offset = (modified ? 1 : 0);

  // For statements that are the same as the forward pass, we are omitting some
  // comments.  We'll focus, in the comments, on differences from the forward
  // pass.
  const int num_s_blocks = S / BLOCK_SIZE + 1,
            // num_t_blocks = T / BLOCK_SIZE + 1,
      num_blocks_this_iter = min(iter + 1, num_s_blocks);

  // px_buf and py_buf are used temporarily to store the px and py values,
  // but then modified to store the "xderiv" and "yderiv" values defined
  // in (eq. 5) and (eq. 6) above.  For out-of-range values, we'll write 0.0
  // here.
  //  Initially (before xderiv/yderiv are written):
  //   px_buf[s][t] contains px[s+s_block_begin][t+t_block_begin];
  //   py_buf[s][t] contains py[s+s_block_begin][t+t_block_begin].
  // Later (see eq. 4 and eq. 5):
  //  px_buf[s][t] contains term1(b,ss,tt) ==
  //    exp(p[b][ss][tt] + px[b][ss][tt] - p[b][ss + 1][tt-t_offset]),
  //  py_buf[s][t] contains term2(b,ss,tt) ==

  // where ss == s + s_block_begin, tt = t + t_block_begin.
  // Unlike in the forward code, there is no offset of 1 in the indexes.
  __shared__ scalar_t px_buf[BLOCK_SIZE][BLOCK_SIZE],
      py_buf[BLOCK_SIZE][BLOCK_SIZE];

  // p_buf is initially used to store p, and then (after we are done putting
  // term1 and term2 into px_buf and py_buf) it is repurposed to store
  // p_grad.
  //
  // Unlike in the forward pass, p_buf has the same numbering as px_buf and
  // py_buf, it's not offset by 1: e.g., for the origin block, p_buf[0][0]
  // refers to p[0][0] and not p[-1][-1].  The p_buf block is larger by 1 than
  // the block for px_buf and py_buf; unlike in the forward pass, we store
  // context on the top and right, not the bottom and left, i.e. the elements at
  // (one past the largest indexes in the block).
  //
  // For out-of-range elements of p_buf, we'll put zero.
  __shared__ scalar_t p_buf[BLOCK_SIZE + 1][BLOCK_SIZE + 1];

  // boundary_buf will be used to store the b'th row of `boundary` if we have
  // boundary information supplied; or (0, 0, S, T) if not.
  __shared__ int64_t boundary_buf[4];

  if (threadIdx.x == 0) {
    boundary_buf[0] = 0;
    boundary_buf[1] = 0;
    boundary_buf[2] = S;
    boundary_buf[3] = T;
  }

  // batch_block_iter iterates over both batch elements (index b), and block
  // indexes in the range [0..num_blocks_this_iter-1].  The order here
  // doesn't matter, since there are no interdependencies between these
  // blocks (they are on a diagonal).
  for (int batch_block_iter = blockIdx.x;
       batch_block_iter < B * num_blocks_this_iter;
       batch_block_iter += gridDim.x) {
    int block = batch_block_iter / B, b = batch_block_iter % B;
    int s_block_begin = block * BLOCK_SIZE,
        t_block_begin = (iter - block) * BLOCK_SIZE;

    if (threadIdx.x < 4)
      boundary_buf[threadIdx.x] = boundary(b, threadIdx.x);
    __syncthreads();

    int s_begin = boundary_buf[0], t_begin = boundary_buf[1],
        s_end = boundary_buf[2], t_end = boundary_buf[3];
    s_block_begin += s_begin;
    t_block_begin += t_begin;

    // block_S and block_T are the actual sizes of this block, no greater than
    // (BLOCK_SIZE, BLOCK_SIZE) but possibly less than that if we are towards
    // the end of the sequence.
    // The last element of the output matrix p_grad we write is (s_end, t_end),
    // i.e. the one-past-the-end index of p_grad is (s_end + 1, t_end + 1).
    int block_S = min(BLOCK_SIZE, s_end + 1 - s_block_begin),
        block_T = min(BLOCK_SIZE, t_end + 1 - t_block_begin);

    if (block_S <= 0 || block_T <= 0)
      continue;

    // Load px_buf and py_buf.  At this point we just set them to the px and py
    // for this block.
    for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int s_in_block = i / BLOCK_SIZE, t_in_block = i % BLOCK_SIZE,
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      // We let px and py default to -infinity if they are out of range, which
      // will cause xderiv and yderiv for out-of-range values to be zero, and
      // cause correct behavior in edge cases (for the top and right blocks).
      // The issue is that p and p_grad are of larger dimension than px and py.
      scalar_t this_px = -INFINITY;
      if (s < s_end && t <= t_end)
        this_px = px(b, s, t);
      px_buf[s_in_block][t_in_block] = this_px;
      scalar_t this_py = -INFINITY;
      if (s <= s_end && t < t_end)
        this_py = py(b, s, t);
      py_buf[s_in_block][t_in_block] = this_py;
    }
    __syncthreads();

    // load p.
    for (int i = threadIdx.x; i < (BLOCK_SIZE + 1) * (BLOCK_SIZE + 1);
         i += blockDim.x) {
      int s_in_block = i / (BLOCK_SIZE + 1), t_in_block = i % (BLOCK_SIZE + 1),
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      // Setting 0.0 for out-of-bounds elements of p, together with setting
      // -INFINITY for out-of-bounds elements of px_buf and py_buf, will
      // ensure that we do the right thing in top and right edge cases,
      // i.e. that no derivatives will be propagated from out-of-bounds points
      // because the corresponding xderiv and yderiv values will be zero.
      scalar_t this_p = 0.0;
      if (s <= s_end && t <= t_end)
        this_p = p(b, s, t);
      // if this_p is -inf, replace with large finite negative value, to avoid
      // NaN's below.
      // TODO: use a value that would work correctly in half precision
      if (this_p < -1.0e+30)
        this_p = -1.0e+30;
      p_buf[s_in_block][t_in_block] = this_p;
    }
    __syncthreads();

    // Set term1 and term2; see equations (4a) and (4b) above.
    for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      // We can apply this formula to the entire block even if we are processing
      // a partial block; we have ensured that x_buf and y_buf contain
      // -infinity, and p contains 0, for out-of-range elements, so we'll get
      // x_buf and y_buf containing 0 after applying the following formulas.
      int s = i / BLOCK_SIZE, t = i % BLOCK_SIZE;
      // Mathematically the following is doing:
      //  term1(b,s,t) = exp(p[b,s,t] + px[b,s,t] - p[b,s+1,t-t_offset])   (4a)
      // (with an offset on the s and t indexes)
      // Use safe_exp() not exp(), as we could have (-inf) - (-inf) = nan, want
      // any finite number in this case as derivs would be zero.
      // Also want -inf->zero.
      px_buf[s][t] =
          safe_exp(p_buf[s][t] + px_buf[s][t] - p_buf[s + 1][t + neg_t_offset]);
      // Mathematically the following is doing:
      // term2(b,s,t) = exp(p[b,s,t] + py[b,s,t] - p[b,s,t+1])             (4b)
      // (with an offset on the s and t indexes)
      py_buf[s][t] = safe_exp(p_buf[s][t] + py_buf[s][t] - p_buf[s][t + 1]);
    }

    __syncthreads();

    // Load p_grad for the top and right elements in p_buf: i.e. for elements
    // p_buf[s][t] where s == block_S (exclusive-or) t == block_T.
    // These are the p_grad values computed by previous instances of this kernel
    // If this is one of the top or right blocks, some or all of the p_grad
    // values we'd be reading here will be out of range, and we use zeros
    // to ensure no gradient gets propagated from those positions.
    if (threadIdx.x <= block_S) {
      int s_in_block = threadIdx.x, t_in_block = block_T,
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      p_buf[s_in_block][t_in_block] =
          (s <= s_end && t <= t_end ? p_grad(b, s, t) : 0.0);
    } else if (static_cast<unsigned int>(static_cast<int>(threadIdx.x) - 64) <
               static_cast<unsigned int>(block_T)) {
      // casting to unsigned before the comparison tests for both negative and
      // out-of-range values of (int)threadIdx.x - 64.
      int s_in_block = block_S, t_in_block = static_cast<int>(threadIdx.x) - 64,
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      p_buf[s_in_block][t_in_block] =
          (s <= s_end && t <= t_end ? p_grad(b, s, t) : 0.0);
    }

    __syncthreads();

    //  The highest-numbered value in p_buf that we need (corresponding,
    // of course, to p_grad), is:
    //    p_buf[block_S - 1][block_T - 1],
    // and the inner iteration number (i) on which we set this is the sum of
    // these indexes, i.e.  (block_S - 1) + (block_T - 1).
    bool is_final_block = (s_block_begin + block_S == s_end + 1 &&
                           t_block_begin + block_T == t_end + 1);

    int first_iter = block_S + block_T - 2;
    if (is_final_block) {
      // The following statement corresponds to:
      // p_grad[b][s_end][t_end] = ans_grad(b)
      // Normally this element of p_buf would be set by the first iteration of
      // the loop below, so if it's set this way we have to decrement first_iter
      // to prevent it from being overwritten.
      p_buf[block_S - 1][block_T - 1] = ans_grad(b);
      --first_iter;
    }

    {
      int s = threadIdx.x;
      for (int i = first_iter; i >= 0; --i) {
        __syncwarp();
        int t = i - s;
        if (s < block_S &&
            static_cast<unsigned int>(t) < static_cast<unsigned int>(block_T)) {
          // The following statement is really operating on the gradients;
          // it corresponds, with offsets of s_block_begin and t_block_begin
          // on the indexes, to equation (3a) above, i.e.:
          //   p_grad[b,s,t]  =
          //      p_grad[b,s+1,t-t_offset] * term1(b,s,t)  +             (3a)
          //      p_grad[b,s,t+1] * term2(b,s,t)
          p_buf[s][t] = (p_buf[s + 1][t + neg_t_offset] * px_buf[s][t] +
                         p_buf[s][t + 1] * py_buf[s][t]);
        }
      }
    }

    __syncthreads();

    // Write out p_grad, px_grad and py_grad.
    for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int s_in_block = i / BLOCK_SIZE, t_in_block = i % BLOCK_SIZE,
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      // s_end and t_end are the one-past-the-end of the (x,y) sequences, but
      // the one-past-the-end element of p_grad would be (s_end + 1, t_end + 1).
      if (t <= t_end && s <= s_end) {
        p_grad(b, s, t) = p_buf[s_in_block][t_in_block];

        if (s < s_end && t <= t_end - neg_t_offset) {
          // write px_grad, which is of shape [B][S][T + 1] if !modified,
          // [B][S][T] if modified.  the condition "t <= t_end - neg_t_offset"
          // becomes "t <= t_end" if !modified, and "t <= t_end - 1" if
          // modified, keeping us within the bounds of px_grad.

          // From (eq. 3b):
          // px_grad[b,s,t] = p_grad[b,s+1,t-t_offset] * term1(b,s,t)
          px_grad(b, s, t) = (p_buf[s_in_block + 1][t_in_block + neg_t_offset] *
                              px_buf[s_in_block][t_in_block]);
        }
        if (t < t_end) { // write py_grad, which is of shape [B][S + 1][T]
          // from (eq. 3c):
          // py_grad[b,s,t] = p_grad[b,s,t+1] * term2(b,s,t)
          py_grad(b, s, t) = (p_buf[s_in_block][t_in_block + 1] *
                              py_buf[s_in_block][t_in_block]);
        }
      }
    }

    if (threadIdx.x == 0 && s_block_begin == s_begin &&
        t_block_begin == t_begin && overwrite_ans_grad)
      ans_grad(b) = p_buf[0][0];
  }
}

// forward of mutual_information.  See """... """ comment of
// `mutual_information` in mutual_information.py for documentation of the
// behavior of this function.
template <typename scalar_t>
int MutualInformationCuda(
    typename tf::TTypes<scalar_t, 3>::ConstTensor& px, // [B][S][T+1] if !modified, [B][S][T] if modified.
    typename tf::TTypes<scalar_t, 3>::ConstTensor& py, // [B][S+1][T]
    typename tf::TTypes<int64_t>::ConstMatrix& boundary, // [B][4], int.
    typename tf::TTypes<scalar_t, 3>::Tensor& p, 
    typename tf::TTypes<scalar_t>::Vec& ans,           //  [B][S+1][T+1]; an output
    cudaStream_t stream) {                        
  Assert(px.NumDimensions == 3, "px must be 3-dimensional");
  Assert(py.NumDimensions == 3, "py must be 3-dimensional.");
  Assert(p.NumDimensions == 3, "p must be 3-dimensional.");
//  Assert(px.device().is_cuda() && py.device().is_cuda() &&
//                  p.device().is_cuda(),
//              "inputs must be CUDA tensors");

  const int B = px.dimension(0), S = px.dimension(1), T = py.dimension(2);
  Assert(px.dimension(2) == T || px.dimension(2) == T + 1);
  Assert(py.dimension(0) == B && py.dimension(1) == S + 1 && py.dimension(2) == T);
  Assert(p.dimension(0) == B && p.dimension(1) == S + 1 && p.dimension(2) == T + 1);

  Assert(boundary.dimension(0) == B && boundary.dimension(1) == 4);
//  Assert(boundary.device().is_cuda() && boundary.dtype() == torch::kInt64);

  // num_threads and num_blocks and BLOCK_SIZE can be tuned.
  // (however, num_threads may not be less than 128).
  const int num_threads = 128, num_blocks = 256, BLOCK_SIZE = 32;

  // The blocks cover the 'p' matrix, which is of dimension (B, S+1, T+1),
  // so dividing by BLOCK_SIZE rounding up we get e.g.
  // (S+1 + BLOCK_SIZE-1) / BLOCK_SIZE == S / BLOCK_SIZE + 1
  const int num_s_blocks = S / BLOCK_SIZE + 1,
            num_t_blocks = T / BLOCK_SIZE + 1,
            num_iters = num_s_blocks + num_t_blocks - 1;

  for (int iter = 0; iter < num_iters; ++iter) {
    mutual_information_kernel<scalar_t, BLOCK_SIZE>
        <<<num_blocks, num_threads, 0, stream>>>(
            px,
            py,
            p,
            boundary,
            ans,
            iter);
  }

  return 1;
}

// backward of mutual_information; returns (grad_px, grad_py)
// If overwrite_ans_grad == true, will overwrite ans_grad with a value which
// should be identical to the original ans_grad if the computation worked
// as it should.
template <typename scalar_t>
int MutualInformationBackwardCuda(
    typename tf::TTypes<scalar_t, 3>::ConstTensor& px, 
    typename tf::TTypes<scalar_t, 3>::ConstTensor& py, 
    typename tf::TTypes<int64_t>::ConstMatrix& boundary,
    typename tf::TTypes<scalar_t, 3>::Tensor& p, 
    typename tf::TTypes<scalar_t, 3>::Tensor& p_grad, 
    typename tf::TTypes<scalar_t, 3>::Tensor& px_grad, 
    typename tf::TTypes<scalar_t, 3>::Tensor& py_grad, 
    typename tf::TTypes<scalar_t>::Vec& ans_grad, 
    bool overwrite_ans_grad,
    cudaStream_t stream) {
  Assert(px.NumDimensions == 3, "px must be 3-dimensional");
  Assert(py.NumDimensions == 3, "py must be 3-dimensional.");
  Assert(p.NumDimensions == 3, "p must be 3-dimensional.");
  Assert(ans_grad.NumDimensions == 1, "ans_grad must be 1-dimensional.");

//  Assert(px.device().is_cuda() && py.device().is_cuda() &&
//              p.device().is_cuda() && ans_grad.device().is_cuda() &&
//              "inputs must be CUDA tensors");
  
  const int B = px.dimension(0), S = px.dimension(1), T = py.dimension(2);

  Assert(px.dimension(2) == T ||
              px.dimension(2) == T + 1); // modified case ||  not-modified case
  Assert(py.dimension(0) == B && py.dimension(1) == S + 1);
  Assert(p.dimension(0) == B && p.dimension(1) == S + 1 && p.dimension(2) == T + 1);

  Assert(boundary.dimension(0) == B && boundary.dimension(1) == 4);
//  Assert(boundary.device().is_cuda() && boundary.dtype() == torch::kInt64);
  Assert(ans_grad.dimension(0) == B);

  // num_threads and num_blocks and BLOCK_SIZE can be tuned.
  // (however, num_threads may not be less than 128).
  const int num_threads = 128, num_blocks = 256, BLOCK_SIZE = 32;

  // The blocks cover the 'p' matrix, which is of dimension (B, S+1, T+1),
  // so dividing by BLOCK_SIZE rounding up we get e.g.
  // (S+1 + BLOCK_SIZE-1) / BLOCK_SIZE == S / BLOCK_SIZE + 1
  const int num_s_blocks = S / BLOCK_SIZE + 1,
            num_t_blocks = T / BLOCK_SIZE + 1,
            num_iters = num_s_blocks + num_t_blocks - 1;

  for (int iter = num_iters - 1; iter >= 0; --iter) {
    mutual_information_backward_kernel<scalar_t, BLOCK_SIZE>
        <<<num_blocks, num_threads, 0, stream>>>(
            px,
            py,
            p,
            ans_grad,
            p_grad,
            px_grad,
            py_grad,
            boundary, iter,
            overwrite_ans_grad);
  }
  return 1;
}

template<typename scalar_t, typename idx_t, typename BinaryOperation>
__device__ void binary_op_update(const scalar_t lhs, scalar_t& rhs, const idx_t lhs_idx, idx_t& rhs_idx, BinaryOperation binary_op) {
  if(rhs > lhs) {
    rhs = lhs;
    rhs_idx = lhs_idx;
  }
}

// borrow pytorch's implementation
/* Perform an inclusive scan along the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<typename scalar_t, int num_threads_x, int num_threads_y, class BinaryFunction>
__global__ void tensor_kernel_scan_innermost_dim_with_indices(const scalar_t *self_, scalar_t *values_,
                                                int num_rows, int row_size,
                                                scalar_t init, BinaryFunction binary_op) {
  __shared__ scalar_t vbuf[num_threads_y][2 * num_threads_x];
  __shared__ int64_t ibuf[num_threads_y][2 * num_threads_x];
  scalar_t* row_buf = vbuf[threadIdx.y];
  int64_t* row_idx_buf = ibuf[threadIdx.y];

  for (int block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    int row = block_row + threadIdx.y;
    const scalar_t *row_self = self_ + row * row_size;
    scalar_t *row_values = values_ + row * row_size;
    scalar_t block_total = init;
    int64_t block_idx_final = 0;
    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (int block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      int col1 = block_col + threadIdx.x;
      int col2 = block_col + num_threads_x + threadIdx.x;
      if (row < num_rows) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = *reinterpret_cast<const scalar_t*>(&row_self[col1]);
          row_idx_buf[threadIdx.x] = col1;
        } else {
          row_buf[threadIdx.x] = init;
          // No need to set the index here as the value in init will never be selected
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = *reinterpret_cast<const scalar_t*>(&row_self[col2]);
          row_idx_buf[num_threads_x + threadIdx.x] = col2;
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
          // No need to set the index here as the value in init will never be selected
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (threadIdx.x == 0) {
          binary_op_update(block_total, row_buf[0], block_idx_final, row_idx_buf[0], binary_op);
        }
      }
      __syncthreads();

      // Parallel reduction (up-sweep).
      for (int s = num_threads_x, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < num_rows && threadIdx.x < s) {
          int offset = (2 * threadIdx.x + 1) * d - 1;
          binary_op_update(row_buf[offset], row_buf[offset + d], row_idx_buf[offset], row_idx_buf[offset + d], binary_op);
        }
        __syncthreads();
      }

      // Down-sweep.
      for (int s = 2, d = num_threads_x / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && threadIdx.x < s - 1) {
          int offset = 2 * (threadIdx.x + 1) * d - 1;
          binary_op_update(row_buf[offset], row_buf[offset + d], row_idx_buf[offset], row_idx_buf[offset + d], binary_op);
        }
        __syncthreads();
      }

      // Write back to output.
      if (row < num_rows) {
        if (col1 < row_size){
          row_values[col1] = row_buf[threadIdx.x];
        }
        if (col2 < row_size) {
          row_values[col2] = row_buf[num_threads_x + threadIdx.x];
        }
      }
      block_total = row_buf[2 * num_threads_x - 1];
      block_idx_final = row_idx_buf[2 * num_threads_x - 1];
      __syncthreads();
    }
  }
}

// borrow pytorch's implementation
/**
   Computes ceil(a / b)
*/
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
__forceinline__ FT_CUDA_HOSTDEV T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename scalar_t, class BinaryFunction>
__host__ void scan_innermost_dim_with_indices(
    typename tf::TTypes<scalar_t>::ConstMatrix& in,
    typename tf::TTypes<scalar_t>::Matrix& out,
    BinaryFunction binary_op,
    cudaStream_t stream) {
  int ndim = in.NumDimensions;
  // Treat all outer dimensions as a single dimension.
  int num_rows = in.dimension(0);
  int row_size = in.dimension(1);

  dim3 threads(16, 32);
  dim3 grid(ceil_div(num_rows, int(threads.y)));

  tensor_kernel_scan_innermost_dim_with_indices<scalar_t, 16, 32><<<grid, threads, 0, stream>>>(
    in.data(), out.data(),
    num_rows, row_size, std::numeric_limits<scalar_t>::max(), binary_op);
}

template <typename scalar_t>
int CumminCuda(
    typename tf::TTypes<scalar_t>::ConstMatrix& in,
    typename tf::TTypes<scalar_t>::Matrix& out,
    cudaStream_t stream) {
  scan_innermost_dim_with_indices<scalar_t>(in, out, std::less_equal<scalar_t>(), stream);

  return 1;
}
} // namespace tf_fast_rnnt

template int tf_fast_rnnt::MutualInformationCuda<float>(
    typename tf::TTypes<float, 3>::ConstTensor& px, // [B][S][T+1] if !modified, [B][S][T] if modified.
    typename tf::TTypes<float, 3>::ConstTensor& py, // [B][S+1][T]
    typename tf::TTypes<int64_t>::ConstMatrix& boundary, // [B][4], int.
    typename tf::TTypes<float, 3>::Tensor& p, 
    typename tf::TTypes<float>::Vec& ans,
    cudaStream_t stream);                        //  [B][S+1][T+1]; an output
template int tf_fast_rnnt::MutualInformationBackwardCuda<float>(
    typename tf::TTypes<float, 3>::ConstTensor& px, 
    typename tf::TTypes<float, 3>::ConstTensor& py, 
    typename tf::TTypes<int64_t>::ConstMatrix& boundary,
    typename tf::TTypes<float, 3>::Tensor& p, 
    typename tf::TTypes<float, 3>::Tensor& p_grad, 
    typename tf::TTypes<float, 3>::Tensor& px_grad, 
    typename tf::TTypes<float, 3>::Tensor& py_grad, 
    typename tf::TTypes<float>::Vec& ans_grad, 
    bool overwrite_ans_grad,
    cudaStream_t stream);
template int tf_fast_rnnt::CumminCuda<int64_t>(
    typename tf::TTypes<int64_t>::ConstMatrix& in,
    typename tf::TTypes<int64_t>::Matrix& out,
    cudaStream_t stream);
