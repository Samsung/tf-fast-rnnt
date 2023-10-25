#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corporation   (authors: Daniel Povey,
#                                                     Wei Kang)
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

# To run this single test, use
#
#  ctest --verbose -R rnnt_loss_test_py

import tensorflow as tf

import unittest
import tf_fast_rnnt
import random
import numpy as np
np.set_printoptions(precision=8, suppress=True, floatmode='fixed')

class TestRnntLoss(unittest.TestCase):
    @classmethod
    #@tf.function
    def test_rnnt_loss_pruned_simple(self):
        B = 2
        T = 10
        S = 7
        C = 4

        num_gpu = len(tf.config.list_physical_devices('GPU'))

        if num_gpu == 0:
            print("No GPU available")
            exit()
        else:
            print("Num GPUs Available: ", num_gpu)

        with tf.device('/GPU:0'):
            np.random.seed(1234)
            frames = np.random.randint(S, T, (B,))
            seq_length = np.random.randint(3, S - 1, (B,))

            T = np.amax(frames)
            S = np.amax(seq_length)

            am = np.random.randn(B, T, C).astype('f')
            lm = np.random.randn(B, S + 1, C).astype('f')
            symbols = np.random.randint(0, C - 1, (B, S)).astype(np.int64)
            terminal_symbol = C - 1

            boundary = np.zeros((B, 4))
            boundary[:, 2] = seq_length
            boundary[:, 3] = frames
            boundary = tf.convert_to_tensor(boundary, dtype=tf.int64)

            for rnnt_type in ["regular"]:
                # normal rnnt
                logits = tf.expand_dims(am, 2) + tf.expand_dims(lm, 1)

                # nonlinear transform
                logits = tf.sigmoid(logits)

                with tf.GradientTape() as tape:
                    tape.watch(logits)
                    rnnt_loss = tf_fast_rnnt.rnnt_loss(
                        logits=logits,
                        symbols=symbols,
                        termination_symbol=terminal_symbol,
                        boundary=boundary,
                        rnnt_type=rnnt_type,
                        reduction="mean",
                        delay_penalty=0.2,
                        return_grad=True
                    )

                grad = tape.gradient(rnnt_loss, logits)

                print('rnnt_loss', rnnt_loss)
#                print(f"Unpruned rnnt loss grad with {rnnt_type} rnnt : {grad}")

                # tf pruning
                simple_loss, (px_grad, py_grad) = tf_fast_rnnt.rnnt_loss_simple(
                    lm=lm,
                    am=am,
                    symbols=symbols,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                    return_grad=True,
                    reduction="none",
                    delay_penalty=0.2,
                )

#                for r in range(2, 50, 5):
                for r in range(3, 5, 5):
                    ranges = tf_fast_rnnt.get_rnnt_prune_ranges(
                        px_grad=px_grad,
                        py_grad=py_grad,
                        boundary=boundary,
                        s_range=r,
                    )

                    # (B, T, r, C)
                    pruned_am, pruned_lm = tf_fast_rnnt.do_rnnt_pruning(
                        am=am, lm=lm, ranges=ranges
                    )

                    logits = pruned_am + pruned_lm

                    # nonlinear transform
                    logits = tf.sigmoid(logits)

                    with tf.GradientTape() as tape:
                        tape.watch(logits)
                        pruned_loss = tf_fast_rnnt.rnnt_loss_pruned(
                            logits=logits,
                            symbols=symbols,
                            ranges=ranges,
                            termination_symbol=terminal_symbol,
                            boundary=boundary,
                            rnnt_type=rnnt_type,
                            reduction="mean",
                            delay_penalty=0.2,
                            training=True
                        )

                    grad = tape.gradient(pruned_loss, logits)

                    print(f"Pruning loss with range {r} : {pruned_loss}")
#                    print(f"Pruned rnnt loss grad with {rnnt_type} rnnt : {grad}")

    @classmethod
    #@tf.function
    def test_rnnt_loss_pruned_stress(self):
        B = 2
        T = 200
        S = 50
        C = 50

        num_gpu = len(tf.config.list_physical_devices('GPU'))

        if num_gpu == 0:
            print("No GPU available")
            exit()
        else:
            print("Num GPUs Available: ", num_gpu)

        with tf.device('/GPU:0'):
            np.random.seed(12345)
            frames = np.random.randint(S, T, (B,))
            seq_length = np.random.randint(3, S - 1, (B,))

            T = np.amax(frames)
            S = np.amax(seq_length)

            am = np.random.randn(B, T, C).astype('f')
            lm = np.random.randn(B, S + 1, C).astype('f')
            symbols = np.random.randint(0, C - 1, (B, S)).astype(np.int64)
            terminal_symbol = C - 1

            boundary = np.zeros((B, 4))
            boundary[:, 2] = seq_length
            boundary[:, 3] = frames
            boundary = tf.convert_to_tensor(boundary, dtype=tf.int64)

            for rnnt_type in ["regular"]:
                # normal rnnt
                logits = tf.expand_dims(am, 2) + tf.expand_dims(lm, 1)

                # nonlinear transform
                logits = tf.sigmoid(logits)

                with tf.GradientTape() as tape:
                    tape.watch(logits)
                    rnnt_loss = tf_fast_rnnt.rnnt_loss(
                        logits=logits,
                        symbols=symbols,
                        termination_symbol=terminal_symbol,
                        boundary=boundary,
                        rnnt_type=rnnt_type,
                        reduction="mean",
                        delay_penalty=0.2,
                        return_grad=True
                    )

                grad = tape.gradient(rnnt_loss, logits)

#                print('rnnt_loss', rnnt_loss)

                # tf pruning
                simple_loss, (px_grad, py_grad) = tf_fast_rnnt.rnnt_loss_simple(
                    lm=lm,
                    am=am,
                    symbols=symbols,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                    return_grad=True,
                    reduction="none",
                    delay_penalty=0.2,
                )

                grad_mean = 0

                for r in range(5, 51, 1):
                    ranges = tf_fast_rnnt.get_rnnt_prune_ranges(
                        px_grad=px_grad,
                        py_grad=py_grad,
                        boundary=boundary,
                        s_range=r,
                    )

                    # (B, T, r, C)
                    pruned_am, pruned_lm = tf_fast_rnnt.do_rnnt_pruning(
                        am=am, lm=lm, ranges=ranges
                    )

                    logits = pruned_am + pruned_lm

                    # nonlinear transform
                    logits = tf.sigmoid(logits)

                    with tf.GradientTape() as tape:
                        tape.watch(logits)
                        pruned_loss = tf_fast_rnnt.rnnt_loss_pruned(
                            logits=logits,
                            symbols=symbols,
                            ranges=ranges,
                            termination_symbol=terminal_symbol,
                            boundary=boundary,
                            rnnt_type=rnnt_type,
                            reduction="mean",
                            delay_penalty=0.2,
                            training=True
                        )

                    grad = tape.gradient(pruned_loss, logits)
                    grad_mean += tf.reduce_mean(grad, [1, 2])

                    print(f"Pruning loss with range {r} : {pruned_loss}")

                print(f"Pruned rnnt loss grad_mean with {rnnt_type} rnnt : {grad_mean}")

if __name__ == "__main__":
    unittest.main()
