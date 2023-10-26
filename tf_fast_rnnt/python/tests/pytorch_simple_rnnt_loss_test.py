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

import unittest

import fast_rnnt
import random
import torch

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(profile="full", sci_mode=False, precision=8)

class TestRnntLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and fast_rnnt.with_cuda():
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))
        try:
            import torchaudio
            import torchaudio.functional

            if hasattr(torchaudio.functional, "rnnt_loss"):
                cls.has_torch_rnnt_loss = True
            else:
                cls.has_torch_rnnt_loss = False
                print(
                    f"Current torchaudio version: {torchaudio.__version__}\n"
                    "Skipping the tests of comparing rnnt loss with torch "
                    "one, to enable these tests please install a "
                    "version >= 0.10.0"
                )
        except ImportError as e:
            cls.has_torch_rnnt_loss = False
            print(
                f"Import torchaudio error, error message: {e}\n"
                "Skipping the tests of comparing rnnt loss with torch "
                "one, to enable these tests, please install torchaudio "
                "with version >= 0.10.0"
            )

    def test_rnnt_loss_pruned_smoothed(self):
        B = 2
        T = 10
        S = 7
        C = 4

        np.random.seed(1234)

        frames = torch.from_numpy(np.random.randint(S, T, (B,)))
        seq_length = torch.from_numpy(np.random.randint(3, S - 1, (B,)))
#        frames = np.random.randint(S, S+1, (B,))
#        seq_length = np.random.randint(C, C+1, (B,))

        T = torch.max(frames)
        S = torch.max(seq_length)

        am_ = torch.from_numpy(np.random.randn(B, T, C).astype('f'))
        lm_ = torch.from_numpy(np.random.randn(B, S + 1, C).astype('f'))
        symbols_cpu = torch.from_numpy(np.random.randint(0, C - 1, (B, S)))
        terminal_symbol = C - 1

        boundary_ = torch.zeros((B, 4), dtype=torch.int64)
        boundary_[:, 2] = seq_length
        boundary_[:, 3] = frames

        for rnnt_type in ["regular"]:
            for device in self.devices[1:]:
                print('device', device)
                # normal rnnt
                am = am_.to(device)
                lm = lm_.to(device)
                symbols = symbols_cpu.to(device)
                boundary = boundary_.to(device)
                logits = am.unsqueeze(2) + lm.unsqueeze(1)
                logits = logits.float()

                # nonlinear transform
                logits = torch.sigmoid(logits)
                logits.requires_grad_()

                fast_loss = fast_rnnt.rnnt_loss(
                    logits=logits,
                    symbols=symbols,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                    reduction="mean",
                    delay_penalty=0.2,
                )

                fast_grad = torch.autograd.grad(fast_loss, logits)
                fast_grad = fast_grad[0]

                print(f"Unpruned rnnt loss with {rnnt_type} rnnt : {fast_loss}")
                print(f"Unpruned rnnt loss grad with {rnnt_type} rnnt : {fast_grad}")

                # pruning
                simple_loss, (px_grad, py_grad) = fast_rnnt.rnnt_loss_smoothed(
                    lm=lm,
                    am=am,
                    symbols=symbols,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    lm_only_scale=0.1,
                    am_only_scale=0.2,
                    rnnt_type=rnnt_type,
                    return_grad=True,
                    reduction="none",
                    delay_penalty=0.2,
                )

                print('simple_loss', simple_loss)
                print('px_grad', px_grad)
                print('py_grad', py_grad)

#                for r in range(2, 50, 5):
                for r in range(3, 5, 5):
                    ranges = fast_rnnt.get_rnnt_prune_ranges(
                        px_grad=px_grad,
                        py_grad=py_grad,
                        boundary=boundary,
                        s_range=r,
                    )

                    # (B, T, r, C)
                    pruned_am, pruned_lm = fast_rnnt.do_rnnt_pruning(
                        am=am, lm=lm, ranges=ranges
                    )

                    logits = pruned_am + pruned_lm

                    # nonlinear transform
                    logits = torch.sigmoid(logits)
                    logits.requires_grad_()

                    pruned_loss = fast_rnnt.rnnt_loss_pruned(
                        logits=logits,
                        symbols=symbols,
                        ranges=ranges,
                        termination_symbol=terminal_symbol,
                        boundary=boundary,
                        rnnt_type=rnnt_type,
                        reduction="mean",
                        delay_penalty=0.2,
                    )

                    fast_grad = torch.autograd.grad(pruned_loss, logits)
                    fast_grad = fast_grad[0]

                    print(f"Pruning loss with range {r} : {pruned_loss}")
                    print(f"Pruned rnnt loss grad with {rnnt_type} rnnt : {fast_grad}")

    @unittest.skip
    def test_rnnt_loss_pruned_simple(self):
        B = 2
        T = 10
        S = 7
        C = 4

        np.random.seed(1234)

        frames = torch.from_numpy(np.random.randint(S, T, (B,)))
        seq_length = torch.from_numpy(np.random.randint(3, S - 1, (B,)))
#        frames = np.random.randint(S, S+1, (B,))
#        seq_length = np.random.randint(C, C+1, (B,))

        T = torch.max(frames)
        S = torch.max(seq_length)

        am_ = torch.from_numpy(np.random.randn(B, T, C).astype('f'))
        lm_ = torch.from_numpy(np.random.randn(B, S + 1, C).astype('f'))
        symbols_cpu = torch.from_numpy(np.random.randint(0, C - 1, (B, S)))
        terminal_symbol = C - 1

        boundary_ = torch.zeros((B, 4), dtype=torch.int64)
        boundary_[:, 2] = seq_length
        boundary_[:, 3] = frames

        for rnnt_type in ["regular"]:
            for device in self.devices[1:]:
                print('device', device)
                # normal rnnt
                am = am_.to(device)
                lm = lm_.to(device)
                symbols = symbols_cpu.to(device)
                boundary = boundary_.to(device)
                logits = am.unsqueeze(2) + lm.unsqueeze(1)
                logits = logits.float()

                # nonlinear transform
                logits = torch.sigmoid(logits)
                logits.requires_grad_()

                fast_loss = fast_rnnt.rnnt_loss(
                    logits=logits,
                    symbols=symbols,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                    reduction="mean",
                    delay_penalty=0.2,
                )

                fast_grad = torch.autograd.grad(fast_loss, logits)
                fast_grad = fast_grad[0]

                print(f"Unpruned rnnt loss with {rnnt_type} rnnt : {fast_loss}")
                print(f"Unpruned rnnt loss grad with {rnnt_type} rnnt : {fast_grad}")

                # pruning
                simple_loss, (px_grad, py_grad) = fast_rnnt.rnnt_loss_simple(
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

                print('simple_loss', simple_loss)
                print('px_grad', px_grad)
                print('py_grad', py_grad)

#                for r in range(2, 50, 5):
                for r in range(3, 5, 5):
                    ranges = fast_rnnt.get_rnnt_prune_ranges(
                        px_grad=px_grad,
                        py_grad=py_grad,
                        boundary=boundary,
                        s_range=r,
                    )

                    # (B, T, r, C)
                    pruned_am, pruned_lm = fast_rnnt.do_rnnt_pruning(
                        am=am, lm=lm, ranges=ranges
                    )

                    logits = pruned_am + pruned_lm

                    # nonlinear transform
                    logits = torch.sigmoid(logits)
                    logits.requires_grad_()

                    pruned_loss = fast_rnnt.rnnt_loss_pruned(
                        logits=logits,
                        symbols=symbols,
                        ranges=ranges,
                        termination_symbol=terminal_symbol,
                        boundary=boundary,
                        rnnt_type=rnnt_type,
                        reduction="mean",
                        delay_penalty=0.2,
                    )

                    fast_grad = torch.autograd.grad(pruned_loss, logits)
                    fast_grad = fast_grad[0]

                    print(f"Pruning loss with range {r} : {pruned_loss}")
                    print(f"Pruned rnnt loss grad with {rnnt_type} rnnt : {fast_grad}")

    #@unittest.skip
    def test_rnnt_loss_pruned_stress(self):
        B = 2
        T = 200
        S = 50
        C = 50

        np.random.seed(12345)

        frames = torch.from_numpy(np.random.randint(S, T, (B,)))
        seq_length = torch.from_numpy(np.random.randint(3, S - 1, (B,)))
#        frames = np.random.randint(S, S+1, (B,))
#        seq_length = np.random.randint(C, C+1, (B,))

        T = torch.max(frames)
        S = torch.max(seq_length)

        am_ = torch.from_numpy(np.random.randn(B, T, C).astype('f'))
        lm_ = torch.from_numpy(np.random.randn(B, S + 1, C).astype('f'))
        symbols_cpu = torch.from_numpy(np.random.randint(0, C - 1, (B, S)))
        terminal_symbol = C - 1

        boundary_ = torch.zeros((B, 4), dtype=torch.int64)
        boundary_[:, 2] = seq_length
        boundary_[:, 3] = frames

        for rnnt_type in ["regular"]:
            for device in self.devices[1:]:
                print('device', device)
                # normal rnnt
                am = am_.to(device)
                lm = lm_.to(device)
                symbols = symbols_cpu.to(device)
                boundary = boundary_.to(device)
                logits = am.unsqueeze(2) + lm.unsqueeze(1)
                logits = logits.float()

                # nonlinear transform
                logits = torch.sigmoid(logits)
                logits.requires_grad_()

                fast_loss = fast_rnnt.rnnt_loss(
                    logits=logits,
                    symbols=symbols,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    rnnt_type=rnnt_type,
                    reduction="mean",
                    delay_penalty=0.2,
                )

                fast_grad = torch.autograd.grad(fast_loss, logits)
                fast_grad = fast_grad[0]

#                print(f"Unpruned rnnt loss with {rnnt_type} rnnt : {fast_loss}")
#                print(f"Unpruned rnnt loss grad with {rnnt_type} rnnt : {fast_grad}")

                # pruning
                simple_loss, (px_grad, py_grad) = fast_rnnt.rnnt_loss_smoothed(
                    lm=lm,
                    am=am,
                    symbols=symbols,
                    termination_symbol=terminal_symbol,
                    boundary=boundary,
                    lm_only_scale=0.1,
                    am_only_scale=0.2,
                    rnnt_type=rnnt_type,
                    return_grad=True,
                    reduction="none",
                    delay_penalty=0.2,
                )
                
#                print('simple_loss', simple_loss)
#                print('px_grad', px_grad)
#                print('py_grad', py_grad)

                grad_mean = 0

                for r in range(5, 51, 1):
                    ranges = fast_rnnt.get_rnnt_prune_ranges(
                        px_grad=px_grad,
                        py_grad=py_grad,
                        boundary=boundary,
                        s_range=r,
                    )

                    # (B, T, r, C)
                    pruned_am, pruned_lm = fast_rnnt.do_rnnt_pruning(
                        am=am, lm=lm, ranges=ranges
                    )

                    logits = pruned_am + pruned_lm

                    # nonlinear transform
                    logits = torch.sigmoid(logits)
                    logits.requires_grad_()

                    pruned_loss = fast_rnnt.rnnt_loss_pruned(
                        logits=logits,
                        symbols=symbols,
                        ranges=ranges,
                        termination_symbol=terminal_symbol,
                        boundary=boundary,
                        rnnt_type=rnnt_type,
                        reduction="mean",
                        delay_penalty=0.2,
                    )

                    fast_grad = torch.autograd.grad(pruned_loss, logits)
                    fast_grad = fast_grad[0]
                    grad_mean += fast_grad.mean([1, 2])

                    print(f"Pruning loss with range {r} : {pruned_loss}")

                print(f"Pruned rnnt loss grad_mean with {rnnt_type} rnnt : {grad_mean}")



if __name__ == "__main__":
    unittest.main()
