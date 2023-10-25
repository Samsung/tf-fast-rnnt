This project implements pruned rnnt for tensorflow with small limitations.

```
1. supports GPU only.
2. boundary information is mandatory.
```

It's based on the original fast RNN-T loss implementation in [k2](https://github.com/k2-fsa/k2) project and following is based from https://github.com/k2-fsa/fast_rnnt

---
This project implements a method for faster and more memory-efficient RNN-T loss computation, called `pruned rnnt`.

Note: The original fast RNN-T loss implementation is in [k2](https://github.com/k2-fsa/k2) project and this project is based on https://github.com/k2-fsa/fast_rnnt.
We make `tf_fast_rnnt` a stand-alone project for tensorflow users.

## How does the pruned-rnnt work ?

We first obtain pruning bounds for the RNN-T recursion using a simple joiner network that is just an addition of the encoder and decoder, then we use those pruning bounds to evaluate the full, non-linear joiner network.

The picture below display the gradients (obtained by `rnnt_loss_simple` with `return_grad=true`) of lattice nodes, at each time frame, only a small set of nodes have a non-zero gradient, which justifies the pruned RNN-T loss, i.e., putting a limit on the number of symbols per frame.

<img src="https://user-images.githubusercontent.com/5284924/158116784-4dcf1107-2b84-4c0c-90c3-cb4a02f027c9.png" width="900" height="250" />

> This picture is taken from [here](https://github.com/k2-fsa/icefall/pull/251)

## Installation

You can install it via `pip`:

```
pip install tf_fast_rnnt
```

You can also install from source:

```
cd tf_fast_rnnt
pip install .
```

To check that `tf_fast_rnnt` was installed successfully, please run

```
python3 -c "import tf_fast_rnnt; print(tf_fast_rnnt.__version__)"
```

which should print the version of the installed `tf_fast_rnnt`, e.g., `1.2`.


### How to display installation log ?

Use

```
pip install --verbose tf_fast_rnnt
```

### How to reduce installation time ?

Use

```
export FT_MAKE_ARGS="-j"
pip install --verbose tf_fast_rnnt
```

It will pass `-j` to `make`.

### Which version of tensorflow is supported ?

It has been tested on tensorflow >= 2.9.3.

Note: The cuda version of the tensorflow should be the same as the cuda version in your environment,
or it will cause a compilation error.

## Usage

### For rnnt_loss_simple

This is a simple case of the RNN-T loss, where the joiner network is just
addition.

Note: termination_symbol plays the role of blank in other RNN-T loss implementations, we call it termination_symbol as it terminates symbols of current frame.

```python
am = np.random.randn(B, T, C).astype('f')
lm = np.random.randn(B, S + 1, C).astype('f')
symbols = np.random.randint(0, C - 1, (B, S)).astype(np.int64)
terminal_symbol = C - 1

boundary = np.zeros((B, 4))
boundary[:, 2] = seq_length
boundary[:, 3] = frames
boundary = tf.convert_to_tensor(boundary, dtype=tf.int64)

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
```

### For rnnt_loss_pruned

`rnnt_loss_pruned` can not be used alone, it needs the gradients returned by `rnnt_loss_simple/rnnt_loss_smoothed` to get pruning bounds.

```python
am = np.random.randn(B, T, C).astype('f')
lm = np.random.randn(B, S + 1, C).astype('f')
symbols = np.random.randint(0, C - 1, (B, S)).astype(np.int64)
terminal_symbol = C - 1

boundary = np.zeros((B, 4))
boundary[:, 2] = seq_length
boundary[:, 3] = frames
boundary = tf.convert_to_tensor(boundary, dtype=tf.int64)

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
s_range = 5  # can be other values
ranges = tf_fast_rnnt.get_rnnt_prune_ranges(
    px_grad=px_grad,
    py_grad=py_grad,
    boundary=boundary,
    s_range=s_range,
)

am_pruned, lm_pruned = tf_fast_rnnt.do_rnnt_pruning(am=am, lm=lm, ranges=ranges)

logits = model.joiner(am_pruned, lm_pruned)
pruned_loss = tf_fast_rnnt.rnnt_loss_pruned(
    logits=logits,
    symbols=symbols,
    ranges=ranges,
    termination_symbol=termination_symbol,
    boundary=boundary,
    reduction="sum",
)
```

You can also find recipes [here](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless) that uses `rnnt_loss_pruned` to train a model.

## Benchmarking

The [repo](https://github.com/csukuangfj/transducer-loss-benchmarking) compares the speed and memory usage of several transducer losses, the summary in the following table is taken from there, you can check the repository for more details.

Note: As we declared above, `fast_rnnt` is also implemented in [k2](https://github.com/k2-fsa/k2) project, so `k2` and `fast_rnnt` are equivalent in the benchmarking.

|Name	               |Average step time (us) | Peak memory usage (MB)|
|--------------------|-----------------------|-----------------------|
|torchaudio          |601447                 |12959.2                |
|fast_rnnt(unpruned) |274407                 |15106.5                |
|fast_rnnt(pruned)   |38112                  |2647.8                 |
|optimized_transducer|567684                 |10903.1                |
|warprnnt_numba      |229340                 |13061.8                |
|warp-transducer     |210772                 |13061.8                |
