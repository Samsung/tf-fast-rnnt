# PYTHON_EXECUTABLE is set by pybind11.cmake
message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')" OUTPUT_VARIABLE Tensorflow_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE Tensorflow_LINK_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

# set the global CMAKE_CXX_FLAGS so that
# optimized_transducer uses the same abi flag as PyTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Tensorflow_COMPILE_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${Tensorflow_COMPILE_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed ${Tensorflow_LINK_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
