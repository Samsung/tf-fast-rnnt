include_directories(${CMAKE_SOURCE_DIR})
include(tensorflow)

set(srcs
mutual_information_cuda.cu
)

add_library(tf_mutual_information_core ${srcs})

target_include_directories(tf_mutual_information_core PUBLIC ${PYTHON_INCLUDE_DIRS})
