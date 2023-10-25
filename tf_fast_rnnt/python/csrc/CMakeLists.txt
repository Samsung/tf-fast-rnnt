include_directories(${CMAKE_SOURCE_DIR})

include(transform)
include(tensorflow)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# please keep the list sorted
set(fast_rnnt_srcs
  tf_fast_rnnt_op.cc
)

add_library(_tf_fast_rnnt ${fast_rnnt_srcs})
set_target_properties(_tf_fast_rnnt PROPERTIES PREFIX "")

target_include_directories(
  _tf_fast_rnnt 
    PRIVATE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(
  _tf_fast_rnnt 
  PRIVATE
  tf_mutual_information_core
    ${CUDART_LIBRARY}
)

if(NOT FT_WITH_CUDA)
  transform(OUTPUT_VARIABLE fast_rnnt_srcs SRCS ${fast_rnnt_srcs})
endif()
