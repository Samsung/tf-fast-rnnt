include_directories(${CMAKE_SOURCE_DIR})

include(transform)
include(tensorflow)

# please keep the list sorted
set(fast_rnnt_srcs
  tf_fast_rnnt_op.cc
)

add_library(_tf_fast_rnnt ${fast_rnnt_srcs})
set_target_properties(_tf_fast_rnnt PROPERTIES PREFIX "")

find_package(CUDA REQUIRED)

# Do what the new package does
find_library(CUDA_DRIVER_LIBRARY
             NAMES cuda_driver cuda
             HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                   ENV CUDA_PATH
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
if (NOT CUDA_DRIVER_LIBRARY)
    # Don't try any stub directories until we have exhausted all other search locations.
    find_library(CUDA_DRIVER_LIBRARY
                 NAMES cuda_driver cuda
                 HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                       ENV CUDA_PATH
                 PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs)
endif ()
mark_as_advanced(CUDA_DRIVER_LIBRARY)
##

target_include_directories(
  _tf_fast_rnnt 
    PRIVATE 
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(
  _tf_fast_rnnt 
  PRIVATE
  tf_mutual_information_core
  ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY}
)

if(NOT FT_WITH_CUDA)
  transform(OUTPUT_VARIABLE fast_rnnt_srcs SRCS ${fast_rnnt_srcs})
endif()
