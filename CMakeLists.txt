cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(marian CXX)
SET(CMAKE_CXX_FLAGS " -std=c++11 -g -O3 -Wno-unused-result -Wno-deprecated -fPIC")
LIST(APPEND CUDA_NVCC_FLAGS --default-stream per-thread; -std=c++11; -g; -O3; -arch=sm_35; --use_fast_math; -Xcompiler '-fPIC')
add_definitions(-DCUDA_API_PER_THREAD_DEFAULT_STREAM)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

include_directories(${amunn_SOURCE_DIR})
find_package(CUDA REQUIRED)

find_package(ZLIB)
if(ZLIB_FOUND)
    include_directories(${ZLIB_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${ZLIB_LIBRARIES})
else(ZLIB_FOUND)
    message(SEND_ERROR "Cannot find zlib" )
endif(ZLIB_FOUND)

find_package(Boost COMPONENTS system timer iostreams filesystem chrono program_options)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
else(Boost_FOUND)
    message(SEND_ERROR "Cannot find Boost libraries. Terminating." )
endif(Boost_FOUND)

include_directories(${marian_SOURCE_DIR}/src)
add_subdirectory(src)

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)
