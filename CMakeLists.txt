cmake_minimum_required(VERSION 3.12)
project(yolov5-dnn-cpp-python-v2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_RELEASE)


# for windows dll symbol
IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

# Let's set compiler-specific flags
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    # G++
    target_compile_options(demo PRIVATE -Wall -Wextra)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    # MSVC
    target_compile_options(demo PRIVATE /EHsc /MTd /W2 /c)
    # Set the DLLEXPORT variable to export symbols
    target_compile_definitions(demo PRIVATE WIN_EXPORT)
endif()


# the cv
IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(CUDA_VERSION 8.0)
    find_package(CUDA 8.0 REQUIRED)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")


add_executable(yolov5-dnn-cpp-python-v2 main_yolo.cpp)
find_package(OpenCV REQUIRED)
target_link_libraries(yolov5-dnn-cpp-python-v2 ${OpenCV_LIBS})
