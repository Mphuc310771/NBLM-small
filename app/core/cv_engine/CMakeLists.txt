cmake_minimum_required(VERSION 3.10)
project(VisionEngine)

set(CMAKE_CXX_STANDARD 17)

# Check for OpenCV and Tesseract development libraries on the system
find_package(OpenCV QUIET)
find_path(TESSERACT_INCLUDE_DIR NAMES tesseract/baseapi.h)
find_library(TESSERACT_LIBRARY NAMES tesseract)

if(OpenCV_FOUND AND TESSERACT_INCLUDE_DIR AND TESSERACT_LIBRARY)
    message(STATUS "OpenCV and Tesseract found. Building high-performance native vision processor library.")
    include_directories(${OpenCV_INCLUDE_DIRS} ${TESSERACT_INCLUDE_DIR})
    add_library(vision_engine SHARED vision_processor.cpp)
    target_link_libraries(vision_engine ${OpenCV_LIBS} ${TESSERACT_LIBRARY})
else()
    message(STATUS "OpenCV or Tesseract libraries not detected. Building native vision processor with mock fallback enabled.")
    add_definitions(-DMOCK_VISION_ENGINE)
    add_library(vision_engine SHARED vision_processor.cpp)
endif()
