cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(face_reconstruction)

# Options
# Set C++ flags
set(CMAKE_CXX_STANDARD 14)
set(LIBRARY_DIR Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(OpenCV_DIR ${LIBRARY_DIR}/OpenCV/lib/cmake/opencv4 CACHE PATH "Path to installed OpenCV")
set(glog_DIR ${LIBRARY_DIR}/Glog/lib/cmake/glog CACHE PATH "Path to installed glog")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/lib/cmake/Ceres CACHE PATH "Path to installed Ceres")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
# Define header and source files
set(HEADERS
        include/Eigen.h
        include/FreeImageHelper.h
        include/BFMManager.h
        include/Renderer.h
        include/Optimizer.h
        )

set(SOURCES
        load_binary.cpp
        src/FreeImageHelper.cpp
        src/BFMManager.cpp
        src/Renderer.cpp
        src/Optimizer.cpp
        )

add_executable(face_reconstruction ${HEADERS} ${SOURCES})
target_include_directories(face_reconstruction PUBLIC ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})
target_link_libraries(face_reconstruction Eigen3::Eigen glfw GLEW  freeimage ceres ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${OPENGL_gl_LIBRARY} )