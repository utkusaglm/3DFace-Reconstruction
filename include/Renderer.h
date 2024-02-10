#pragma once
#define GLFW_INCLUDE_GLU
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <ceres/ceres.h>

#include "Eigen.h"


GLFWwindow* initRendering(int width, int height);
void terminateRendering();
cv::Mat renderMesh(GLFWwindow* window, int width, int height, MatrixXi &triangles, MatrixXd &vertices, MatrixXd &colors, std::vector<int> landmarks, bool draw_landmarks);
MatrixXd getIntrinsicsMatrix(MatrixXd &intrinsics, double width, double height);
MatrixXd getExtrinsicsMatrix(MatrixXd &rotation, MatrixXd &translation);
MatrixXd calculatePerspectiveProjection(int width, int height, MatrixXd &rotation, MatrixXd &translation, MatrixXd &vertices, double fov);
MatrixXd calculate_perspective_matrix(double width, double height, double fov);
template <typename T>
static Matrix<T, 3, 4> calculate_transformation_matrix(Matrix<T, 3, 1> translation, Matrix<T, 3, 3> rotation);