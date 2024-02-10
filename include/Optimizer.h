#include "Eigen.h"
#include <FreeImage.h>
#include <iostream>
#include "BFMManager.h"
#include "ceres/ceres.h"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "Renderer.h"
#include "Regularization.h"
#include <tuple>

tuple<BFMParameters,MatrixXd,MatrixXd,double> optimize(BFMManager bfm, cv::Mat image, MatrixXd targetLandmarks);