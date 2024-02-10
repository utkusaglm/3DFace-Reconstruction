#pragma once

#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "Eigen.h"
#include <string>
#include <FreeImage.h>


using namespace std;


struct BFMParameters {
    VectorXd w_color;
    VectorXd w_expression;
    VectorXd w_shape;
};

struct Image {
        FIBITMAP* image;
        string path;
};

class BFMManager {
public:
        template <typename T>
        std::vector<T> loadBFMData(const std::string& filename);
        std::vector<double> toDouble(std::vector<float> floatVector);
        void BFMSetup();

        void loadLandmarks();

        void render(cv::Mat image, BFMParameters params, MatrixXd &rotation, MatrixXd &translation, double fov, const string filename);

        MatrixXd calcCoef(MatrixXd &mean, MatrixXd &basis, MatrixXd &variance, VectorXd &param);

        BFMParameters getMeanParams();
        BFMParameters getRandomParams();

        static void writeObj(MatrixXd &shape, MatrixXd &expression, MatrixXd &color, MatrixXi &triangle);
        static void writeObj(MatrixXd &vertices, MatrixXd &color, MatrixXi &triangle);

        MatrixXd getShapeMean() const { return shapeMean; }
        MatrixXd getShapePCABasis() const {return shapePCABasis; }
        MatrixXd getShapePCAVariance() const { return shapePCAVariance; }

        MatrixXd getExpressionMean() const { return exprMean; }
        MatrixXd getExpressionPCABasis() const { return exprPCABasis; }
        MatrixXd getExpressionPCAVariance() const { return exprPCAVariance; }

        MatrixXd getColorMean() const { return colorMean; }
        MatrixXd getColorPCABasis() const { return colorPCABasis; }
        MatrixXd getColorPCAVariance() const { return colorPCAVariance; }

        MatrixXi getTriangle() const { return triangle; }

        std::vector<int> getLandmarks() const { return landmarkIndices; }
        
        MatrixXd projected2D(MatrixXd imageLandmark); //project the 3D coordinate points to 2D

        MatrixXd loadImageLandmark(const string filename);

public:
        int width;
        int height;
        MatrixXd shapeMean;
        MatrixXd shapePCABasis;
        MatrixXd shapePCAVariance;

        MatrixXd exprMean;
        MatrixXd exprPCABasis;
        MatrixXd exprPCAVariance;

        MatrixXd colorMean;
        MatrixXd colorPCABasis;
        MatrixXd colorPCAVariance;

        MatrixXi triangle;

        std::vector<int> landmarkIndices;
};