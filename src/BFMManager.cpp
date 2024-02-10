#include "../include/BFMManager.h"
#include "../include/Renderer.h"


static std::string color_mean_path = "../data/color_model_mean.bin"; // 85764
static std::string color_pcaBasis_path = "../data/color_model_pcaBasis.bin"; // 85764 * 199
static std::string color_pcaVariance_path = "../data/color_model_pcaVariance.bin"; // 199

static std::string expression_mean_path = "../data/expression_model_mean.bin"; // 85764
static std::string expression_pcaBasis_path = "../data/expression_model_pcaBasis.bin"; // 85764 * 100
static std::string expression_pcaVariance_path = "../data/expression_model_pcaVariance.bin"; // 100

static std::string shape_mean_path = "../data/shape_model_mean.bin"; // 85764
static std::string shape_pcaBasis_path = "../data/shape_model_pcaBasis.bin"; // 85764 * 199
static std::string shape_pcaVariance_path = "../data/shape_model_pcaVariance.bin"; // 199

static std::string triangles_path = "../data/shape_representer_cells.bin"; // 3 * 56572
static std::string landmarks_path = "../data/Landmarks68_BFM.anl";


template <typename T>
std::vector<T> BFMManager::loadBFMData(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Determine the size of the file
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Allocate a vector to hold the file contents
    std::vector<char> fileContents(fileSize);

    // Read the file into the vector
    file.read(fileContents.data(), fileSize);

    if (!file) {
        std::cerr << "Error reading file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy raw bytes into the vector of type T
    std::vector<T> TVector(fileSize / sizeof(T));
    std::memcpy(TVector.data(), fileContents.data(), fileSize);

    return TVector;
}


std::vector<double> BFMManager::toDouble(std::vector<float> floatVector) {
    std::vector<double> doubleVector(floatVector.begin(), floatVector.end());
    return doubleVector;
}


void BFMManager::BFMSetup() {
    std::cout << "INSIDE BFM SETUP!" << std::endl;

    std::vector<double> colorMeanV = toDouble(loadBFMData<float>(color_mean_path.c_str()));
    std::vector<double> colorPCABasisV = toDouble(loadBFMData<float>(color_pcaBasis_path.c_str()));
    std::vector<double> colorPCAVarianceV = toDouble(loadBFMData<float>(color_pcaVariance_path.c_str()));

    std::vector<double> exprMeanV = toDouble(loadBFMData<float>(expression_mean_path.c_str()));
    std::vector<double> exprPCABasisV = toDouble(loadBFMData<float>(expression_pcaBasis_path.c_str()));
    std::vector<double> exprPCAVarianceV = toDouble(loadBFMData<float>(expression_pcaVariance_path.c_str()));

    std::vector<double> shapeMeanV = toDouble(loadBFMData<float>(shape_mean_path.c_str()));
    std::vector<double> shapePCABasisV = toDouble(loadBFMData<float>(shape_pcaBasis_path.c_str()));
    std::vector<double> shapePCAVarianceV = toDouble(loadBFMData<float>(shape_pcaVariance_path.c_str()));
    std::vector<int> triangleV = loadBFMData<int>(triangles_path.c_str());

    colorMean = Map<Matrix<double, 85764, 1>>(colorMeanV.data());
    colorPCABasis = Map<Matrix<double, 85764, 199, RowMajor>>(colorPCABasisV.data());
    colorPCAVariance = Map<Matrix<double, 199, 1>>(colorPCAVarianceV.data());

    exprMean = Map<Matrix<double, 85764, 1>>(exprMeanV.data());
    exprPCABasis = Map<Matrix<double, 85764, 100, RowMajor>>(exprPCABasisV.data());
    exprPCAVariance = Map<Matrix<double, 100, 1>>(exprPCAVarianceV.data());

    shapeMean = Map<Matrix<double, 85764, 1>>(shapeMeanV.data());
    shapePCABasis = Map<Matrix<double, 85764, 199, RowMajor>>(shapePCABasisV.data());
    shapePCAVariance = Map<Matrix<double, 199, 1>>(shapePCAVarianceV.data());
    triangle = Map<Matrix<int, 56572, 3>>(triangleV.data());

    loadLandmarks();
}

MatrixXd BFMManager::loadImageLandmark(const string filename){
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Read the points from the file
    std::vector<std::pair<double, double>> pointList;
    double x, y;
    while (file >> x >> y) {
        pointList.push_back(std::make_pair(x, y));
    }
    file.close();

    MatrixXd points(2, pointList.size());
    for (size_t i = 0; i < pointList.size(); ++i) {
        points(0, i) = pointList[i].first;
        points(1, i) = pointList[i].second;
    }

    return points.transpose();
}


void BFMManager::loadLandmarks() {
    std::ifstream infile(landmarks_path.c_str());
    int number;
    while (true) {
        if (infile.peek() == EOF) {
            break;
        }
        else if (infile.peek() == '%') {
            // ignore lines that are commented out
            infile.ignore(1000000, '\n');
        }
        else {
            if (infile >> number) {
                landmarkIndices.push_back(number);
            }
            else {
                // Handle error or invalid input
                infile.clear(); // Clear the error flags
                infile.ignore(1000000, '\n'); // Ignore the rest of the line
            }
        }
    }
    infile.close();
}


MatrixXd BFMManager::calcCoef(MatrixXd &mean, MatrixXd &basis, MatrixXd &variance, VectorXd &param) {
    MatrixXd result = mean + basis * variance.cwiseSqrt().cwiseProduct(param);
    return result;
}


void BFMManager::render(cv::Mat image, BFMParameters params, MatrixXd &rotation, MatrixXd &translation, double fov, const string filename) {
    std::cout << "init Render" << std::endl;
    int width = image.cols;
    int height = image.rows;
    std::cout << width << ", " << height << std::endl;
    auto context = initRendering(width, height);

    std::cout << "Calc Coef" << std::endl;
    MatrixXd shape_result = calcCoef(shapeMean, shapePCABasis, shapePCAVariance, params.w_shape);
    MatrixXd expr_result = calcCoef(exprMean, exprPCABasis, exprPCAVariance, params.w_expression);
    MatrixXd color_result = calcCoef(colorMean, colorPCABasis, colorPCAVariance, params.w_color);

    std::cout << "Get Vertices" << std::endl;
    MatrixXd vertices = shape_result + expr_result;
    writeObj(shape_result, expr_result, color_result, triangle);
    std::cout << "Project Vertices" << std::endl;
    MatrixXd projected_vertices = calculatePerspectiveProjection(width, height, rotation, translation, vertices, fov);
    writeObj(projected_vertices, color_result, triangle);

    std::cout << "Render IMAGE" << std::endl;
    auto image_render = renderMesh(context, width, height, triangle, projected_vertices, color_result, landmarkIndices, false);
    // std::cout << image_render << std::endl;
    std::cout << "WRITE IMAGE" << std::endl;
    //to remove the background
    double alpha =1;
    double beta;
    beta = 1 - alpha;
    cv::Mat dst;
    addWeighted(image_render, alpha, image, beta, 0.0, dst);
    cv::imwrite(filename, dst);
    terminateRendering();
}


BFMParameters BFMManager::getMeanParams() {
    BFMParameters params;
    params.w_color = VectorXd::Zero(199);
    params.w_shape = VectorXd::Zero(199);
    params.w_expression = VectorXd::Zero(100);

    return params;
}


BFMParameters BFMManager::getRandomParams() {
    BFMParameters params;
    params.w_color = VectorXd::Random(199) * 0.02;
    params.w_shape = VectorXd::Random(199) * 0.02;
    params.w_expression = VectorXd::Random(100) * 0.02;

    return params;
}


void BFMManager::writeObj(MatrixXd &shape, MatrixXd &expression, MatrixXd &color, MatrixXi &triangle) {
    std::ofstream obj_file("face.obj");

    int nVertices = 85764 / 3;
    int nTriangles = 56572;
    MatrixXd blendShape = shape + expression;
        for (int i = 0; i < nVertices; i++) {
        obj_file << "v " << blendShape(i * 3) << " " << blendShape(i * 3 + 1) << " " << blendShape(i * 3 + 2) << " "
                 << color(i * 3) << " " << color(i * 3 + 1) << " " << color(i * 3 + 2) << "\n";
    }

    for (int i = 0; i < nTriangles; i++) {
        obj_file << "f " << triangle(i, 0) + 1 << " " << triangle(i, 1) + 1 << " " << triangle(i, 2) + 1 << "\n";
    }

    obj_file.close();

}


void BFMManager::writeObj(MatrixXd &vertices, MatrixXd &color, MatrixXi &triangle) {
    std::ofstream obj_file("face_projected.obj");

    int nVertices = 85764 / 3;
    int nTriangles = 56572;
    for (int i = 0; i < nVertices; i++) {
        // int z = vertices(i, 2);
        obj_file << "v " << vertices(i, 0) << " " << vertices(i, 1) << " " << vertices(i, 2) << " "
                 << color(i * 3) << " " << color(i * 3 + 1) << " " << color(i * 3 + 2) << "\n";
    }

    for (int i = 0; i < nTriangles; i++) {
        obj_file << "f " << triangle(i, 0) + 1 << " " << triangle(i, 1) + 1 << " " << triangle(i, 2) + 1 << "\n";
    }

    obj_file.close();

}