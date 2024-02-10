#include <FreeImage.h>
#include "include/Optimizer.h"



#define SHAPE_COUNT 199
#define EXP_COUNT 100
#define COLOR_COUNT 199

using namespace std;

int main() {
    std::cout << "Initialize BFM ..." << std::endl;
    BFMManager bfm;
    bfm.BFMSetup();

    //Wolverine
    const char* points_file_path = "../wolverine.txt";
    cv::Mat image_wolverine = cv::imread("../images/wolverine.jpg");
    MatrixXd imageLandmark_wolverine = bfm.loadImageLandmark(points_file_path);
    BFMParameters param_wolverine;
    MatrixXd rotation_wolverine(3, 3);
    MatrixXd translation_wolverine(3, 1);
    auto result = optimize(bfm, image_wolverine, imageLandmark_wolverine);
    param_wolverine = std::get<0>(result);
    rotation_wolverine = std::get<1>(result);
    translation_wolverine = std::get<2>(result);
    double fov_wolverine = std::get<3>(result);
    //
    bfm.render(image_wolverine, param_wolverine, rotation_wolverine, translation_wolverine, fov_wolverine,"../reconstruction/wolverine_reconstruction.png");

    //Micheal
    const char* points_file_path_micheal = "../micheal.txt";
    cv::Mat image_micheal = cv::imread("../images/micheal.jpeg");
    MatrixXd imageLandmark_micheal = bfm.loadImageLandmark(points_file_path_micheal);
    BFMParameters param_micheal;
    MatrixXd rotation_micheal(3, 3);
    MatrixXd translation_micheal(3, 1);
    auto result_micheal = optimize(bfm, image_micheal, imageLandmark_micheal);
    param_micheal = std::get<0>(result_micheal);
    rotation_micheal = std::get<1>(result_micheal);
    translation_micheal = std::get<2>(result_micheal);
    double fov_micheal = std::get<3>(result_micheal);

    bfm.render(image_micheal, param_micheal, rotation_micheal, translation_micheal, fov_micheal,"../reconstruction/micheal_reconstruction.png");

    param_wolverine.w_expression = param_micheal.w_expression;

    bfm.render(image_wolverine, param_wolverine, rotation_wolverine, translation_wolverine, fov_wolverine,"../reconstruction/wolverine_reconstruction_final.png");

    return 0;
}