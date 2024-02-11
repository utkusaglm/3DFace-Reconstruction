#include <FreeImage.h>
#include "include/Optimizer.h"
#include <cstdlib>

#define SHAPE_COUNT 199
#define EXP_COUNT 100
#define COLOR_COUNT 199

using namespace std;

int main(int argc, char *argv[]) {

    //python3 show_landmark.py --source_image ../images/micheal.jpeg --target_image ../images/wolverine.jpg
    //run the python script with argv[1] as the source image and argv[2] as the target image
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <source_image_path> <source_landmark_path>   <target_image_path> <target_landmark_path>" << endl;
        return 1;
    }
    string command = "python3 ../show_landmark.py --source_image " + std::string(argv[1])+ " --target_image " + string(argv[2]);
    system(command.c_str());
    cout << "Initialize BFM ..." << endl;

    BFMManager bfm;
    bfm.BFMSetup();

    string source_image_path  = argv[1];
    size_t last_slash_idx = source_image_path.find_last_of("/\\");
    size_t last_dot_idx = source_image_path.find_last_of(".");
    string source_image_name = source_image_path.substr(last_slash_idx + 1, last_dot_idx - last_slash_idx - 1);
    string source_landmark_path = "../landmarks/" + source_image_name + "_landmarks.txt";


    string target_image_path  = argv[2];
    last_slash_idx = target_image_path.find_last_of("/\\");
    last_dot_idx = target_image_path.find_last_of(".");
    string target_image_name = target_image_path.substr(last_slash_idx + 1, last_dot_idx - last_slash_idx - 1);
    string target_landmark_path = "../landmarks/" + target_image_name + "_landmarks.txt";

    cout << "Source Image: " << source_image_path << endl;
    cout << "Source Landmark: " << source_landmark_path << endl;
    cout << "Target Image: " << target_image_path << endl;
    cout << "Target Landmark: " << target_landmark_path << endl;

    //Source Image
    cv::Mat source_image = cv::imread(source_image_path);
    MatrixXd source_imageLandmark = bfm.loadImageLandmark(source_landmark_path);
    BFMParameters source_param;
    MatrixXd source_rotation(3, 3);
    MatrixXd source_translation(3, 1);
    auto source_result = optimize(bfm, source_image, source_imageLandmark);
    source_param = get<0>(source_result);
    source_rotation = get<1>(source_result);
    source_translation = get<2>(source_result);
    double source_fov = get<3>(source_result);
    bfm.render(source_image, source_param, source_rotation, source_translation, source_fov,"../reconstruction/source_reconstruction.png");

    //Target Image
    cv::Mat target_image = cv::imread(target_image_path);
    MatrixXd target_imageLandmark = bfm.loadImageLandmark(target_landmark_path);
    BFMParameters target_param;
    MatrixXd target_rotation(3, 3);
    MatrixXd target_translation(3, 1);
    auto target_result = optimize(bfm, target_image, target_imageLandmark);
    target_param = std::get<0>(target_result);
    target_rotation = std::get<1>(target_result);
    target_translation = std::get<2>(target_result);
    double target_fov = std::get<3>(target_result);
    bfm.render(target_image, target_param, target_rotation, target_translation, target_fov,"../reconstruction/target_reconstruction.png");

    //Blendshape
    target_param.w_expression = source_param.w_expression;
    bfm.render(target_image, target_param, target_rotation, target_translation, target_fov,"../reconstruction/target_reconstruction_final.png");

    return 0;
}