#include "../include/Optimizer.h"


#define SHAPE_COUNT 199
#define EXP_COUNT 100
#define COLOR_COUNT 199

#define SHAPE_REGULARIZATION_WEIGHT 100
#define EXPRESSION_REGULARIZATION_WEIGHT 24
#define COLOR_REGULARIZATION_WEIGHT 24

//1. calculate face geometry: face = shape_result + expression_result, where these 2 comes from calcCoef() function with its own respective input.
//2. ⁠After that, extract the point by face[landmarkIndices * 3], face[landmarkIndices * 3 + 1], face[landmarkIndices * 3 + 2].
//3. ⁠Finally, project this point into 2D by calling the functions I implemented at Renderer.cpp, then compare with the observed landmarks


struct SparseCost {
    SparseCost(BFMManager* bfm_, Vector2d observed_landmark_, int vertex_id_, int image_width_, int image_height_) :
            bfm{bfm_}, observed_landmark{observed_landmark_}, vertex_id{vertex_id_}, image_width{image_width_}, image_height{image_height_}
    {}
    template <typename T>
    bool operator()(T const* camera_rotation, T const* camera_translation, T const* fov, T const* shape_weights, T const* exp_weights, T* residuals) const {
        Matrix<T, 4, 1> face_model;

        face_model(0) = T(bfm->shapeMean(vertex_id*3, 0)) + T(bfm->exprMean(vertex_id * 3,0));
        face_model(1) = T(bfm->shapeMean(vertex_id*3+1, 0)) + T(bfm->exprMean(vertex_id * 3+1,0));
        face_model(2) = T(bfm->shapeMean(vertex_id*3+2, 0)) + T(bfm->exprMean(vertex_id * 3+2,0));
        face_model(3) = T(1);

        for (int i = 0; i < SHAPE_COUNT; i++) {
            T value = T(sqrt(bfm->shapePCAVariance(i,0))) * shape_weights[i];
            face_model(0) += T(bfm->shapePCABasis(vertex_id*3, i)) * value;
            face_model(1) += T(bfm->shapePCABasis(vertex_id*3+1, i)) * value;
            face_model(2) += T(bfm->shapePCABasis(vertex_id*3+2, i)) * value;
        }

        for (int i = 0; i < EXP_COUNT; i++) {
            T value = T(sqrt(bfm->exprPCAVariance(i,0))) * exp_weights[i];
            face_model(0) += T(bfm->exprPCABasis(vertex_id*3, i)) * value;
            face_model(1) += T(bfm->exprPCABasis(vertex_id*3+1, i)) * value;
            face_model(2) += T(bfm->exprPCABasis(vertex_id*3+2, i)) * value;
        }

        Matrix<T, 4, 4> extrinsics_matrix{
            {camera_rotation[0], camera_rotation[3], camera_rotation[6], camera_translation[0]},
            {camera_rotation[1], camera_rotation[4], camera_rotation[7], camera_translation[1]},
            {camera_rotation[2], camera_rotation[5], camera_rotation[8], camera_translation[2]},
            {T(0), T(0), T(0), T(1)}
        };

        T zNear = T(0.1);
        T zFar = T(1000.0);
        T rad = fov[0] * T(M_PI / 180);
        T tanHalfFovy = tan(rad / T(2));
        T aspect_ratio = T(image_width) / T(image_height);

        Matrix<T, 4, 4> intrinsics_matrix{
            {T(0), T(0), T(0), T(0)},
            {T(0), T(0), T(0), T(0)},
            {T(0), T(0), T(0), T(0)},
            {T(0), T(0), T(0), T(0)}
        };

        intrinsics_matrix(0, 0) = T(1) / (aspect_ratio * tanHalfFovy);
        intrinsics_matrix(1, 1) = T(1) / tanHalfFovy;
        intrinsics_matrix(2, 2) = -(zFar + zNear) / (zFar - zNear);
        intrinsics_matrix(3, 2) = T(-1);
        intrinsics_matrix(2, 3) = -(T(2) * zFar * zNear) / (zFar - zNear);

        Matrix<T, 4, 1> projection = extrinsics_matrix * face_model;
        Matrix<T, 4, 1> projection_final = intrinsics_matrix * projection;

        T w = projection_final(3);
        T transformed_x = (projection_final(0) / w + T(1)) / T(2) * T(image_width);
		T transformed_y = (projection_final(1) / w + T(1)) / T(2) * T(image_height);
		transformed_y = T(image_height) - transformed_y;

        residuals[0] = transformed_x - T(observed_landmark[0]);
		residuals[1] = transformed_y - T(observed_landmark[1]);

        return true;
    }
private:
    const BFMManager* bfm;
    const Vector2d observed_landmark;
    const int vertex_id;
    const int image_width;
    const int image_height;
};


struct DenseCost {
    DenseCost(BFMManager* bfm_, cv::Mat* image_, int vertex_id_, MatrixXd &extrinsics_matrix_, MatrixXd &intrinsics_matrix_, VectorXd &w_shape_, VectorXd &w_expression_) :
            bfm{bfm_}, image{image_}, vertex_id{vertex_id_}, extrinsics_matrix{extrinsics_matrix_}, intrinsics_matrix{intrinsics_matrix_}, w_shape{w_shape_}, w_expression{w_expression_}
    {}
    template <typename T>
    bool operator()(T const* color_weights, T* residuals) const {
        Matrix<double, 4, 1> face_model;
        Matrix<T, 3, 1> face_color;

        face_model(0) = bfm->shapeMean(vertex_id * 3, 0) + bfm->exprMean(vertex_id * 3, 0);
        face_model(1) = bfm->shapeMean(vertex_id * 3 + 1, 0) + bfm->exprMean(vertex_id * 3 + 1, 0);
        face_model(2) = bfm->shapeMean(vertex_id * 3 + 2, 0) + bfm->exprMean(vertex_id * 3 + 2, 0);
        face_model(3) = 1;

        face_color(0) = T(bfm->colorMean(vertex_id * 3, 0));
        face_color(1) = T(bfm->colorMean(vertex_id * 3 + 1, 0));
        face_color(2) = T(bfm->colorMean(vertex_id * 3 + 2, 0));

        for (int i = 0; i < SHAPE_COUNT; i++) {
            double value = sqrt(bfm->shapePCAVariance(i, 0)) * w_shape[i];
            face_model(0) += bfm->shapePCABasis(vertex_id * 3, i) * value;
            face_model(1) += bfm->shapePCABasis(vertex_id * 3 + 1, i) * value;
            face_model(2) += bfm->shapePCABasis(vertex_id * 3 + 2, i) * value;
        }

        for (int i = 0; i < EXP_COUNT; i++) {
            double value = sqrt(bfm->exprPCAVariance(i,0)) * w_expression[i];
            face_model(0) += bfm->exprPCABasis(vertex_id * 3, i) * value;
            face_model(1) += bfm->exprPCABasis(vertex_id * 3 + 1, i) * value;
            face_model(2) += bfm->exprPCABasis(vertex_id * 3 + 2, i) * value;
        }

        for (int i = 0; i < COLOR_COUNT; i++) {
            T value = T(sqrt(bfm->colorPCAVariance(i,0))) * color_weights[i];
            face_color(0) += T(bfm->colorPCABasis(vertex_id * 3, i)) * value;
            face_color(1) += T(bfm->colorPCABasis(vertex_id * 3 + 1, i)) * value;
            face_color(2) += T(bfm->colorPCABasis(vertex_id * 3 + 2, i)) * value;
        }

        Matrix<double, 4, 1> projection = intrinsics_matrix * extrinsics_matrix * face_model;

        double w = projection(3);
        double i_screen = (projection(0) / w + 1) / 2 * image->cols;
		double j_screen = image->rows - ((projection(1) / w + 1) / 2 * image->rows);

        int idx_i = floor(i_screen);
        int idx_j = floor(j_screen);

        double w1 = ((idx_i + 1) - i_screen) * ((idx_j + 1) - j_screen);
        double w2 = (i_screen - idx_i) * ((idx_j + 1) - j_screen);
        double w3 = ((idx_i + 1) - i_screen) * (j_screen - idx_j);
        double w4 = (i_screen - idx_i) * (j_screen - idx_j);

        auto rgb_1 = image->at<cv::Vec3b>(int(idx_j), int(idx_i));
        auto rgb_2 = image->at<cv::Vec3b>(int(idx_j), int(idx_i + 1));
        auto rgb_3 = image->at<cv::Vec3b>(int(idx_j + 1), int(idx_i));
        auto rgb_4 = image->at<cv::Vec3b>(int(idx_j + 1), int(idx_i + 1));


        T rgb_r = T(T(w1) * T(rgb_1[2]) + T(w2) * T(rgb_2[2]) + T(w3) * T(rgb_3[2]) + T(w4) * T(rgb_4[2])) / T(w1 + w2 + w3 + w4);
        T rgb_g = T(T(w1) * T(rgb_1[1]) + T(w2) * T(rgb_2[1]) + T(w3) * T(rgb_3[1]) + T(w4) * T(rgb_4[1])) / T(w1 + w2 + w3 + w4);
        T rgb_b = T(T(w1) * T(rgb_1[0]) + T(w2) * T(rgb_2[0]) + T(w3) * T(rgb_3[0]) + T(w4) * T(rgb_4[0])) / T(w1 + w2 + w3 + w4);

        residuals[0] = face_color(0) - (rgb_r / T(255.0));
        residuals[1] = face_color(1) - (rgb_g / T(255.0));
        residuals[2] = face_color(2) - (rgb_b / T(255.0));

        return true;
    }
private:
    const BFMManager* bfm;
    const cv::Mat* image;
    const int vertex_id;
    const MatrixXd extrinsics_matrix;
    const MatrixXd intrinsics_matrix;
    const VectorXd w_shape;
    const VectorXd w_expression;
};


tuple<BFMParameters,MatrixXd,MatrixXd,double> optimize(BFMManager bfm, cv::Mat image, MatrixXd targetLandmarks)
{
    cout << "Start Optimization " << endl;

    BFMParameters mean_param = bfm.getMeanParams();
    vector<int> bfm_landmarks = bfm.getLandmarks();

    MatrixXd rotation(3, 3);
    rotation << 1, 0, 0,
                0, 1, 0,
                0, 0, 1;
    MatrixXd translation(3, 1);
    translation << 0, 0, -400;
    double* fov = new double[1];
    fov[0] = 60.0;
    {
        ceres::Problem sparse_problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.num_threads = 12;
        options.minimizer_progress_to_stdout = true;

        for (int j = 0; j < targetLandmarks.rows(); j++) {
            Vector2d targetImageLandmark = { targetLandmarks(j, 0), targetLandmarks(j, 1) };

            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 9, 3, 1, SHAPE_COUNT, EXP_COUNT>(
                    new SparseCost(&bfm, targetImageLandmark, bfm_landmarks[j], image.cols, image.rows)
            );
            sparse_problem.AddResidualBlock(cost_function, nullptr, &rotation.data()[0], &translation.data()[0], fov, &mean_param.w_shape.data()[0], &mean_param.w_expression.data()[0]);
        }
        // keep the shape and expression constant
        sparse_problem.SetParameterBlockConstant(mean_param.w_shape.data());
        sparse_problem.SetParameterBlockConstant(mean_param.w_expression.data());
        sparse_problem.SetParameterLowerBound(&rotation.data()[0], 8, 1.0);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &sparse_problem, &summary);
    }

    {
        ceres::Problem sparse_problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.num_threads = 12;
        options.minimizer_progress_to_stdout = true;

        ceres::CostFunction* shape_cost = new ceres::AutoDiffCostFunction<ShapeCostFunction, SHAPE_COUNT, SHAPE_COUNT>(
				new ShapeCostFunction(bfm, SHAPE_REGULARIZATION_WEIGHT)
				);
		sparse_problem.AddResidualBlock(shape_cost, NULL, mean_param.w_shape.data());

		ceres::CostFunction* expression_cost = new ceres::AutoDiffCostFunction<ExpressionCostFunction, EXP_COUNT, EXP_COUNT>(
				new ExpressionCostFunction(bfm, EXPRESSION_REGULARIZATION_WEIGHT)
				);
		sparse_problem.AddResidualBlock(expression_cost, NULL, mean_param.w_expression.data());

        for (int j = 0; j < targetLandmarks.rows(); j++) {
            Vector2d targetImageLandmark = { targetLandmarks(j, 0), targetLandmarks(j, 1) };

            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 9, 3, 1, SHAPE_COUNT, EXP_COUNT>(
                    new SparseCost(&bfm, targetImageLandmark, bfm_landmarks[j], image.cols, image.rows)
            );
            sparse_problem.AddResidualBlock(cost_function, nullptr, &rotation.data()[0], &translation.data()[0], fov, &mean_param.w_shape.data()[0], &mean_param.w_expression.data()[0]);
        }

        // keep the shape and expression constant
        sparse_problem.SetParameterBlockConstant(fov);
        sparse_problem.SetParameterBlockConstant(rotation.data());
        sparse_problem.SetParameterBlockConstant(translation.data());
        ceres::Solver::Summary summary;
        ceres::Solve(options, &sparse_problem, &summary);
    }

    {
        MatrixXd extrinsics_matrix = getExtrinsicsMatrix(rotation, translation);
        MatrixXd intrinsics_matrix = calculate_perspective_matrix(image.cols, image.rows, fov[0]);

        ceres::Problem dense_problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.num_threads = 12;
        options.minimizer_progress_to_stdout = true;

        ceres::CostFunction* color_cost = new ceres::AutoDiffCostFunction<ColorCostFunction, COLOR_COUNT, COLOR_COUNT>(
				new ColorCostFunction(bfm, COLOR_REGULARIZATION_WEIGHT)
				);
		dense_problem.AddResidualBlock(color_cost, NULL, mean_param.w_color.data());

        for (int j = 0; j < 28588; j+=1) {
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DenseCost, 3, COLOR_COUNT>(
                    new DenseCost(&bfm, &image, j, extrinsics_matrix, intrinsics_matrix, mean_param.w_shape, mean_param.w_expression)
            );
            dense_problem.AddResidualBlock(cost_function, nullptr, &mean_param.w_color.data()[0]);
        }

        ceres::Solver::Summary summary;
        ceres::Solve(options, &dense_problem, &summary);

    }

    cout << "Finish Optimization " << endl;

    return {mean_param, rotation, translation, fov[0]};


}