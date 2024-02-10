#include <iostream>
#include <GL/glew.h>
#include "../include/Renderer.h"


GLFWwindow* initRendering(int width, int height) {
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return nullptr;

    /* Create a windowed mode window and its OpenGL context */
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return nullptr;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glShadeModel(GL_SMOOTH);

    glewInit();

    return window;
}


void terminateRendering() {
    glfwTerminate();
}


cv::Mat renderMesh(GLFWwindow* window, int width, int height, MatrixXi &triangles, MatrixXd &vertices, MatrixXd &colors, std::vector<int> landmarks, bool draw_landmarks) {
    unsigned int fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLuint depthrenderbuffer;
    glGenRenderbuffers(1, &depthrenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

    GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, DrawBuffers);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, width, height);

    glShadeModel(GL_SMOOTH);

    
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < 56572; i++) {

        for (int t = 0; t < 3; t++) {
            // int vertex_index = triangles[i + t * 56572];
            int vertex_index = triangles(i, t);
            double w = vertices(vertex_index, 3);
            glColor3d(colors(vertex_index * 3), colors(vertex_index * 3 + 1), colors(vertex_index * 3 + 2));
            // std::cout << colors(vertex_index * 3, 0) << ", " << colors(vertex_index * 3 + 1, 0) << ", " << colors(vertex_index * 3 + 2, 0) << std::endl;
            glVertex3d(vertices(vertex_index, 0) / w, vertices(vertex_index, 1) / w, vertices(vertex_index, 2) / w);
            // std::cout << vertices(vertex_index, 0) / w << ", " << vertices(vertex_index, 1) / w << ", " << vertices(vertex_index, 2) / w << std::endl;
        }
    }
    glEnd(); 

   if (draw_landmarks) {
        glPointSize(6);
        glBegin(GL_POINTS);
        for (int i = 0; i < landmarks.size(); i++) {
            int vertex_index = landmarks[i];
            double w = vertices(vertex_index, 3);
            glColor3d(1, 0, 0);
            glVertex3d(vertices(vertex_index, 0) / w, vertices(vertex_index, 1) / w, vertices(vertex_index, 2) / w);
            // std::cout << vertices(vertex_index, 0) << ", " << vertices(vertex_index, 1) << ", " << vertices(vertex_index, 2) << std::endl;
        }
        glEnd();
    }      

    unsigned char* gl_texture_bytes = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);
    glReadPixels(0, 0, width, height, 0x80E0, GL_UNSIGNED_BYTE, gl_texture_bytes);
    cv::Mat img(height, width, CV_8UC3, gl_texture_bytes);
    cv::flip(img, img, 0);

    glDeleteFramebuffers(1, &fbo);

    glfwSwapBuffers(window);
    glfwPollEvents();
    
    return img;
}


template <typename T>
static Matrix<T, 3, 4> calculate_transformation_matrix(Matrix<T, 3, 1> translation, Matrix<T, 3, 3> rotation) {
    Matrix<T, 3, 4> transformation;
    transformation.setIdentity();
    transformation.block<3, 3>(0, 0) = rotation;
    transformation.block<3, 1>(0, 3) = translation;

    return transformation;
}


MatrixXd getIntrinsicsMatrix(MatrixXd &intrinsics, double width, double height) {
    // int focal_length = 50;
    // int sensor_width = 36;
    // int sensor_height = 24;

    MatrixXd intrinsics_matrix = MatrixXd::Identity(3, 3);
    intrinsics_matrix(0, 0) = intrinsics(0);
    intrinsics_matrix(1, 1) = intrinsics(1);
    intrinsics_matrix(0, 2) = width / 2;
    intrinsics_matrix(1, 2) = height / 2;

    return intrinsics_matrix;
}


MatrixXd calculate_perspective_matrix(double width, double height, double fov) {
    double zNear = 0.1;
    double zFar = 1000.0;
    double rad = fov * M_PI / 180;
    double tanHalfFovy = tan(rad / 2);
    double aspect_ratio = width / height;

    MatrixXd projection_matrix(4, 4);
    projection_matrix.setConstant(0);

    projection_matrix(0, 0) = 1 / (aspect_ratio * tanHalfFovy);
    projection_matrix(1, 1) = 1 / (tanHalfFovy);
    projection_matrix(2, 2) = -(zFar + zNear) / (zFar - zNear);
    projection_matrix(3, 2) = -1;
    projection_matrix(2, 3) = -(2 * zFar * zNear) / (zFar - zNear);

    return projection_matrix;
}

MatrixXd getExtrinsicsMatrix(MatrixXd &rotation, MatrixXd &translation) {
    MatrixXd extrinsics_matrix;
    extrinsics_matrix.setIdentity(4, 4);
    extrinsics_matrix.block<3, 3>(0, 0) = rotation;
    extrinsics_matrix.block<3, 1>(0, 3) = translation;

    return extrinsics_matrix;
}

MatrixXd calculatePerspectiveProjection(int width, int height, MatrixXd &rotation, MatrixXd &translation, MatrixXd &vertices, double fov) {
    MatrixXd reshaped_vertices = vertices.reshaped<RowMajor>(28588, 3);
    reshaped_vertices.conservativeResize(reshaped_vertices.rows(), reshaped_vertices.cols() + 1);
    reshaped_vertices.col(3).fill(1.0);

    MatrixXd extrinsics_matrix = getExtrinsicsMatrix(rotation, translation);
    std::cout << extrinsics_matrix << std::endl;
    MatrixXd intrinsics_matrix = calculate_perspective_matrix((double) width, (double) height, fov);
    std::cout << intrinsics_matrix << std::endl;

    MatrixXd projected_vertices = intrinsics_matrix * extrinsics_matrix * reshaped_vertices.transpose();

    projected_vertices.transposeInPlace();
    // for (unsigned int i = 0; i < projected_vertices.rows(); i++) {
    //     projected_vertices(i, 0) = projected_vertices(i, 0) / width * 2 - 1;
    //     projected_vertices(i, 1) = projected_vertices(i, 1) / height * 2 - 1;
    // }
    return projected_vertices;
}
