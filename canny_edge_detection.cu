#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

__global__ void grayscaleKernel(unsigned char* input, unsigned char* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        gray[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

__global__ void gradientKernel(unsigned char* gray, float* gradX, float* gradY, float* gradMag, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;

        float Gx = (-1 * gray[idx - width - 1] + 1 * gray[idx - width + 1]) +
                   (-2 * gray[idx - 1]         + 2 * gray[idx + 1]) +
                   (-1 * gray[idx + width - 1] + 1 * gray[idx + width + 1]);

        float Gy = (-1 * gray[idx - width - 1] - 2 * gray[idx - width] - 1 * gray[idx - width + 1]) +
                   (1 * gray[idx + width - 1]  + 2 * gray[idx + width] + 1 * gray[idx + width + 1]);

        gradX[idx] = Gx;
        gradY[idx] = Gy;
        gradMag[idx] = sqrtf(Gx * Gx + Gy * Gy);
    }
}

__global__ void nonMaxSuppressionKernel(float* gradMag, float* gradX, float* gradY, unsigned char* edges, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;

        float angle = atan2f(gradY[idx], gradX[idx]) * 180.0f / 3.14159265f;
        angle = angle < 0 ? angle + 180 : angle;

        float q = 255, r = 255;

        if ((angle > 0 && angle <= 22.5) || (angle > 157.5 && angle <= 180)) {
            q = gradMag[idx + 1];
            r = gradMag[idx - 1];
        } else if (angle > 22.5 && angle <= 67.5) {
            q = gradMag[idx + width + 1];
            r = gradMag[idx - width - 1];
        } else if (angle > 67.5 && angle <= 112.5) {
            q = gradMag[idx + width];
            r = gradMag[idx - width];
        } else if (angle > 112.5 && angle <= 157.5) {
            q = gradMag[idx + width - 1];
            r = gradMag[idx - width + 1];
        }

        edges[idx] = (gradMag[idx] >= q && gradMag[idx] >= r) ? (unsigned char)gradMag[idx] : 0;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./canny_edge <input_image> <output_image>" << endl;
        return -1;
    }

    Mat inputImage = imread(argv[1], IMREAD_COLOR);
    if (inputImage.empty()) {
        cout << "Error: Could not load image!" << endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    unsigned char* h_input = inputImage.data;
    unsigned char* h_gray = new unsigned char[width * height];
    unsigned char* h_edges = new unsigned char[width * height];

    float *h_gradX = new float[width * height];
    float *h_gradY = new float[width * height];
    float *h_gradMag = new float[width * height];

    unsigned char *d_input, *d_gray, *d_edges;
    float *d_gradX, *d_gradY, *d_gradMag;

    cudaMalloc(&d_input, width * height * 3 * sizeof(unsigned char));
    cudaMalloc(&d_gray, width * height * sizeof(unsigned char));
    cudaMalloc(&d_edges, width * height * sizeof(unsigned char));
    cudaMalloc(&d_gradX, width * height * sizeof(float));
    cudaMalloc(&d_gradY, width * height * sizeof(float));
    cudaMalloc(&d_gradMag, width * height * sizeof(float));

    cudaMemcpy(d_input, h_input, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_gray, width, height);
    cudaMemcpy(h_gray, d_gray, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    Mat grayImage(height, width, CV_8UC1, h_gray);
    imwrite("gray.jpg", grayImage);

    gradientKernel<<<gridSize, blockSize>>>(d_gray, d_gradX, d_gradY, d_gradMag, width, height);
    cudaMemcpy(h_gradX, d_gradX, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    Mat gradXImage(height, width, CV_32FC1, h_gradX);
    imwrite("gradX.jpg", gradXImage);

    nonMaxSuppressionKernel<<<gridSize, blockSize>>>(d_gradMag, d_gradX, d_gradY, d_edges, width, height);

    cudaMemcpy(h_edges, d_edges, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    Mat edgesImage(height, width, CV_8UC1, h_edges);
    imwrite(argv[2], edgesImage);

    delete[] h_gray;
    delete[] h_edges;
    delete[] h_gradX;
    delete[] h_gradY;
    delete[] h_gradMag;

    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_edges);
    cudaFree(d_gradX);
    cudaFree(d_gradY);
    cudaFree(d_gradMag);

    cout << "Canny edge detection completed successfully." << endl;
    return 0;
}

