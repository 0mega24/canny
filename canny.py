import cv2
import cupy
import numpy as np
import time

# CUDA kernel to take an image and get the human eye grayscale version
grayscale_kernel_code = """
extern "C" __global__ void grayscaleKernel(unsigned char* input, unsigned char* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // RGB image index
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        gray[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        // The values used for RGB to grayscale conversion are standard values
        // that are used to represent how the human eye sees each of the colors.
    }
}
"""

# CUDA kernel to apply Gaussian blur to an image
gaussian_blur_kernel_code = """
extern "C" __global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height, float* kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image bounds
    if (x < width && y < height) {
        float sum = 0.0f;
        int kernel_radius = kernel_size / 2;
        
        // Iterate over the kernel's radius of effect
        for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
            for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                int idx = iy * width + ix;
                int kernel_idx = (ky + kernel_radius) * kernel_size + (kx + kernel_radius);
                sum += input[idx] * kernel[kernel_idx];
            }
        }
        // return the bounded image from 0 to 255
        output[y * width + x] = (unsigned char)min(max(sum, 0.0f), 255.0f);
    }
}
"""

# CUDA implemantation of the Sobel operator to calculate gradients
# https://en.wikipedia.org/wiki/Sobel_operator
gradient_kernel_code = """
extern "C" __global__ void gradientKernel(unsigned char* gray, float* gradX, float* gradY, float* gradMag, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;

        // Sobel kernels
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
"""

# CUDA kernel to apply non-maximum suppression to the gradient magnitude
nms_kernel_code = """
extern "C" __global__ void nonMaxSuppressionKernel(float* gradMag, float* gradX, float* gradY, unsigned char* edges, int width, int height) {
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
"""

# This is broken I did not have time to make it work fully what it is supposed to do is to make effectively a pair of overlapping negatives that filter down the edges to be more accurate
double_thresholding_kernel_code = """
extern "C" __global__ void doubleThresholdingKernel(unsigned char* input, unsigned char* strong_edges, unsigned char* weak_edges, int width, int height, float low_threshold, float high_threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char value = input[idx];  // Values in range [0, 255]

        if (value >= high_threshold) {
            strong_edges[idx] = 255;  // Strong edge
            weak_edges[idx] = 0;      // No weak edge
        } else if (value >= low_threshold) {
            strong_edges[idx] = 0;    // No strong edge
            weak_edges[idx] = 255;    // Weak edge
        } else {
            strong_edges[idx] = 0;    // No strong edge
            weak_edges[idx] = 0;      // No weak edge
        }
    }
}
"""

# Same thing here it is not working as intended, I am not sure why because I can not tell if it is the thresholding causing it to be non functional
# It is supposed to look at a pixel neighbor and if it is a strong edge it will connect the weak edge to it to have a more accurate edge
hysteresis_kernel_code = """
extern "C" __global__ void edgeTrackingByHysteresisKernel(unsigned char* strong_edges, unsigned char* weak_edges, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        if (strong_edges[idx] == 255) {
            output[idx] = 255;  // Strong edges remain
        } else if (weak_edges[idx] == 255) {
            // Check neighbors for connection to strong edge
            bool connected_to_strong_edge = false;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        if (strong_edges[ny * width + nx] == 255) {
                            connected_to_strong_edge = true;
                            break;
                        }
                    }
                }
                if (connected_to_strong_edge) break;
            }
            if (connected_to_strong_edge) {
                output[idx] = 255;  // Connect weak edge to strong edge
            } else {
                output[idx] = 0;    // Discard weak edge if not connected
            }
        }
    }
}
"""


def grayscale_conversion(input_image, device_id):
    """Convert the image to grayscale using CuPy and kernel."""
    height, width, _ = input_image.shape
    d_input = cupy.asarray(input_image, dtype=cupy.uint8)
    d_output = cupy.zeros(
        (height, width), dtype=cupy.uint8
    )

    start_time = time.time()
    with cupy.cuda.Device(device_id):
        grayscale_kernel = cupy.RawKernel(grayscale_kernel_code, "grayscaleKernel")
        block_size = (16, 16, 1)
        grid_size = (
            int(np.ceil(width / block_size[0])),
            int(np.ceil(height / block_size[1])),
        )
        grayscale_kernel(
            grid_size,
            block_size,
            (d_input, d_output, np.int32(width), np.int32(height)),
        )
    end_time = time.time()
    print(f"Grayscale conversion time (GPU): {end_time - start_time:.4f} seconds")

    return d_output


def gaussian_blur(input_image, device_id, kernel_size=5):
    height, width = input_image.shape
    d_input = cupy.asarray(input_image, dtype=cupy.uint8)
    d_output = cupy.zeros_like(d_input, dtype=cupy.uint8)

    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()

    d_kernel = cupy.asarray(kernel, dtype=cupy.float32)

    start_time = time.time()
    with cupy.cuda.Device(device_id):
        gaussian_blur_kernel = cupy.RawKernel(
            gaussian_blur_kernel_code, "gaussianBlurKernel"
        )
        block_size = (16, 16, 1)
        grid_size = (
            int(np.ceil(width / block_size[0])),
            int(np.ceil(height / block_size[1])),
        )
        gaussian_blur_kernel(
            grid_size,
            block_size,
            (
                d_input,
                d_output,
                np.int32(width),
                np.int32(height),
                d_kernel,
                np.int32(kernel_size),
            ),
        )

    end_time = time.time()
    print(f"Gaussian blur time (GPU): {end_time - start_time:.4f} seconds")

    return d_output


def gradient_calculation(gray_image, device_id):
    height, width = gray_image.shape
    d_grad_x = cupy.zeros_like(gray_image, dtype=cupy.float32)
    d_grad_y = cupy.zeros_like(gray_image, dtype=cupy.float32)
    d_grad_mag = cupy.zeros_like(gray_image, dtype=cupy.float32)

    start_time = time.time()
    with cupy.cuda.Device(device_id):
        gradient_kernel = cupy.RawKernel(gradient_kernel_code, "gradientKernel")
        block_size = (16, 16, 1)
        grid_size = (
            int(np.ceil(width / block_size[0])),
            int(np.ceil(height / block_size[1])),
        )
        gradient_kernel(
            grid_size,
            block_size,
            (
                gray_image,
                d_grad_x,
                d_grad_y,
                d_grad_mag,
                np.int32(width),
                np.int32(height),
            ),
        )
    end_time = time.time()
    print(f"Gradient calculation time (GPU): {end_time - start_time:.4f} seconds")

    return d_grad_x, d_grad_y, d_grad_mag


def non_maximum_suppression(grad_mag, grad_x, grad_y, device_id):
    """Apply non-maximum suppression using CuPy kernel."""
    height, width = grad_mag.shape
    d_edges = cupy.zeros_like(grad_mag, dtype=cupy.uint8)

    start_time = time.time()
    with cupy.cuda.Device(device_id):
        nms_kernel = cupy.RawKernel(nms_kernel_code, "nonMaxSuppressionKernel")
        block_size = (16, 16, 1)
        grid_size = (
            int(np.ceil(width / block_size[0])),
            int(np.ceil(height / block_size[1])),
        )
        nms_kernel(
            (grid_size),
            (block_size),
            (grad_mag, grad_x, grad_y, d_edges, np.int32(width), np.int32(height)),
        )
    end_time = time.time()
    print(f"Non-Maximum Suppression time (GPU): {end_time - start_time:.4f} seconds")

    return d_edges


def double_thresholding(
    input_image, low_threshold=0.1, high_threshold=0.3, device_id=0
):
    """Apply double thresholding to classify edges into strong, weak, and non-edges."""
    height, width = input_image.shape
    d_input = cupy.asarray(input_image, dtype=cupy.float32)
    d_strong_edges = cupy.zeros_like(input_image, dtype=cupy.uint8)
    d_weak_edges = cupy.zeros_like(input_image, dtype=cupy.uint8)

    start_time = time.time()
    with cupy.cuda.Device(device_id):
        double_thresholding_kernel = cupy.RawKernel(
            double_thresholding_kernel_code, "doubleThresholdingKernel"
        )
        block_size = (16, 16, 1)
        grid_size = (
            int(np.ceil(width / block_size[0])),
            int(np.ceil(height / block_size[1])),
        )
        double_thresholding_kernel(
            grid_size,
            block_size,
            (
                d_input,
                d_strong_edges,
                d_weak_edges,
                np.int32(width),
                np.int32(height),
                low_threshold,
                high_threshold,
            ),
        )
    end_time = time.time()
    print(f"Double Thresholding time (GPU): {end_time - start_time:.4f} seconds")

    return d_strong_edges, d_weak_edges


def edge_tracking_by_hysteresis(strong_edges, weak_edges, device_id=0):
    """Track weak edges connected to strong edges using hysteresis."""
    height, width = strong_edges.shape
    d_output = cupy.zeros_like(strong_edges, dtype=cupy.uint8)

    start_time = time.time()
    with cupy.cuda.Device(device_id):
        edge_tracking_by_hysteresis_kernel = cupy.RawKernel(
            hysteresis_kernel_code, "edgeTrackingByHysteresisKernel"
        )
        block_size = (16, 16, 1)
        grid_size = (
            int(np.ceil(width / block_size[0])),
            int(np.ceil(height / block_size[1])),
        )
        edge_tracking_by_hysteresis_kernel(
            grid_size,
            block_size,
            (strong_edges, weak_edges, d_output, np.int32(width), np.int32(height)),
        )
    end_time = time.time()
    print(
        f"Edge Tracking by Hysteresis time (GPU): {end_time - start_time:.4f} seconds"
    )

    return d_output


def canny_edge_detection(image_path, output_path, device_id=0):
    input_image = cv2.imread(image_path)
    if input_image is None:
        print("Error: Image could not be loaded.")
        return

    print("Step 1: Grayscale conversion...")
    gray_image = grayscale_conversion(input_image, device_id)
    cv2.imwrite("step1_gray.png", cupy.asnumpy(gray_image))

    print("Step 2: Gaussian blur...")
    blurred_image = gaussian_blur(gray_image, device_id)
    cv2.imwrite("step2_blurred.png", cupy.asnumpy(blurred_image))

    print("Step 3: Gradient calculation...")
    grad_x, grad_y, grad_mag = gradient_calculation(gray_image, device_id)
    cv2.imwrite("step3_grad_x.png", cupy.asnumpy(grad_x))
    cv2.imwrite("step3_grad_y.png", cupy.asnumpy(grad_y))
    cv2.imwrite("step3_grad_mag.png", cupy.asnumpy(grad_mag))

    print("Step 4: Applying Non-Maximum Suppression...")
    nms_image = non_maximum_suppression(grad_mag, grad_x, grad_y, device_id)
    cv2.imwrite("step4_nms.png", cupy.asnumpy(nms_image))
    cv2.imwrite(output_path, cupy.asnumpy(nms_image))


canny_edge_detection("step0_original.png", "output.png", device_id=0)
