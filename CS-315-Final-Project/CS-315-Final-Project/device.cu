#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "RGB.h"
#include <math.h>
#include <stdio.h>
#include <iostream>

/**
* Helper function to calculate the greyscale value based on R, G, and B
*/
__device__ int greyscale(BYTE red, BYTE green, BYTE blue)
{
	int grey = 0.3 * red + 0.59 * green + 0 * 11 * blue; // calculate grey scale
	return min(grey, 255);
}

/**
* Kernel for executing on GPY
*/
__global__ void greyscaleKernel(RGB* d_pixels, int height, int width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (y >= height || y >= width)
		return;

	int index = y * width + x;

	int grey = greyscale(d_pixels[index].red, d_pixels[index].green, d_pixels[index].blue); // calculate grey scale

	d_pixels[index].red = grey;
	d_pixels[index].green = grey;
	d_pixels[index].blue = grey;

}

// Kernel to blur an image on the GPU
__global__ void blurKernel(RGB* d_pixels, int height, int width)
{
	// determine the current pixel
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// if the pixel is still inside the dimensions of the image
	if (col < width && row < height) {
		// temp values to hold the new rgb values for our blurred pixel
		int redVal = 0;
		int greenVal = 0;
		int blueVal = 0;
		// a count of how many pixels were used to determine the blurred values
		int pixels = 0;

		// change the blurRow and blurCol for a different stencil to make the image more or less blurred
		for (int blurRow = -5; blurRow <= 5; ++blurRow) {
			for (int blurCol = -5; blurCol <= 5; ++blurCol) {
				int curRow = row + blurRow;
				int curCol = col + blurCol;
				// check to make sure that this is an existing pixel that we want to use for blurring
				if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
					// add the rgb values of neighboring pixels to the temp values
					redVal += d_pixels[curRow * width + curCol].red;
					greenVal += d_pixels[curRow * width + curCol].green;
					blueVal += d_pixels[curRow * width + curCol].blue;
					// increment pixels to show how many pixels we looked at
					pixels++;
				}
			}
		}
		// average the rgb values by dividing by the count of pixels
		d_pixels[row * width + col].red = (unsigned char)(redVal / pixels);
		d_pixels[row * width + col].green = (unsigned char)(greenVal / pixels);
		d_pixels[row * width + col].blue = (unsigned char)(blueVal / pixels);
	}
}

// Sobel Filter
// |Gx(x,y)| = -P(x-1,y-1) + -2 *P(x-1,y) + -P(x-1,y+1) + P(x+1,y-1) + 2 * P(x + 1, y) + P(x + 1, y + 1)
// |Gy(x,y)| = P(x-1,y-1) + 2*P(x,y-1) + P(x+1,y-1) + -P(x-1,y+1) + –2 * P(x, y + 1) - P(x + 1, y + 1)
__global__ void edgeDetectionKernel(RGB* d_pixels, RGB* d_result, int height, int width)
{
	// determine the current pixel
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float reddx, reddy;
	float greendx, greendy;
	float bluedx, bluedy;

	if (col > 0 && row > 0 && col < width - 1 && row < height - 1) {
		//red
		reddx = (-1 * d_pixels[(row - 1) * width + (col - 1)].red) + (-2 * d_pixels[row * width + (col - 1)].red) + (-1 * d_pixels[(row + 1) * width + (col - 1)].red) + (d_pixels[(row - 1) * width + (col + 1)].red) + (2 * d_pixels[row * width + (col + 1)].red) + (d_pixels[(row + 1) * width + (col + 1)].red);
		reddy = (d_pixels[(row - 1) * width + (col - 1)].red) + (2 * d_pixels[(row - 1) * width + col].red) + (d_pixels[(row - 1) * width + (col + 1)].red) + (-1 * d_pixels[(row + 1) * width + (col - 1)].red) + (-2 * d_pixels[(row + 1) * width + col].red) + (-1 * d_pixels[(row + 1) * width + (col + 1)].red);

		d_result[row * width + col].red = (unsigned char)(sqrt((reddx * reddx) + (reddy * reddy)));

		//green
		greendx = (-1 * d_pixels[(row - 1) * width + (col - 1)].green) + (-2 * d_pixels[row * width + (col - 1)].green) + (-1 * d_pixels[(row + 1) * width + (col - 1)].green) + (d_pixels[(row - 1) * width + (col + 1)].green) + (2 * d_pixels[row * width + (col + 1)].green) + (d_pixels[(row + 1) * width + (col + 1)].green);
		greendy = (d_pixels[(row - 1) * width + (col - 1)].green) + (2 * d_pixels[(row - 1) * width + col].green) + (d_pixels[(row - 1) * width + (col + 1)].green) + (-1 * d_pixels[(row + 1) * width + (col - 1)].green) + (-2 * d_pixels[(row + 1) * width + col].green) + (-1 * d_pixels[(row + 1) * width + (col + 1)].green);

		d_result[row * width + col].green = (unsigned char)(sqrt((greendx * greendx) + (greendy * greendy)));

		//blue
		bluedx = (-1 * d_pixels[(row - 1) * width + (col - 1)].blue) + (-2 * d_pixels[row * width + (col - 1)].blue) + (-1 * d_pixels[(row + 1) * width + (col - 1)].blue) + (d_pixels[(row - 1) * width + (col + 1)].blue) + (2 * d_pixels[row * width + (col + 1)].blue) + (d_pixels[(row + 1) * width + (col + 1)].blue);
		bluedy = (d_pixels[(row - 1) * width + (col - 1)].blue) + (2 * d_pixels[(row - 1) * width + col].blue) + (d_pixels[(row - 1) * width + (col + 1)].blue) + (-1 * d_pixels[(row + 1) * width + (col - 1)].blue) + (-2 * d_pixels[(row + 1) * width + col].blue) + (-1 * d_pixels[(row + 1) * width + (col + 1)].blue);

		d_result[row * width + col].blue = (unsigned char)(sqrt((bluedx * bluedx) + (bluedy * bluedy)));

	}
}

// -1  -1  -1
// -1   8  -1
// -1  -1  -1
// Laplacian filter
__global__ void laplacianKernel(RGB* d_pixels, RGB* d_result, int height, int width)
{
	// determine the current pixel
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float red, green, blue;

	// apply the filter shown above to red, green, and blue for the current pixel
	if (col > 0 && row > 0 && col < width - 1 && row < height - 1) {
		red = (-1 * d_pixels[(row - 1) * width + (col - 1)].red) + (-1 * d_pixels[(row - 1) * width + col].red) + (-1 * d_pixels[(row - 1) * width + (col + 1)].red) + (-1 * d_pixels[row * width + (col - 1)].red) + (8 * d_pixels[row * width + col].red) + (-1 * d_pixels[row * width + (col + 1)].red) + (-1 * d_pixels[(row + 1) * width + (col - 1)].red) + (-1 * d_pixels[(row + 1) * width + col].red) + (-1 * d_pixels[(row + 1) * width + (col - 1)].red);
		green = (-1 * d_pixels[(row - 1) * width + (col - 1)].green) + (-1 * d_pixels[(row - 1) * width + col].green) + (-1 * d_pixels[(row - 1) * width + (col + 1)].green) + (-1 * d_pixels[row * width + (col - 1)].green) + (8 * d_pixels[row * width + col].green) + (-1 * d_pixels[row * width + (col + 1)].green) + (-1 * d_pixels[(row + 1) * width + (col - 1)].green) + (-1 * d_pixels[(row + 1) * width + col].green) + (-1 * d_pixels[(row + 1) * width + (col - 1)].green);
		blue = (-1 * d_pixels[(row - 1) * width + (col - 1)].blue) + (-1 * d_pixels[(row - 1) * width + col].blue) + (-1 * d_pixels[(row - 1) * width + (col + 1)].blue) + (-1 * d_pixels[row * width + (col - 1)].blue) + (8 * d_pixels[row * width + col].blue) + (-1 * d_pixels[row * width + (col + 1)].blue) + (-1 * d_pixels[(row + 1) * width + (col - 1)].blue) + (-1 * d_pixels[(row + 1) * width + col].blue) + (-1 * d_pixels[(row + 1) * width + (col - 1)].blue);

		d_result[row * width + col].red = red;
		d_result[row * width + col].green = green;
		d_result[row * width + col].blue = blue;
	}
}

// increase the red, green, or blue contrast without going out of bounds (0-255)
__global__ void contrastKernel(RGB* d_pixels, int height, int width, int rincrease, int gincrease, int bincrease)
{
	// determine the current pixel
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < width && row < height) {
		int index = row * width + col;
		//red
		if (d_pixels[index].red + rincrease < 256 && d_pixels[index].red + rincrease > -1) {
			d_pixels[index].red += rincrease;
		}
		else if (d_pixels[index].red + rincrease > 255) {
			d_pixels[index].red = 255;
		}
		else if (d_pixels[index].red + rincrease < 0) {
			d_pixels[index].red = 0;
		}
		//green
		if (d_pixels[index].green + gincrease < 256 && d_pixels[index].green + gincrease > -1) {
			d_pixels[index].green += gincrease;
		}
		else if (d_pixels[index].green + gincrease > 255) {
			d_pixels[index].green = 255;
		}
		else if (d_pixels[index].green + gincrease < 0) {
			d_pixels[index].green = 0;
		}
		// blue
		if (d_pixels[index].blue + bincrease < 256 && d_pixels[index].blue + bincrease > -1) {
			d_pixels[index].blue += bincrease;
		}
		else if (d_pixels[index].blue + bincrease > 255) {
			d_pixels[index].blue = 255;
		}
		else if (d_pixels[index].blue + bincrease < 0) {
			d_pixels[index].blue = 0;
		}
	}
}

// increase the brightness of an image without going out of bounds (0-255)
__global__ void brightnessKernel(RGB* d_pixels, int height, int width, int bright)
{
	// determine the current pixel
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < width && row < height) {
		int index = row * width + col;
		//red
		if (d_pixels[index].red * bright < 256 && d_pixels[index].red * bright > -1) {
			d_pixels[index].red *= bright;
		}
		else if (d_pixels[index].red * bright > 255) {
			d_pixels[index].red = 255;
		}
		else if (d_pixels[index].red * bright < 0) {
			d_pixels[index].red = 0;
		}
		//green
		if (d_pixels[index].green * bright < 256 && d_pixels[index].green * bright > -1) {
			d_pixels[index].green *= bright;
		}
		else if (d_pixels[index].green * bright > 255) {
			d_pixels[index].green = 255;
		}
		else if (d_pixels[index].green * bright < 0) {
			d_pixels[index].green = 0;
		}
		//blue
		if (d_pixels[index].blue * bright < 256 && d_pixels[index].blue * bright > -1) {
			d_pixels[index].blue *= bright;
		}
		else if (d_pixels[index].blue * bright > 255) {
			d_pixels[index].blue = 255;
		}
		else if (d_pixels[index].blue * bright < 0) {
			d_pixels[index].blue = 0;
		}
	}
}

//finds the gradient and edge direction of the image (used for canny)
__global__ void GradiantStrength(RGB* d_pixels, int* edgeDir, int* gradiant, int height, int width) {
	int col = blockIdx.x * blockDim.x + threadIdx.x; // width
	int row = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (row >= height || col >= width) {
		return;
	}

	int index = row * width + col;
	float Gx = 0;
	float Gy = 0;
	int newAngle = 2000;

	if (col < width && row < height) {
		if (col > 0 && row > 0 && col < width && row < height) {
			Gx = (-1 * d_pixels[(row - 1) * width + (col - 1)].red) + (-2 * d_pixels[row * width + (col - 1)].red) + (-1 * d_pixels[(row + 1) * width + (col - 1)].red) + (d_pixels[(row - 1) * width + (col + 1)].red) + (2 * d_pixels[row * width + (col + 1)].red) + (d_pixels[(row + 1) * width + (col + 1)].red);
			Gy = (d_pixels[(row - 1) * width + (col - 1)].red) + (2 * d_pixels[(row - 1) * width + col].red) + (d_pixels[(row - 1) * width + (col + 1)].red) + (-1 * d_pixels[(row + 1) * width + (col - 1)].red) + (-2 * d_pixels[(row + 1) * width + col].red) + (-1 * d_pixels[(row + 1) * width + (col + 1)].red);

		}
	}

	gradiant[index] = sqrt((Gx * Gx) + (Gy * Gy));	// Calculate gradient strength						
	double thisAngle = (atan2(Gx, Gy) / 3.14159) * 180.0;		// Calculate actual direction of edge
	//std::cout << "this Angle is: " << thisAngle << " gradiant: " << sqrt((Gx * Gx) + (Gy * Gy)) << std::endl;
	//printf("this angle is: %d, gradient: %d \n", thisAngle, sqrt((Gx * Gx) + (Gy * Gy)));
	/* Convert actual edge direction to approximate value */
	if (((thisAngle < 22.5) && (thisAngle > -22.5)) || (thisAngle > 157.5) || (thisAngle < -157.5)) {
		newAngle = 0;
	}
	if (((thisAngle > 22.5) && (thisAngle < 67.5)) || ((thisAngle < -112.5) && (thisAngle > -157.5))) {
		newAngle = 45;
	}
	if (((thisAngle > 67.5) && (thisAngle < 112.5)) || ((thisAngle < -67.5) && (thisAngle > -112.5))) {
		newAngle = 90;
	}
	if (((thisAngle > 112.5) && (thisAngle < 157.5)) || ((thisAngle < -22.5) && (thisAngle > -67.5))) {
		newAngle = 135;
	}

	edgeDir[index] = newAngle;	// Store the approximate edge direction of each pixel in one array
}


/**
*	Helper function to calculate the number of blocks on an axis based on the total grid size and number of threads in that axis
*/
__host__ int calcBlockDim(int total, int num_threads)
{
	int r = total / num_threads;
	if (total % num_threads != 0) // add one to cover all the threads per block
		++r;
	return r;
}

/**
*	Host function for launching greyscale kernel
*/
__host__ void d_convert_greyscale(RGB* pixel, int height, int width)
{
	RGB* d_pixel;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	greyscaleKernel << <grid, block >> > (d_pixel, height, width);

	cudaMemcpy(pixel, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}

__host__ void d_convert_blur(RGB* pixel, int height, int width)
{
	RGB* d_pixel;

	// allocate and copy memory to the GPU
	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	// determine the grid and block for the GPU
	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	// blur the image
	blurKernel << <grid, block >> > (d_pixel, height, width);

	// copy the result back to the CPU
	cudaMemcpy(pixel, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}

__host__ void d_edge_detection(RGB* pixel, int height, int width)
{
	RGB* d_pixel;
	RGB* d_result;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);
	cudaMalloc(&d_result, height * width * sizeof(RGB));
	cudaMemcpy(d_result, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	edgeDetectionKernel << <grid, block >> > (d_pixel, d_result, height, width);

	cudaMemcpy(pixel, d_result, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}

__host__ void d_laplacian(RGB* pixel, int height, int width)
{
	RGB* d_pixel;
	RGB* d_result;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);
	cudaMalloc(&d_result, height * width * sizeof(RGB));
	cudaMemcpy(d_result, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	laplacianKernel << <grid, block >> > (d_pixel, d_result, height, width);

	cudaMemcpy(pixel, d_result, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}

__host__ void d_contrast(RGB* pixel, int height, int width, int rincrease, int gincrease, int bincrease)
{
	RGB* d_pixel;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(width, block.y);

	contrastKernel << <grid, block >> > (d_pixel, height, width, rincrease, gincrease, bincrease);

	cudaMemcpy(pixel, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}

__host__ void d_brightness(RGB* pixel, int height, int width, int bright)
{
	RGB* d_pixel;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(width, block.y);

	brightnessKernel << <grid, block >> > (d_pixel, height, width, bright);

	cudaMemcpy(pixel, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}

__host__ void gradiantLauncher(RGB* pixels, int* edgeDir, int* gradiant, int height, int width) {
	RGB* d_pixel;
	int* d_edgeDir;
	int* d_gradiant;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixels, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	cudaMalloc(&d_edgeDir, height * width * sizeof(int));
	cudaMemcpy(d_edgeDir, edgeDir, height * width * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&d_gradiant, height * width * sizeof(int));
	cudaMemcpy(d_gradiant, gradiant, height * width * sizeof(int), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	GradiantStrength << <grid, block >> > (d_pixel, d_edgeDir, d_gradiant, height, width);
	cudaMemcpy(pixels, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
	cudaMemcpy(edgeDir, d_edgeDir, height * width * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(gradiant, d_gradiant, height * width * sizeof(int), cudaMemcpyDeviceToHost);
}