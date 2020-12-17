// Mason Caird, Tyler Gamlem, Jeremy Knight
// CS 315
// Edge Detection and other Image Manupulation

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

#include "bmp.h"

#define NUM_IMAGES	4
#define EDGE_FILTERS 7

void d_convert_greyscale(RGB* pixels, int height, int width);

void d_convert_blur(RGB* pixels, int height, int width);

void d_edge_detection(RGB* pixels, int height, int width);

void d_laplacian(RGB* pixels, int height, int width);

void d_contrast(RGB* pixels, int height, int width, int rincrease, int gincrease, int bincrease);

void d_brightness(RGB* pixels, int height, int width, int bright);

void gradiantLauncher(RGB* pixels, int* edgeDir, int* gradiant, int height, int width);

/**
*  Computes the average of the red, green, and blue components of an image
*
* @param pixels  The array of RGB (Red, Green, Blue) components of each pixel in the image
* @param height  The height of the image
* @param width   The width of the image
*/
void compute_component_average(RGB* pixels, int height, int width)
{
	double total_red = 0, total_green = 0, total_blue = 0;

	for (int y = 0; y < height; ++y) { // for each row in image
		for (int x = 0; x < width; ++x) { // for each pixel in the row
			int index = y * width + x; // compute index position of (y,x) coordinate
			total_red += pixels[index].red; // add the red value at this pixel to total
			total_green += pixels[index].green; // add the green value at this pixel to total
			total_blue += pixels[index].blue; // add the blue value at this pixel to total
		}
	}

	cout << "Red average: " << total_red / (height * width) << endl;
	cout << "Green average: " << total_green / (height * width) << endl;
	cout << "Blue average: " << total_blue / (height * width) << endl;

}


void findEdge(RGB* pixels, int* edgeDir, int* gradient, int rowShift, int colShift, int row, int col, int dir, int lowerThreshold, int width, int height) {
	int newRow = 0;
	int newCol = 0;
	unsigned long i;
	bool edgeEnd = false;

	int index = col * width + row;

	/* Find the row and column values for the next possible pixel on the edge */
	if (colShift < 0) {
		if (col > 0)
			newCol = col + colShift;
		else
			edgeEnd = true;
	}
	else if (col < width - 1) {
		newCol = col + colShift;
	}
	else
		edgeEnd = true;		// If the next pixel would be off image, don't do the while loop
	if (rowShift < 0) {
		if (row > 0)
			newRow = row + rowShift;
		else
			edgeEnd = true;
	}
	else if (row < height - 1) {
		newRow = row + rowShift;
	}
	else
		edgeEnd = true;

	/* Determine edge directions and gradient strengths */
	int newIndex = newCol * width + newRow;
	while ((edgeDir[newIndex] == dir) && !edgeEnd && (gradient[newIndex] > lowerThreshold)) {
		/* Set the new pixel as white to show it is an edge */
		pixels[newIndex].red = 255;
		pixels[newIndex].blue = 255;
		pixels[newIndex].green = 255;

		if (colShift < 0) {
			if (newCol > 0)
				newCol = newCol + colShift;
			else
				edgeEnd = true;
		}
		else if (newCol < width - 1) {
			newCol = newCol + colShift;
		}
		else
			edgeEnd = true;
		if (rowShift < 0) {
			if (newRow > 0)
				newRow = newRow + rowShift;
			else
				edgeEnd = true;
		}
		else if (newRow < height - 1) {
			newRow = newRow + rowShift;
		}
		else
			edgeEnd = true;

		newIndex = newCol * width + newRow;
	}
}

void traceEdge(RGB* pixels, int* edgeDir, int* gradient, int height, int width) {
	/* Trace along all the edges in the image */
	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			bool edgeEnd = false;
			int index = r * width + c;
			//std::cout << gradient[index] << std::endl;
			if (gradient[index] > 60) {		//this went over // Check to see if current pixel has a high enough gradient strength to be part of an edge
				/* Switch based on current pixel's edge direction */
				switch (edgeDir[index]) {
				case 0:
					findEdge(pixels, edgeDir, gradient, 0, 1, r, c, 0, 30, width, height);
					break;
				case 45:
					findEdge(pixels, edgeDir, gradient, 1, 1, r, c, 45, 30, width, height);
					break;
				case 90:
					findEdge(pixels, edgeDir, gradient, 1, 0, r, c, 90, 30, width, height);
					break;
				case 135:
					findEdge(pixels, edgeDir, gradient, 1, -1, r, c, 135, 30, width, height);
					break;
				default:
					//i = (unsigned long)(row * 3 * W + 3 * col);
					pixels[index].red = 0;
					pixels[index].blue = 0;
					pixels[index].green = 0;
					break;
				}
			}
			else {
				pixels[index].red = 0;
				pixels[index].blue = 0;
				pixels[index].green = 0;
			}
		}
	}

	/* Suppress any pixels not changed by the edge tracing */
	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			// Recall each pixel is composed of 3 bytes
			//i = (unsigned long)(row * 3 * width + 3 * col);
			int index = r * width + c;
			// If a pixel's grayValue is not black or white make it black
			if (((pixels[index].red != 255) && (pixels[index].green != 0)) || ((pixels[index].blue != 255))) { // && (*(m_destinationBmp + i + 1) != 0)) || ((*(m_destinationBmp + i + 2) != 255))) {
				//&& (*(m_destinationBmp + i + 2) != 0)))
				pixels[index].red = 0;
				pixels[index].blue = 0;
				pixels[index].green = 0;
			}
		}
	}
}
/*
void CannyShit(RGB* pixels, int* edgeDir, int* gradient, int height, int width) {
	d_convert_greyscale(pixels, height, width);
	gradiantLauncher(pixels, edgeDir, gradient, height, width);
	traceEdge(pixels, edgeDir, gradient, height, width);
}
*/
int main()
{

	cout << "\n********* Filter Calculation Program *********" << endl;
	cout << "Written by Tyler Gamlem, Jeremy Knight, and Mason Caird" << endl << endl;

	while (true) {

		string image_archive[NUM_IMAGES] = { "lena.bmp", "marbles.bmp", "sierra_02.bmp", "tiger.bmp" };
		string edge_array[EDGE_FILTERS] = { "Grey Scale", "Sobel", "lapacain", "constrast", "brightness", "blur", "Canny" };

		cout << "Select an image: \n";
		for (int i = 0; i < NUM_IMAGES; ++i)
			cout << i << ": " << image_archive[i] << endl;
		cout << NUM_IMAGES << ": exit\n";

		int choice;
		int filter;
		do {
			cout << "Please choice: ";
			cin >> choice;
			if (choice == NUM_IMAGES) {
				cout << "Goodbye!\n";
				exit(0);
			}
		} while (choice < 0 || choice > NUM_IMAGES);

		// Filter Selection

		if (choice >= 0) {

			cout << "\nSelect a filter: \n";

			for (int i = 0; i < EDGE_FILTERS; ++i)
				cout << i << ": " << edge_array[i] << endl;

			cout << EDGE_FILTERS << ": Back\n";

			do {
				cout << "Filter Choice: ";
				cout << endl;
				cin >> filter;
				if (filter == EDGE_FILTERS) {
					main();
				}
			} while (filter < 0 || filter > EDGE_FILTERS);
		}

		BitMap image(image_archive[choice]); // Load the bitmap image into the BitMap object

		// Display some of the image's properties
		cout << "Image properties\n";
		cout << setw(15) << left << "Dimensions: " << image.getHeight() << " by " << image.getWidth() << endl;
		cout << setw(15) << left << "Size: " << image.getImageSize() << " bytes\n";
		cout << setw(15) << left << "Bit encoding: " << image.getBitCount() << " bits\n\n";

		int* gradiant = (int*)malloc((image.getHeight() * image.getWidth()) * 300);
		int* edgeDir = (int*)malloc((image.getHeight() * image.getWidth()) * 300);

		RGB* pixels = image.getRGBImageArray(); // get the image array of RGB (Red, Green, and Blue) components

		auto start = std::chrono::high_resolution_clock::now(); // start timer
		auto stop = std::chrono::high_resolution_clock::now(); // stop timer
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
		int value = 0; // user Input
		string rgb_types[3] = { "Red", "Green", "Blue" }; // RGB Types

		switch (filter) {

			// greyscale (pixels, height, width)
		case 0:

			start = std::chrono::high_resolution_clock::now(); // start timer
			d_convert_greyscale(pixels, image.getHeight(), image.getWidth()); // kernel
			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("greyscale.bmp"); // Save bmp
			system("sh executeImage.sh");
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Greyscale Time: " << duration.count() << " milliseconds" << endl;
			break;

			// edgedetection (pixels, height, width)
		case 1:

			start = std::chrono::high_resolution_clock::now(); // start timer
			d_edge_detection(pixels, image.getHeight(), image.getWidth()); // kernel
			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("Sobel.bmp"); // Save bmp
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Sobel Time: " << duration.count() << " milliseconds" << endl;
			break;

			// laplacain (pixels, height, width)
		case 2:

			start = std::chrono::high_resolution_clock::now(); // start timer
			d_laplacian(pixels, image.getHeight(), image.getWidth()); // kernel
			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("Laplacian.bmp"); // Save bmp
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Laplacian Time: " << duration.count() << " milliseconds" << endl;
			break;

			// contrast (pexels, height, width, red increase, int green increase, blue increase)
		case 3:

			int rgb_values[3];
			cout << "\n********* RGB value *********" << endl;

			// pick color
			for (int i = 0; i < 3; i++) {
				cout << rgb_types[i] << " Value (-255 to 255): ";
				cin >> value;
				rgb_values[i] = value;
			}

			// Display Colors
			for (int i = 0; i < 3; i++) {
				cout << rgb_types[i] << ": " << rgb_values[i];
			}

			start = std::chrono::high_resolution_clock::now(); // start timer
			d_contrast(pixels, image.getHeight(), image.getWidth(), rgb_values[0], rgb_values[1], rgb_values[2]); // kernel
			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("Contrast.bmp"); // Save bmp
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Contrast Time: " << duration.count() << " milliseconds" << endl;
			break;

			// brightness (pixels, height, width, bright);
		case 4:

			do {
				cout << "******* Brightness Multiplier *******";
				cout << "brightness: 2 - 5";
				cin >> choice;
				if (choice == NUM_IMAGES) {
					cout << "Goodbye!\n";
					exit(0);
				}
			} while (choice < 2 || choice > EDGE_FILTERS);

			start = std::chrono::high_resolution_clock::now(); // start timer
			d_brightness(pixels, image.getHeight(), image.getWidth(), choice); // kernel
			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("brightness.bmp"); // Save bmp
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Brightness Time: " << duration.count() << " milliseconds" << endl;

			break;

		case 5:
			start = std::chrono::high_resolution_clock::now(); // start timer
			d_convert_blur(pixels, image.getHeight(), image.getWidth()); // kernel
			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("Blur.bmp"); // Save bmp
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Blur Time: " << duration.count() << " milliseconds" << endl;
			break;

		case 6:
			start = std::chrono::high_resolution_clock::now(); // start timer
			//CannyShit(pixels, edgeDir, gradiant, image.getHeight(), image.getWidth());
			d_convert_greyscale(pixels, image.getHeight(), image.getWidth());
			//if printed, this doesn't get printed as grey for some reason
			gradiantLauncher(pixels, edgeDir, gradiant, image.getHeight(), image.getWidth());
			traceEdge(pixels, edgeDir, gradiant, image.getHeight(), image.getWidth());

			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("Canny.bmp"); // Save bmp
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Canny Time: " << duration.count() << " milliseconds" << endl;

			break;
		default:
			// Somehow go back
			break;

		}

		cout << "Check out the results\n\n";
		char response = 'y';
		cout << "Do you wish to repeat? [y/n] ";
		cin >> response;
		if (response != 'y') {
			cout << "Sorry to see you go ...\n";
			exit(0);
		}
	}
}

