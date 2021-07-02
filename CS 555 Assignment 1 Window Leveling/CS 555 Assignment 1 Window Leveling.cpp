// CS 555 Assignment 1 Window Leveling.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
using namespace std;
using namespace cv;


void transformer(int LookUpTable[320][512], int window, int level, string imageName, Mat image);

int main()
{
     // Load the image
     Mat originalImage = imread("C:\\Users\\Aaron\\Downloads\\guide_8bits.bmp", 0);
     // If image doesn't exist output error
     if (originalImage.empty())
     {
          cout << "image not loaded";
     }
     // output image
     namedWindow("Original Image", WINDOW_AUTOSIZE);
     imshow("Original Image", originalImage);
     waitKey(0);
     
     // create LUT
     int LUT[320][512];
     for (int y = 0; y < originalImage.rows; y++)
     {
          for (int x = 0; x < originalImage.cols; x++)
          {
               LUT[y][x] = (int)originalImage.at<uchar>(y, x);
          }
     }
     // create a different Mat object for each different window/level combination
     Mat L50W30(320, 512, CV_8U), L127W30(320, 512, CV_8U), L200W30(320, 512, CV_8U), 
          L50W100(320, 512, CV_8U), L127W100(320, 512, CV_8U), L200W100(320, 512, CV_8U);
     transformer(LUT, 30, 50, "L50W30", L50W30);
     transformer(LUT, 30, 127, "L127W30", L127W30);
     transformer(LUT, 30, 200, "L200W30", L200W30);
     transformer(LUT, 100, 50, "L50W100", L50W100);
     transformer(LUT, 100, 127, "L127W100", L127W100);
     transformer(LUT, 100, 200, "L200W100", L200W100);
     return 0;
}

void transformer(int LookUpTable[320][512], int window, int level, string imageName, Mat image)
{
     // using level of 50 and window of 30
     int lowerBound = level - (window/2), upperBound = level + (window/2);
     for (int y = 0; y < 320; y++)
     {
          for (int x = 0; x < 512; x++)
          {
               if (LookUpTable[y][x] < lowerBound)
                    LookUpTable[y][x] = 0;
               else if (LookUpTable[y][x] > upperBound)
                    LookUpTable[y][x] = 255;
               else
                    LookUpTable[y][x] = round((255 / window) * (LookUpTable[y][x] - lowerBound));
          }
     }
     // Mat image(320, 512, CV_8U);
     for (int y = 0; y < 320; y++)
     {
          for (int x = 0; x < 512; x++)
          {
               image.at<uchar>(y, x) = LookUpTable[y][x];
          }
     }
     namedWindow(imageName, WINDOW_AUTOSIZE);
     imshow(imageName, image);
     waitKey(0);
};