// CS 555 Assignment 1 Connected Component Labeling.cpp : This file contains the 'main' function. Program execution begins and ends there.
//// 

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

void imhist(Mat image, int histogram[]);
void histDisplay(int histogram[], const char* name);
void dfs(int x, int y, int current_label);
void find_components();

// direction vectors
const int dx[] = { +1, 0, -1, 0 };
const int dy[] = { 0, +1, 0, -1 };

// matrix dimensions
int row_count = 320;
int col_count = 512;

// the input matrix
int matrix[320][512];

// the labels, 0 means unlabeled/black
int label[320][512];

int main()
{
     Mat f1 = imread("C:\\Users\\Aaron\\Downloads\\guide_8bits.bmp", IMREAD_GRAYSCALE);
     Mat image_dst;
     int hist[256];
     imhist(f1, hist);
     histDisplay(hist, "histogram");
     // A good Threshold value to use after looking at the Histogram would be 60
     int T = 60;
     // Given our now Threshold value we will use it to transform our image to a binary image
     for (int y = 0; y < f1.rows; y++)
     {
          for (int x = 0; x < f1.cols; x++)
          {
               // if less than T set it to 0
               if ((int)f1.at<uchar>(y, x) < T)
                    f1.at<uchar>(y, x) = 0;
               // else make it 255
               else
                    f1.at<uchar>(y, x) = 255;
          }
     }
     imshow("Binary image", f1);
     waitKey();
     Mat f2(320, 512, CV_8U);
     for (int y = 0; y < f1.rows; y++)
     {
          for (int x = 0; x < f1.cols; x++)
          {
               // move pixels over to our 2D array for easier pixel manipulation
               matrix[y][x] = (int)f1.at<uchar>(y, x);
          }
     }
     find_components();
     for (int y = 0; y < 320; y++)
     {
          for (int x = 0; x < 512; x++)
          {
               f2.at<uchar>(y, x) = label[y][x] * 100;
          }
     }
     imshow("f2", f2);
     waitKey();
     return 0;
}

void imhist(Mat image, int histogram[])
{

     // initialize all intensity values to 0
     for (int i = 0; i < 256; i++)
     {
          histogram[i] = 0;
     }

     // calculate the no of pixels for each intensity values
     for (int y = 0; y < image.rows; y++)
          for (int x = 0; x < image.cols; x++)
               histogram[(int)image.at<uchar>(y, x)]++;

}

void histDisplay(int histogram[], const char* name)
{
     int hist[256];
     for (int i = 1; i < 256; i++)
     {
          hist[i] = histogram[i];
     }
     // draw the histograms
     int hist_w = 512; int hist_h = 400;
     int bin_w = cvRound((double)hist_w / 256);

     Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));

     // find the maximum intensity element from histogram
     int max = hist[1];
     for (int i = 2; i < 256; i++) {
          if (max < hist[i]) {
               max = hist[i];
          }
     }

     // normalize the histogram between 0 and histImage.rows

     for (int i = 1; i < 256; i++) {
          hist[i] = ((double)hist[i] / max) * histImage.rows;
     }


     // draw the intensity line for histogram
     for (int i = 1; i < 256; i++)
     {
          line(histImage, Point(bin_w * (i), hist_h),
               Point(bin_w * (i), hist_h - hist[i]),
               Scalar(0, 0, 0), 1, 8, 0);
     }

     // display histogram
     namedWindow(name, WINDOW_AUTOSIZE);
     imshow(name, histImage);
};

void dfs(int x, int y, int current_label)
{
     if (x < 0 || x == row_count) return; // out of bounds
     if (y < 0 || y == col_count) return; // out of bounds
     if (label[x][y] || !matrix[x][y]) return; // already labeled or not marked with 1 in m

     // mark the current cell
     label[x][y] = current_label;

     // recursively mark the neighbors
     for (int direction = 0; direction < 4; ++direction)
          dfs(x + dx[direction], y + dy[direction], current_label);
}

void find_components() 
{
     int component = 0;
     for (int i = 0; i < row_count; ++i)
          for (int j = 0; j < col_count; ++j)
               if (!label[i][j] && matrix[i][j]) dfs(i, j, ++component);
}