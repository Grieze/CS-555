#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <cmath>
// https://learnopencv.com/optical-flow-in-opencv/#dense-optical-flow
// https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
using namespace std;
using namespace cv;

const int PI = 3.14;

double alpha(int i)
{
     if (i == 0)
          return (1 / sqrt(2));
     return 1;
}

int xGradient(Mat image, int x, int y)
{
     return image.at<uchar>(y - 1, x - 1) +
          2 * image.at<uchar>(y, x - 1) +
          image.at<uchar>(y + 1, x - 1) -
          image.at<uchar>(y - 1, x + 1) -
          2 * image.at<uchar>(y, x + 1) -
          image.at<uchar>(y + 1, x + 1);
}
int yGradient(Mat image, int x, int y)
{
     return image.at<uchar>(y - 1, x - 1) +
          2 * image.at<uchar>(y - 1, x) +
          image.at<uchar>(y - 1, x + 1) -
          image.at<uchar>(y + 1, x - 1) -
          2 * image.at<uchar>(y + 1, x) -
          image.at<uchar>(y + 1, x + 1);
}

// Apply Sobel Operator on the image
void sobel(Mat src)
{
     int gx, gy, sum;

     // Load an image
     Mat dst = src.clone();
     if (!src.data)
     {
          cout << -1;
     }

     for (int y = 0; y < src.rows; y++)
          for (int x = 0; x < src.cols; x++)
               dst.at<uchar>(y, x) = 0.0;

     for (int y = 1; y < src.rows - 1; y++) {
          for (int x = 1; x < src.cols - 1; x++) {
               gx = xGradient(src, x, y);
               gy = yGradient(src, x, y);
               sum = abs(gx) + abs(gy);
               sum = sum > 255 ? 255 : sum;
               sum = sum < 0 ? 0 : sum;
               dst.at<uchar>(y, x) = sum;
          }
     }

     namedWindow("final");
     imshow("final", dst);

     namedWindow("initial");
     imshow("initial", src);

     waitKey();
}

// Function to add 2 images
Mat add_images(Mat src, Mat dest)
{
     for (int y = 0; y < src.rows; y++)
     {
          for (int x = 0; x < src.cols; x++)
          {
               dest.at<uchar>(y, x) = src.at<uchar>(y, x) + dest.at<uchar>(y, x);
          }
     }

     return dest;
}

// Function to subtract 2 images
Mat subtract_images(Mat src, Mat dest)
{
     for (int y = 0; y < src.rows; y++)
     {
          for (int x = 0; x < src.cols; x++)
          {
               dest.at<uchar>(y, x) = src.at<uchar>(y, x) - dest.at<uchar>(y, x);
          }
     }

     return dest;
}
// Calculate the motion vectors
Mat opticalFlow(Mat frame1, Mat frame2)
{
     // Create some random colors
     vector<Scalar> colors;
     RNG rng;
     for (int i = 0; i < 100; i++)
     {
          int r = rng.uniform(0, 256);
          int g = rng.uniform(0, 256);
          int b = rng.uniform(0, 256);
          colors.push_back(Scalar(r, g, b));
     }
     vector<Point2f> p0, p1;
     // Take the first frame and find corners in it
     goodFeaturesToTrack(frame1, p0, 10000, 0.3, 7, Mat(), 7, false, 0.04);

     // Create a mask image for drawing purposes
     Mat mask = Mat::zeros(frame1.size(), frame1.type());

     // Calculate optical flow
     vector<uchar> status;
     vector<float> err;
     TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
     calcOpticalFlowPyrLK(frame1, frame2, p0, p1, status, err, Size(16, 16), 2, criteria);

     vector<Point2f> good_new;
     for (uint i = 0; i < p0.size(); i++)
     {
          // Select good points
          if (status[i] == 1)
          {
               good_new.push_back(p1[i]);
               // draw the displacement vectors
               line(mask, p1[i], p0[i], colors[i], 2);
               circle(frame2, p1[i], 5, colors[i], -1);
          }
     }
     Mat img;
     add(frame2, mask, img);

     //imshow("Motion Vectors", img);
     //waitKey();

     return img;
}

// Compute the DCT for the given image
Mat DCTImage(Mat src_image)
{
     int height = src_image.rows - (src_image.rows % 8);
     int width = src_image.cols - (src_image.cols % 8);
     float temp;
     Mat dst_image = src_image.clone();
     for (int m = 0; m < height; m += 8)
     {
          for (int n = 0; n < width; n += 8)					//These two loops divide the image into 8x8 blocks
          {
               for (int i = m; i < m + 8; i++)
               {
                    for (int j = n; j < n + 8; j++)
                    {
                         temp = 0.0;
                         for (int x = m; x < m + 8; x++)
                         {
                              for (int y = n; y < n + 8; y++)				//These four loops calculate the DCT coefficients for each 8x8 block
                              {
                                   temp += ((int)src_image.at<uchar>(x, y)) *
                                        (cos((((2 * x) + 1)) * ((i * PI)) / (2 * 8))) *
                                        (cos((((2 * y) + 1)) * ((j * PI)) / (2 * 8)));			//Formula for calculating DCT
                              }
                         }
                         temp *= ((1 / sqrt(2 * 8))) * (alpha(i) * alpha(j));
                         dst_image.at<uchar>(i, j) = int(temp);				//Assigning the calculated DCT coefficient to each intensity pixel in I
                    }

               }
          }
     }
     return dst_image;
}

//Computes the inverse discrete cosine transform
Mat IDCTImage(Mat src_image)
{
     int height = src_image.rows - (src_image.rows % 8);
     int width = src_image.cols - (src_image.cols % 8);
     float temp;
     Mat dst_image = src_image.clone();

     for (int m = 0; m < height; m += 8)
     {
          for (int n = 0; n < width; n += 8)					//These two loops divide the image into 8x8 blocks
          {
               for (int i = m; i < m + 8; i++)
               {
                    for (int j = n; j < n + 8; j++)
                    {
                         temp = 0.0;
                         for (int x = m; x < m + 8; x++)
                         {
                              for (int y = n; y < n + 8; y++)				//These four loops calculate the IDCT coefficients for each 8x8 block
                              {
                                   //Formula for calculating IDCT
                                   temp += ((int)src_image.at<uchar>(x, y)) * (cos((((2 * x) + 1)) * ((i * PI)) / (2 * 8))) * (cos((((2 * y) + 1)) * ((j * PI)) / (2 * 8))) * (alpha(x) * alpha(y));
                              }
                         }
                         dst_image.at<uchar>(i, j) = int(temp);				//Assigning the calculated IDCT coefficient to each intensity pixel in I

                    }
               }
          }
     }

     return dst_image;
}

double getMSE(const Mat& I1, const Mat& I2)
{
     Mat s1;
     absdiff(I1, I2, s1);       
     s1.convertTo(s1, CV_32F);  
     s1 = s1.mul(s1);           

     Scalar s = sum(s1);        

     double sse = s.val[0] + s.val[1] + s.val[2]; 


     double mse = sse / (double)(I1.channels() * I1.total());

     return mse;

}

int main()
{
     Mat im1frame1 = imread("OneStopNoEnter1cor0249.bmp", IMREAD_GRAYSCALE);
     Mat im1frame2 = imread("OneStopNoEnter1cor0251.bmp", IMREAD_GRAYSCALE);
     Mat im2frame1 = imread("seq00.avi0426g.bmp", IMREAD_GRAYSCALE);
     Mat im2frame2 = imread("seq00.avi0428g.bmp", IMREAD_GRAYSCALE);
     sobel(im1frame1);
     sobel(im1frame2);
     // Steps 2 to 6 for the first pair of images
     // step 2 for first image
     Mat df1 = im1frame2.clone();
     df1 = subtract_images(im1frame1, df1);
     imshow("df1",df1);
     waitKey();

     // step 3 for first image
     Mat mdf1 = opticalFlow(im1frame1, df1);
     imshow("mdf1", mdf1);
     waitKey();

     // step 4 for first image
     Mat DCTdf1 = DCTImage(df1);
     imshow("DCTdf1",DCTdf1);
     waitKey();
     Mat cdf1 = IDCTImage(DCTdf1);
     imshow("IDCTdf1", cdf1);
     waitKey();
     Mat DCTmdf1 = DCTImage(mdf1);
     imshow("DCTmdf1", DCTmdf1);
     waitKey();
     Mat cmdf1 = IDCTImage(DCTmdf1);
     imshow("IDCTmdf1", cmdf1);
     waitKey();

     // step 5 for first image
     Mat f2prime = add_images(im1frame1, cdf1);
     imshow("f2prime", f2prime);
     waitKey();
     Mat f2dprime = add_images(im1frame1, cmdf1);
     imshow("f2dprime", f2dprime);
     waitKey();

     // step 6 for first image
     double msef2andf2prime = getMSE(f2prime, im1frame2);
     cout << "The mean squared error of f2 and f2' is: " << msef2andf2prime <<endl;
     double msef2andf2dprime = getMSE(f2dprime, im1frame2);
     cout << "The mean squared error of f2 and f2'' is: " << msef2andf2dprime <<endl;

     // steps 2 to 6 for the second pair of images
     // step 2 for second image
     Mat df2 = im2frame2.clone();
     df2 = subtract_images(im2frame1, df2);
     imshow("df2", df2);
     waitKey();

     // step 3 for second image
     Mat mdf2 = opticalFlow(im2frame1, df2);
     imshow("mdf1", mdf2);
     waitKey();

     // step 4 for second image
     Mat DCTdf2 = DCTImage(df2);
     imshow("DCTdf2", DCTdf2);
     waitKey();
     Mat cdf2 = IDCTImage(DCTdf2);
     imshow("IDCTdf2", cdf2);
     waitKey();
     Mat DCTmdf2 = DCTImage(mdf2);
     imshow("DCTmdf2", DCTmdf2);
     waitKey();
     Mat cmdf2 = IDCTImage(DCTmdf2);
     imshow("IDCTmdf2", cmdf2);
     waitKey();

     // step 5 for second image
     f2prime = add_images(im2frame1, cdf2);
     imshow("f2prime", f2prime);
     waitKey();
     f2dprime = add_images(im2frame1, cmdf2);
     imshow("f2dprime", f2dprime);
     waitKey();

     // step 6 for second image
     msef2andf2prime = getMSE(f2prime, im1frame2);
     cout << "The mean squared error of f2 and f2' is: " << msef2andf2prime << endl;
     msef2andf2dprime = getMSE(f2dprime, im1frame2);
     cout << "The mean squared error of f2 and f2'' is: " << msef2andf2dprime << endl;

     return 0;
}
