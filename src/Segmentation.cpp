#include <iostream>
#include <math.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Segmentation.h"
#include "Filter.h"

////////////////////////////////////////////////////////////////////////////////////
// constructor and destructor
////////////////////////////////////////////////////////////////////////////////////
Segmentation::Segmentation(){}

Segmentation::~Segmentation(){}


////////////////////////////////////////////////////////////////////////////////////
// Compute Hough Transformation
////////////////////////////////////////////////////////////////////////////////////
void Segmentation::houghTransform(const cv::Mat &input, float phiStep, cv::Mat &output)
{
    int rows = input.rows;
    int cols = input.cols;

    int dmax = (int) sqrt(pow(rows,2) + pow(cols,2)); //find the maximum distance from the middle ... 
    int phiMax = (int) (180.0f/phiStep); //find the phi for the requirement of x axis in the case of the polar coordinate system 
    int phiRad = phiStep*(3.14/180.0f); // find the radian based step for the phi (sin and cosine works on radians here :( ))

    std::cout << "rows : " << rows; 
    std::cout << "cols : " << cols; 
    
    output.release();

    // create a 32bit signed integer image initialized with zeros
    output = cv::Mat::zeros( 2*dmax, phiMax, CV_32S); // CREATE A MATRIX WHICH WOULD MIMIC A POLAR COORDINATE SYSTEM HERE ... HENCE, D AND PHI ARE THE Y AND X Axis

     for(int r = 0 ; r < rows ; ++rows )
     {  
         for (int c = 0 ; c < cols ; ++c)
         {
            std::cout << "Calculating .. " << std::endl; 
            if (input.at<float>(r,c) != 0) 
            {
                //calculate for each phistep , the value for the the coordinate system 
                for(int phi = 0 ; phi < phiMax ; ++phi)
                {
                    //radian mechanism
                    float phirad = phi * phiStep; 

                    float d = (float)(c * std::cos(phiRad))+(float)(r * std::sin(phiRad));
                    
                    int dInt = 0; 
                    
                    //rounding 
                    if (d > 0.0f)
                    {
                        dInt = d+0.5;
                    }
                    else
                    {
                        dInt = d-0.5; 
                    } 

                    float dplot = dmax - dInt ; // the graph is between +dmax and -dmax

                    ++output.at<float>(dplot,phi); //Accumulator ... More the number of lines passing through the pixel , more the brightness ...      
                }
            }
         }
     }    
}

////////////////////////////////////////////////////////////////////////////////////
// Find brightest pixel and return its coordinates as Point
////////////////////////////////////////////////////////////////////////////////////
cv::Point Segmentation::findMaximum(const cv::Mat &input)
{
    // declare array to hold the indizes
    int maxIndex[2];
    
    // find the maximum
    cv::minMaxIdx(input, 0, 0, 0, maxIndex);
    
    // create Point and return
    return cv::Point(maxIndex[1], maxIndex[0]);
}

///////////////////////////////////////////////////////////////////////////////
// scale a Hough image for better displaying
///////////////////////////////////////////////////////////////////////////////
void Segmentation::scaleHoughImage(const cv::Mat &input, cv::Mat &output)
{
    // find max value
    double max;
    cv::minMaxLoc(input, 0, &max);

    // scale the image
    input.convertTo(output, CV_32F, 1.0f / max, 0);
}

////////////////////////////////////////////////////////////////////////////////////
// Find the n local maxima and return the coordinates in a cv::Mat
////////////////////////////////////////////////////////////////////////////////////
cv::Mat Segmentation::findMaxima(const cv::Mat &input, int n)
{
    int yMax = input.rows - 1;
    int xMax = input.cols - 1;

    int rOffset = input.rows / 2;

    // create a copy of the input image
    cv::Mat inputCopy = input.clone();

    // create a 32bit signed integer image initialized with zeros
    cv::Mat maxima = cv::Mat::zeros(n, 2, CV_32S);

    // find n local maxima
    for (int i = 0; i < n; ++i)
    {
        int *pMaxima = maxima.ptr<int>(i);

        // find the maximum
        cv::Point maxPoint = findMaximum(inputCopy);

        // store the maximum in output matrix
        *pMaxima = rOffset - maxPoint.y;
        *(pMaxima + 1) = maxPoint.x;

        // clear the found maximum and all pixels around it
        const int dist = 10;

        // find the indices of the region around the found maximum
        int xStart = maxPoint.x - dist;
        if (xStart < 0)
            xStart = 0;

        int xEnd = maxPoint.x + dist;
        if (xEnd > xMax)
            xEnd = xMax;

        int yStart = maxPoint.y - dist;
        if (yStart < 0)
            yStart = 0;

        int yEnd = maxPoint.y + dist;
        if (yEnd > yMax)
            yEnd = yMax;

        // set the pixel's values to zero
        for (int y = yStart; y <= yEnd; ++y)
        {
            int *pInput = inputCopy.ptr<int>(y) + xStart;

            for (int x = xStart; x <= xEnd; ++x)
            {
                *pInput = 0;
                ++pInput;
            }
        }
    }

    // return the maxima as cv::Mat
    return maxima;
}

////////////////////////////////////////////////////////////////////////////////////
// Draw straight lines
////////////////////////////////////////////////////////////////////////////////////
void Segmentation::drawLines(const cv::Mat &input, cv::Mat lines, float phiStep, cv::Mat &output)
{
    int rows = input.rows;
    int cols = input.cols;

    int nLines = lines.rows;
    float phiStepRad = phiStep * M_PI / 180.0f;

    // copy input image to output
    output = input.clone();

    // convert to RGB colour image
    cv::cvtColor(output, output, CV_GRAY2RGB);

    // draw the lines
    for (int l = 0; l < nLines; ++l)
    {
        int *pLines = lines.ptr<int>(l);

        // get d and phi
        int d = *pLines;
        float phi = (float) *(pLines + 1) * phiStep;
        float phiRad = (float) *(pLines + 1) * phiStepRad;

        // find start and end point of line
        cv::Point p1;
        cv::Point p2;

        if (phi == 0.0f)
        {
            // vertical line
            p1 = cv::Point(d, rows - 1);
            p2 = cv::Point(d, 0);
        }
        else if (phi == 90.0f)
        {
            // horizontal line
            p1 = cv::Point(0, rows - d - 1);
            p2 = cv::Point(cols - 1, rows - d - 1);
        }
        else if (phi < 90.0f)
        {
            // monotonically decreasing
            int x = d / cos(phiRad);
            int y = d / sin(phiRad);

            p1 = cv::Point(0, rows - y - 1);
            p2 = cv::Point(x, rows - 1);
        }
        else
        {
            // monotonically increasing
            int y = d / sin(phiRad);
            int x = -1.0f * tan(phiRad) * (rows - y);

            p1 = cv::Point(0, rows - y - 1);
            p2 = cv::Point(x, 0);
        }

        // draw a red line
        cv::line(output, p1, p2, cv::Scalar(0.0f, 0.0f, 1.0f), 1);
    }
}
