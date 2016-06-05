#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/core/core.hpp>

class Segmentation
{
public:
    Segmentation();
    ~Segmentation();

    // Hough Transformation
    void houghTransform(const cv::Mat &input, float phiStep, cv::Mat &output);
    cv::Point findMaximum(const cv::Mat &input);
    void scaleHoughImage(const cv::Mat &input, cv::Mat &output);
    cv::Mat findMaxima(const cv::Mat &input, int n);
    void drawLines(const cv::Mat &input, cv::Mat lines, float phiStep, cv::Mat &output);

private:

};

#endif /* SEGMENTATION_H */
