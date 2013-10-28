/// Method to visualize the separation created by an SVM
/// by projecting the data down to the two principal divisions

#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

cv::Mat downproject(const cv::Mat& data, bool rowVectors = false);