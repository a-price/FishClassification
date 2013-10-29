/// Method to visualize the separation created by an SVM
/// by projecting the data down to the two principal divisions

#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

class mySVM : public cv::SVM
{
public:
	const CvSVMDecisionFunc* getDecisionFunction(const cv::SVM* target) const
	{
		return ((mySVM*)target)->decision_func;
	}
};

cv::Mat svmHyperplane(const cv::SVM& svm, const int numOfFeatures);

cv::Mat downproject(const cv::SVM& svm, const cv::Mat& data, const cv::Mat& labels, bool rowVectors = false);