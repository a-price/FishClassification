#include "SVMVisualization.h"

// get the svm weights by multiplying the support vectors by the alpha values
cv::Mat svmHyperplane(const cv::SVM& svm, const int numOfFeatures)
{
	int numSupportVectors = svm.get_support_vector_count();
	const float *supportVector;
	mySVM msvm;
	const CvSVMDecisionFunc *dec = msvm.getDecisionFunction(&svm);
	cv::Mat svmWeights(numOfFeatures+1, 1, CV_32F);//= (float *) calloc((numOfFeatures+1),sizeof(float));
	for (int i = 0; i < numSupportVectors; ++i)
	{
		float alpha = *(dec[0].alpha + i);
		supportVector = svm.get_support_vector(i);
		for(int j=0;j<numOfFeatures;j++)
			((float*)svmWeights.data)[j] += alpha * *(supportVector+j);
	}
	((float*)svmWeights.data)[numOfFeatures] = - dec[0].rho; //Be careful with the sign of the bias!

	return svmWeights;
}

typedef int DataType;
cv::Mat downproject(const cv::SVM& svm, const cv::Mat& data, const cv::Mat& labels, bool rowVectors)
{
	cv::PCA pca(data, cv::Mat(), rowVectors ? CV_PCA_DATA_AS_ROW : CV_PCA_DATA_AS_COL, 2);
	std::cout << pca.eigenvectors.rows << ", " << pca.eigenvectors.cols << std::endl;
	cv::Mat test = pca.eigenvectors.row(1);
	cv::Mat hyperplane = svmHyperplane(svm, rowVectors ? data.cols : data.rows);
	if (rowVectors)
	{
		cv::transpose(hyperplane, hyperplane);
	}

	cv::Mat temp = pca.eigenvectors;
	temp.row(1) = hyperplane;

	cv::Mat projection;

	// Project into 2D using hyperplane and principal component as basis
	CV_Assert( pca.mean.data && pca.eigenvectors.data &&
		((pca.mean.rows == 1 && pca.mean.cols == data.cols) || (pca.mean.cols == 1 && pca.mean.rows == data.rows)));
	cv::Mat tmp_data, tmp_mean = repeat(pca.mean, data.rows/pca.mean.rows, data.cols/pca.mean.cols);
	int ctype = pca.mean.type();
	if( data.type() != ctype || tmp_mean.data == pca.mean.data )
	{
		data.convertTo( tmp_data, ctype );
		subtract( tmp_data, tmp_mean, tmp_data );
	}
	else
	{
		subtract( data, tmp_mean, tmp_mean );
		tmp_data = tmp_mean;
	}
	if( pca.mean.rows == 1 )
	{
		cv::gemm( tmp_data, pca.eigenvectors, 1, cv::Mat(), 0, projection, cv::GEMM_2_T );
	}
	else
	{
		cv::gemm( pca.eigenvectors, tmp_data, 1, cv::Mat(), 0, projection, 0 );
	}


	//cv::Mat projection = pca.project(data);

	// Put into column vectors
	if (rowVectors)
	{
		cv::transpose(projection, projection);
	}

	// Get bounds of data
	int dataLen = projection.cols;
	float xMin, yMin, xMax, yMax;
	xMin = yMin = std::numeric_limits<float>::max();
	xMax = yMax = std::numeric_limits<float>::min();
	for (int i = 0; i < dataLen; i++)
	{
		float x = ((DataType*)projection.data)[2*i];
		float y = ((DataType*)projection.data)[2*i+1];
		if (x > xMax) { xMax = x; }
		if (x < xMin) { xMin = x; }
		if (y > yMax) { yMax = y; }
		if (y < yMin) { yMin = y; }
	}

	// Compute scales to map to 640x480
	float xScale = 640.0f/(xMax - xMin)/1.1f;
	float yScale = 480.0f/(yMax - yMin)/1.1f;

	// Create a blank image
	cv::Mat img = cv::Mat::zeros(480, 640, CV_32FC3);

	// Draw the points
	for (int i = dataLen - 1; i >= 0; i--)
	{
		float x = ((DataType*)projection.data)[2*i];
		float y = ((DataType*)projection.data)[2*i+1];
		int label = ((int*)labels.data)[i];
		if (label == 1)
		{
			cv::circle(img, cv::Point((x-xMin*1.1f)*xScale, (y-yMin*1.1f)*yScale), 3, cv::Scalar(0, 1.0, 0), -1);
		}
		else
		{
			cv::circle(img, cv::Point((x-xMin*1.1f)*xScale, (y-yMin*1.1f)*yScale), 3, cv::Scalar(0, 0, 1.0), 2);
		}
	}

	return img;
}