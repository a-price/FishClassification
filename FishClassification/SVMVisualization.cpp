#include "SVMVisualization.h"

cv::Mat downproject(const cv::Mat& data, bool rowVectors)
{
	cv::PCA pca(data, cv::Mat(), rowVectors ? CV_PCA_DATA_AS_ROW : CV_PCA_DATA_AS_COL, 2);
	cv::Mat projection = pca.project(data);
	
	// Put into column vectors
	if (rowVectors)
	{
		cv::transpose(projection, projection);
	}

	// Get bounds of data
	int dataLen = projection.rows;
	std::cout << "Data Length: " << dataLen << std::endl;
	float xMin, yMin, xMax, yMax;
	xMin = yMin = std::numeric_limits<float>::max();
	xMax = yMax = std::numeric_limits<float>::min();
	for (int i = 0; i < dataLen; i++)
	{
		float x = ((float*)projection.data)[2*i];
		float y = ((float*)projection.data)[2*i+1];
		if (x > xMax) { xMax = x; }
		if (x < xMin) { xMin = x; }
		if (y > yMax) { yMax = y; }
		if (y < yMin) { yMin = y; }
	}

	// Compute scales to map to 640x480
	float xScale = 640.0f/(xMax - xMin + 1);
	float yScale = 480.0f/(yMax - yMin + 1);

	// Create a blank image
	cv::Mat img = cv::Mat::ones(480, 640, CV_32FC1);

	// Draw the points
	for (int i = 0; i < dataLen; i++)
	{
		float x = ((float*)projection.data)[2*i];
		float y = ((float*)projection.data)[2*i+1];
		cv::circle(img, cv::Point((x-xMin+1)*xScale, (y-yMin+1)*yScale), 3, cv::Scalar(255, 0, 0));
	}

	return img;
}