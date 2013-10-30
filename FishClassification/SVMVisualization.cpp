#include "SVMVisualization.h"

// get the svm weights by multiplying the support vectors by the alpha values
cv::Mat svmHyperplane(const cv::SVM& svm, const int numOfFeatures)
{
	int numSupportVectors = svm.get_support_vector_count();
	const float *supportVector;
	mySVM msvm;
	const CvSVMDecisionFunc *dec = msvm.getDecisionFunction(&svm);
	std::cout << "SVs: " << dec->sv_count << std::endl;
	cv::Mat svmWeights(numOfFeatures+1, 1, CV_32F);//= (float *) calloc((numOfFeatures+1),sizeof(float));
	for (int i = 0; i < numSupportVectors; ++i)
	{
		float alpha = *(dec->alpha + i);
		supportVector = svm.get_support_vector(i);
		for(int j=0;j<numOfFeatures;j++)
			((float*)svmWeights.data)[j] += alpha * *(supportVector+j);
	}
	((float*)svmWeights.data)[numOfFeatures] = - dec->rho; //Be careful with the sign of the bias!

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
	std::cout << temp << std::endl;
	for (int i = 0; i < temp.cols; i++)
	{
		temp.at<float>(1, i) = hyperplane.at<float>(0, i);
	}
	std::cout << hyperplane << std::endl;

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
		cv::gemm( tmp_data, temp, 1, cv::Mat(), 0, projection, cv::GEMM_2_T );
	}
	else
	{
		cv::gemm( temp, tmp_data, 1, cv::Mat(), 0, projection, 0 );
	}
	//pca.project(data, projection);
	std::cout << projection << std::endl;


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
		float x = projection.at<float>(0,i);
		float y = projection.at<float>(1,i);
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
		float x = projection.at<float>(0,i);
		float y = projection.at<float>(1,i);
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

	int c = svm.get_support_vector_count();
	std::cerr << c << std::endl;
	for (int i = 0; i < c; ++i)
	{
		const float* v = svm.get_support_vector(i); // get and then highlight with grayscale
		cv::circle(img, cv::Point( (int) v[0], (int) v[1]), 6, cv::Scalar(128, 128, 128), -1);
	}

	return img;
}

cv::Mat testSVM()
{
	// Testing SVM
	cv::SVM svm;
	const int TRAINING_SIZE = 6;

	// Set up training data
	int labels[TRAINING_SIZE] = {1, 1, 1, -1, -1, -1};
	cv::Mat labelsMat(TRAINING_SIZE, 1, CV_32SC1, labels);

	//float trainingData[TRAINING_SIZE][2] = { {501, 10}, {450, 122}, {255, 10}, {501, 255}, {10, 501} };
	float trainingData[TRAINING_SIZE][2] = { {5, 5}, {10, 100}, {7, 249}, {250, 10}, {255, 101}, {250, 250} };
	cv::Mat trainingDataMat(TRAINING_SIZE, 2, CV_32FC1, trainingData);

	cv::Mat img = cv::Mat::zeros(512, 512, CV_32FC3);
	const int dataLen = trainingDataMat.rows;
	// Draw the points
	for (int i = dataLen - 1; i >= 0; i--)
	{
		float x = trainingDataMat.at<float>(i,0);
		float y = trainingDataMat.at<float>(i,1);
		int label = ((int*)labelsMat.data)[i];
		if (label == 1)
		{
			cv::circle(img, cv::Point(x, y), 3, cv::Scalar(0, 1.0, 0), -1);
		}
		else
		{
			cv::circle(img, cv::Point(x, y), 3, cv::Scalar(0, 0, 1.0), 2);
		}
	}

	cv::imshow("testing0", img);

	// Set up SVM's parameters
	cv::SVMParams params;
	params.svm_type    = cv::SVM::C_SVC;
	params.kernel_type = cv::SVM::LINEAR;
	params.term_crit   = cv::TermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	std::cout << trainingDataMat.rows << ", " << trainingDataMat.cols << std::endl;

	// Train the SVM
	svm.train(trainingDataMat, labelsMat, cv::Mat(), cv::Mat(), params);
	
	cv::Mat result = downproject(svm, trainingDataMat, labelsMat, true);

	cv::imshow("testing1", result);
	cv::waitKey();

	return result;
	
}

cv::Mat sampleSVM()
{
	// Data for visual representation
	int width = 512, height = 512;
	cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	float labels[4] = {1.0, -1.0, -1.0, -1.0};
	cv::Mat labelsMat(4, 1, CV_32FC1, labels);

	float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
	cv::Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train the SVM
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, cv::Mat(), cv::Mat(), params);

	cv::Vec3b green(0,255,0), blue (255,0,0);
	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			cv::Mat sampleMat = (cv::Mat_<float>(1,2) << i,j);
			float response = SVM.predict(sampleMat);

			if (response == 1)
				image.at<cv::Vec3b>(j, i)  = green;
			else if (response == -1)
				 image.at<cv::Vec3b>(j, i)  = blue;
		}

	// Show the training data
	int thickness = -1;
	int lineType = 8;
	circle( image, cv::Point(501,  10), 5, cv::Scalar(  0,   0,   0), thickness, lineType);
	circle( image, cv::Point(255,  10), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	circle( image, cv::Point(501, 255), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	circle( image, cv::Point( 10, 501), 5, cv::Scalar(255, 255, 255), thickness, lineType);

	// Show support vectors
	thickness = 2;
	lineType  = 8;
	int c     = SVM.get_support_vector_count();

	for (int i = 0; i < c; ++i)
	{
		const float* v = SVM.get_support_vector(i);
		circle( image,  cv::Point( (int) v[0], (int) v[1]),   6,  cv::Scalar(128, 128, 128), thickness, lineType);
	}

	//cv::imwrite("result.png", image);        // save the image

	cv::imshow("SVM Simple Example", image); // show it to the user
	cv::waitKey(0);
	return image;
}