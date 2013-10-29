// FishClassification.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FishClassification.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// The one and only application object

CWinApp theApp;

using namespace std;

int _tmain(int argc, TCHAR* argv[], TCHAR* envp[])
{
	int nRetCode = 0;

	HMODULE hModule = ::GetModuleHandle(NULL);

	if (hModule != NULL)
	{
		// initialize MFC and print and error on failure
		if (!AfxWinInit(hModule, NULL, ::GetCommandLine(), 0))
		{
			// TODO: change error code to suit your needs
			_tprintf(_T("Fatal Error: MFC initialization failed\n"));
			nRetCode = 1;
		}
		else
		{
			try {
				// Number of clusters for building BOW vocabulary from SURF features
				int clusters = 1000;    
				categorizer c(argv[1], clusters);
				if(atoi(argv[2]) == 0) {
					c.build_vocab();
					c.train_classifiers();
				} else {
					c.load_vocab();
				}

				//VideoCapture cap(0);
				namedWindow("Detected object");
				c.categorize();
			} catch(cv::Exception &e) {
				printf("Error: %s\n", e.what());
			}
			cin.get();
		}
	}
	else
	{
		// TODO: change error code to suit your needs
		_tprintf(_T("Fatal Error: GetModuleHandle failed\n"));
		nRetCode = 1;
	}

	return nRetCode;
}

inline void DisplayHistogram(Mat &hist, HistInfo &histInfo, string window) {
	//for displaying the histogram
	double maxVal=0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);
	Mat histImg = Mat::zeros( histInfo.s_bins*10,  histInfo.h_bins*10, CV_8UC3);
	for( int h = 0; h < histInfo.h_bins; h++ )
		for( int s = 0; s < histInfo.s_bins; s++ )
		{
			float binVal = hist.at<float>(h, s);
			int intensity = cvRound(binVal*255/maxVal);
			rectangle( histImg, Point(h*10, s*10),
				Point( (h+1)*10 - 1, (s+1)*10 - 1),
				Scalar::all(intensity),
				CV_FILLED );
		}
		namedWindow( window, 1 );
		imshow( window, histImg );
		//cvWaitKey();
}

inline void ComputeHistogram(Mat &img, HistInfo &histInfo, Mat *hist) {
	Mat hist_base;
	calcHist( &img, 1, histInfo.channels, Mat(), hist_base, 2, histInfo.histSize, histInfo.ranges, true, false );
	normalize( hist_base, *hist, 0, 1, NORM_MINMAX, -1, Mat() );
	/*int tmp = hist_base.channels();
	*hist = Mat(1,hist_base.rows * hist_base.cols,hist_base.type());
	MatIterator_<float> pI = hist_base.begin<float>(), pO = hist->begin<float>(), pEnd = hist_base.end<float>();
	while(pI != pEnd) {
		*pO++ = *pI++;
	}*/	
}

string categorizer::remove_extension(string full) {
	int last_idx = full.find_last_of(".");
	string name = full.substr(0, last_idx);
	return name;
}

categorizer::categorizer(string direc, int _clusters) {
	clusters = _clusters;
	// Initialize pointers to all the feature detectors and descriptor extractors
	featureDetector = (new SurfFeatureDetector());
	descriptorExtractor = (new SurfDescriptorExtractor());
	hogDescriptor = (new HOGDescriptor());
	bowtrainer = (new BOWKMeansTrainer(clusters));
	descriptorMatcher = (new FlannBasedMatcher());
	bowDescriptorExtractor = (new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher));


	//set up folders
	template_folder = direc + "templates\\";
	test_folder = direc + "test_images\\";
	train_folder = direc + "train_images\\";
	vocab_folder = direc;

	// Organize the object templates by category
	// Boost::filesystem directory iterator
	for(directory_iterator i(template_folder), end_iter; i != end_iter; i++) {
		// Prepend full path to the file name so we can imread() it
		string filename = string(template_folder) + i->path().filename().string();
		// Get category name by removing extension from name of file
		string category = remove_extension(i->path().filename().string());
		Mat im = imread(filename, CV_LOAD_IMAGE_COLOR), templ_im;
		objects[category] = im;
		cvtColor(im, templ_im, CV_BGR2GRAY);
		templates[category] = templ_im;
	}
	cout << "Initialized" << endl;

	// Organize training images by category
	make_train_set();
}

void categorizer::make_train_set() {
	string category;
	// Boost::filesystem recursive directory iterator to go through all contents of TRAIN_FOLDER
	for(recursive_directory_iterator i(train_folder), end_iter; i != end_iter; i++) {
		// Level 0 means a folder, since there are only folders in TRAIN_FOLDER at the zeroth level
		if(i.level() == 0) {
			// Get category name from name of the folder
			category = (i->path()).filename().string();
			category_names.push_back(category);
		}
		// Level 1 means a training image, map that by the current category
		else {
			// File name with path
			string filename = string(train_folder) + category + string("/") + (i->path()).filename().string();
			//load and downsize the image
			//Mat tmp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
			//Mat downsized;
			//pyrDown(tmp,downsized,Size(tmp.cols / 2, tmp.rows /2));
			// Make a pair of string and Mat to insert into multimap
			pair<string, Mat> p(category, imread(filename));
			train_set.insert(p);
		}
	}
	// Number of categories
	categories = category_names.size();
	cout << "Discovered " << categories << " categories of objects" << endl;
}

inline void PrepImgForHog(const Mat &in, Mat &out) {
	out = in;
	if(in.cols > in.rows) 
		transpose(out,out);
	resize(out,out,Size(64,128));
}

void categorizer::make_pos_neg() {
	// Iterate through the whole training set of images
	for(multimap<string, Mat>::iterator i = train_set.begin(); i != train_set.end(); i++) {
		// Category name is the first element of each entry in train_set
		string category = i->first;
		// Training image is the second elemnt
		Mat im = i->second, feat, im_g, im_hsv, hist;
		cvtColor(im,im_g,CV_BGR2GRAY);
		cvtColor(im,im_hsv, CV_BGR2HSV);
		// Detect keypoints, get the image BOW descriptor
		vector<KeyPoint> kp;
		featureDetector->detect(im_g, kp);
		bowDescriptorExtractor->compute(im_g, kp, feat);
		//compute histogram
		cout << category << endl;
		ComputeHistogram(im_hsv,histInfo,&hist);
		//compute HoG descriptors
		vector<float> descriptors;
		Mat img_small;
		PrepImgForHog(im_g, img_small);
		hogDescriptor->compute(img_small, descriptors);
		//Mat disp = VisualizeHoG(img_small,descriptors);
		//imshow("Hog", disp);
		//cvWaitKey();
		Mat hogs = Mat(descriptors);
		transpose(hogs, hogs);
		// Mats to hold the positive and negative training data for current category
		Mat pos, neg;
		for(int cat_index = 0; cat_index < categories; cat_index++) {
			string check_category = category_names[cat_index];
			// Add BOW feature as positive sample for current category ...
			if(check_category.compare(category) == 0) {
				positive_surf[check_category].push_back(feat);
				hists[check_category].push_back(hist);
				//positive_hist[check_category].push_back(hist);
				positive_hog[check_category].push_back(hogs);
				//... and negative sample for all other categories
			} else {
				negative_surf[check_category].push_back(feat);
				//negative_hist[check_category].push_back(hist);
				negative_hog[check_category].push_back(hogs);
			}
		}
	}

	// Debug message
	for(int i = 0; i < categories; i++) {
		string category = category_names[i];
		cout << "Category " << category << ": " << positive_surf[category].rows << " Positives, " << negative_surf[category].rows << " Negatives" << endl;
	}
}

void categorizer::build_vocab() {
	cout << "Building vocabulary" << endl;
	// Mat to hold SURF descriptors for all templates
	Mat vocab_descriptors;
	// For each template, extract SURF descriptors and pool them into vocab_descriptors
	Mat templ, desc;
	for(map<string, Mat>::iterator i = templates.begin(); i != templates.end(); i++) {
		vector<KeyPoint> kp;
		templ = i->second;
		featureDetector->detect(templ, kp);
		descriptorExtractor->compute(templ, kp, desc);
		vocab_descriptors.push_back(desc);
		//templ.release();
		//desc.release();
	}

	// Add the descriptors to the BOW trainer to cluster
	bowtrainer->add(vocab_descriptors);
	// cluster the SURF descriptors
	vocab = bowtrainer->cluster();

	// Save the vocabulary
	FileStorage fs(vocab_folder + "vocab.xml", FileStorage::WRITE);
	fs << "vocabulary" << vocab;
	fs.release();

	cout << "Built vocabulary" << endl;
} 

void categorizer::load_vocab() {
	//load the vocabulary
	FileStorage fs(vocab_folder + "vocab.xml", FileStorage::READ);
	fs["vocabulary"] >> vocab;
	fs.release();

	// Set the vocabulary for the BOW descriptor extractor
	bowDescriptorExtractor->setVocabulary(vocab);

	//load the classifiers
	for(int i = 0; i < categories; i++) {
		string category = category_names[i];
		string svm_filename = string(vocab_folder) + category + string("SVM.xml");
		svms_surf[category].load(svm_filename.c_str());
		svm_filename = string(vocab_folder) + category + string("HIST.xml");
		FileStorage fs(svm_filename, FileStorage::READ);
		fs["hist"] >> trained_hists[category];
		fs.release();
		//svms_hist[category].load(svm_filename.c_str());
		svm_filename = string(vocab_folder) + category + string("HOG.xml");
		svms_hog[category].load(svm_filename.c_str());
	}
}

void categorizer::train_classifiers() {
	// Set the vocabulary for the BOW descriptor extractor
	bowDescriptorExtractor->setVocabulary(vocab);
	// Extract BOW descriptors for all training images and organize them into positive and negative samples for each category
	make_pos_neg();

	for(int i = 0; i < categories; i++) {
		string category = category_names[i];

		// Postive training data has labels 1
		Mat train_data_surf = positive_surf[category], train_labels_surf = Mat::ones(train_data_surf.rows, 1, CV_32S);
		// Negative training data has labels 0
		train_data_surf.push_back(negative_surf[category]);
		Mat m = Mat::zeros(negative_surf[category].rows, 1, CV_32S);
		train_labels_surf.push_back(m);

		Mat total_hist = hists[category][0];
		for(int j = 1; j < hists[category].size(); j++) {
			total_hist += hists[category][j];
		}
		normalize( total_hist, total_hist, 0, 1, NORM_MINMAX, -1, Mat() );
		trained_hists[category] = total_hist;
		DisplayHistogram(total_hist,histInfo,category);

		Mat train_data_hog = positive_hog[category], train_labels_hog = Mat::ones(train_data_hog.rows, 1, CV_32S);
		// Negative training data has labels 0
		train_data_hog.push_back(negative_hog[category]);
		m = Mat::ones(negative_hog[category].rows, 1, CV_32S) * -1;
		train_labels_hog.push_back(m);
		cout << "Rows: " << train_data_hog.rows << ", Cols: " << train_data_hog.cols << endl;

		// Train SVM!
		svms_surf[category].train(train_data_surf, train_labels_surf);
		//svms_hist[category].train(train_data_hist, train_labels_hist);
		svms_hog[category].train(train_data_hog, train_labels_hog);

		// Save SVM to file for possible reuse
		string svm_filename = string(vocab_folder) + category + string("SVM.xml");
		svms_surf[category].save(svm_filename.c_str());
		//svm_filename = string(vocab_folder) + category + string("SVM2.xml");
		//svms_hist[category].save(svm_filename.c_str());
		svm_filename = string(vocab_folder) + category + string("HOG.xml");
		svms_hog[category].save(svm_filename.c_str());
		svm_filename = string(vocab_folder) + category + string("HIST.xml");
		FileStorage fs(svm_filename, FileStorage::WRITE);
		fs << "hist" << total_hist;
		fs.release();
		cout << "Trained and saved SVM for category " << category << endl;
	}
	//cvWaitKey();
	positive_surf.clear();
	negative_surf.clear();
	positive_hist.clear();
	negative_hist.clear();
	positive_hog.clear();
	negative_hog.clear();
}

void categorizer::categorize(VideoCapture cap) {
	cout << "Starting to categorize objects" << endl;
	namedWindow("Image");

	while(char(waitKey(1)) != 'q') {
		Mat frame, frame_g;
		cap >> frame;
		imshow("Image", frame);

		cvtColor(frame, frame_g, CV_BGR2GRAY);

		// Extract frame BOW descriptor
		vector<KeyPoint> kp;
		Mat test;
		featureDetector->detect(frame_g, kp);
		bowDescriptorExtractor->compute(frame_g, kp, test);

		// Predict using SVMs for all catgories, choose the prediction with the most negative signed distance measure
		float best_score = 777;
		string predicted_category;
		for(int i = 0; i < categories; i++) {
			string category = category_names[i];
			float prediction = svms_surf[category].predict(test, true);
			//cout << category << " " << prediction << " ";
			if(prediction < best_score) {
				best_score = prediction;
				predicted_category = category;
			}
		}
		//cout << endl;

		// Pull up the object template for the detected category and show it in a separate window
		imshow("Detected object", objects[predicted_category]);
	}
}

void categorizer::categorize() {
	cout << "Starting to categorize objects" << endl;
	namedWindow("Image");

	for(directory_iterator i(test_folder), end_iter; i != end_iter; i++) {
		Mat frame, frame_small, frame_g, frame_hsv;
		// Prepend full path to the file name so we can imread() it
		string filename = string(test_folder) + i->path().filename().string();
		cout << "Opening file: " << filename << endl;
		frame = imread(filename);
		//should downsize for speed but need to improve results
		resize(frame, frame_small, Size(), 0.25f, 0.25f);
		//pyrDown(frame, frame_small, Size(frame.cols / 2, frame.rows / 2));
		cvtColor(frame_small, frame_g, CV_BGR2GRAY);
		cvtColor(frame_small, frame_hsv, CV_BGR2HSV);
		imshow("Image", frame_small);
		// Extract frame BOW descriptor
		vector<KeyPoint> kp;
		Mat test_surf, test_hist;
		featureDetector->detect(frame_g, kp);
		bowDescriptorExtractor->compute(frame_g, kp, test_surf);

		ComputeHistogram(frame_hsv,histInfo,&test_hist);
		DisplayHistogram(test_hist,histInfo,"current");

		vector<float> descriptors;
		Mat frameg_small;
		PrepImgForHog(frame, frameg_small);
		hogDescriptor->compute(frameg_small, descriptors);
		Mat test_hog = Mat(descriptors);
		transpose(test_hog,test_hog);
		cout << "Rows: " << test_hog.rows << ", Cols: " << test_hog.cols << endl;

		// Predict using SVMs for all catgories, choose the prediction with the most negative signed distance measure
		float best_score = 777, best_score2 = -1, best_score3 = 777;
		string predicted_category, predicted_category2, predicted_category3;
		for(int i = 0; i < categories; i++) {
			string category = category_names[i];
			float prediction = svms_surf[category].predict(test_surf, true);
			float prediction2 = compareHist(test_hist, trained_hists[category], CV_COMP_CORREL);
			//float prediction3 = svms_hog[category].predict(test_hog, true);
			cout << category << " " << prediction << " " << prediction2 << " ";
			if(prediction < best_score) {
				best_score = prediction;
				predicted_category = category;
			}
			if(prediction2 > best_score2) {
				best_score2 = prediction2;
				predicted_category2 = category;
			}
		}
		cout << endl;

		// Pull up the object template for the detected category and show it in a separate window
		if(!predicted_category.empty() && predicted_category == predicted_category2) {
			imshow("Detected object", objects[predicted_category]);
		} else {
			cout << "Couldn't find a match!\n" << endl;
			imshow("Detected object", NULL);
		}
		cout << "SIFT Result: " << predicted_category << "\tHist Result: " << predicted_category2 << "\tHog Result: " << predicted_category3 << endl;
		waitKey();
	}
}