#pragma once

#include "resource.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
using namespace boost::filesystem;

static class HistInfo {
public:
	/// Using 30 bins for hue and 32 for saturation
	static const int h_bins = 40; 
	static const int s_bins = 25;
	int histSize[2];
	// hue varies from 0 to 256, saturation from 0 to 180
	float h_ranges[2];
	float s_ranges[2];
	const float* ranges[2];
	// Use the o-th and 1-st channels
	int channels[];

	HistInfo() {
		histSize[0] = h_bins; histSize[1] = s_bins;
		// hue varies from 0 to 256, saturation from 0 to 180
		h_ranges[0] = 0; h_ranges[1] = 256;
		s_ranges[0] = 0; s_ranges[1] = 180;
		ranges[0] = h_ranges; ranges[1] = s_ranges;
		// Use the o-th and 1-st channels
		channels[0] = 0; channels[1] = 1;
	}
};

class categorizer {
private:
	map<string, Mat> templates, objects, positive_surf, negative_surf, positive_hist, negative_hist, positive_hog, negative_hog; //maps from category names to data
	multimap<string, Mat> train_set; //training images, mapped by category name
	map<string, CvSVM> svms_surf, svms_hist, svms_hog; //trained SVMs, mapped by category name
	vector<string> category_names; //names of the categories found in TRAIN_FOLDER
	int categories; //number of categories
	int clusters; //number of clusters for SURF features to build vocabulary
	Mat vocab; //vocabulary
	string train_folder, test_folder, template_folder, vocab_folder;

	// Feature detectors and descriptor extractors
	Ptr<FeatureDetector> featureDetector;
	Ptr<DescriptorExtractor> descriptorExtractor;
	Ptr<BOWKMeansTrainer> bowtrainer;
	Ptr<BOWImgDescriptorExtractor> bowDescriptorExtractor;
	Ptr<DescriptorMatcher> descriptorMatcher;
	Ptr<HOGDescriptor> hogDescriptor;
	HistInfo histInfo;

	void make_train_set(); //function to build the training set multimap
	void make_pos_neg(); //function to extract BOW features from training images and organize them into positive and negative samples 
	string remove_extension(string); //function to remove extension from file name, used for organizing templates into categories
public:
	categorizer(string direc, int _clusters); //constructor
	void build_vocab(); //function to build the BOW vocabulary
	void load_vocab(); //function to load the BOW vocabulary and classifiers
	void train_classifiers(); //function to train the one-vs-all SVM classifiers for all categories
	void categorize(VideoCapture); //function to perform real-time object categorization on camera frames
	void categorize(); //function to perform real-time object categorization on saved frames
};

#define THRESH 0.9970f
