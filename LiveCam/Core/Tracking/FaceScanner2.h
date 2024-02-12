#pragma once

#include <vector>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "libTracker.h"
#include <Tracking/TrackingStructs.h>
#include <Tracking/TrackingTarget.h>
#include <Common/PointCalcUtil.h>

class FaceScanner2
{
public:
	FaceScanner2(TRACKER_LIB tLib);
	~FaceScanner2();

	int capacity = 1;
	double activescale = 1.0;
	bool canPreProcess = true;

	float confidenceThresh = CONFIDENCE_THRESHOLD;

	/* --ScannerInterface-- */
	void start();
	void stop();
	void setRefFrame(cv::Mat frame, int x, int y);
	cv::Mat prepScanFrame(cv::Mat frame);
	cv::Mat prepKeypointsFrame(cv::Mat frame);
	std::vector<TrackingTarget> scan(cv::Mat frame);
	std::vector<TrackingTarget> keypoints(
		int frameCols, 
		int frameRows, 
		const float* rawPoints[MAX_TARGETS_COUNT][2],
		float trackerValues[MAX_TARGETS_COUNT][NUMBER_OF_PRESET_VALUES],
		const float* confidence[MAX_TARGETS_COUNT]
	);
	void scale(double newscale);

private:
	TrackerConfigStruct scanConfig;
	TrackerConfigStruct keypointsConfig;
};
