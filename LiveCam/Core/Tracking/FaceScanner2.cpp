#include "FaceScanner2.h"

#include <Common/CommonClasses.h>
#include <Common/Resolutions.h>

FaceScanner2::FaceScanner2(TRACKER_LIB tLib)
{
	capacity = MAX_TARGETS_COUNT;
	scanConfig.size = cv::Size(300, 300);
	scanConfig.mean = cv::Scalar(104.0, 177.0, 123.0);
	scanConfig.inputName = "data";
	scanConfig.outputName = "detection_out";
	confidenceThresh = 0.5f;
}

FaceScanner2::~FaceScanner2()
{
}

void FaceScanner2::start()
{}

void FaceScanner2::stop()
{}

void FaceScanner2::setRefFrame(cv::Mat frame, int x, int y)
{}

cv::Mat FaceScanner2::prepScanFrame(cv::Mat frame)
{
	return frame;
}

cv::Mat FaceScanner2::prepKeypointsFrame(cv::Mat frame)
{
	return frame;
}

std::vector<TrackingTarget> FaceScanner2::scan(cv::Mat frame)
{
	float values[MAX_TARGETS_COUNT][NUMBER_OF_PRESET_VALUES];
	const float* rawPoints[MAX_TARGETS_COUNT][2];
	const float* confidence[MAX_TARGETS_COUNT];


	std::vector<TrackingTarget> result;

	int rtnstate;

	//process face tracking
	float yawAngle;

	rtnstate = ULS_TrackerProcessByte((void*)frame.data, frame.cols, frame.rows, 1, 0, &yawAngle);
	//draw results
	if (rtnstate == -50) {
		values[0][0] = -1.0f;
		return {};
	}
	else {
		int n = ULS_GetTrackerPointNum();
		for (int j = 0; j < MAX_TARGETS_COUNT; j++) {

			//	rawPoints[j] = NULL;
			const float* pts = ULS_GetTrackerPoint(j);

			const float* pts3d = ULS_GetTrackerPoint3D(j);

			if (pts == NULL) {
				values[j][POINT_ID] = -1.0f;
				continue;
			}

			int fx, fy, fw, fh;

			bool faceRectResult = ULS_GetTrackerFacerect(j, fx, fy, fw, fh);

			if (!faceRectResult)
			{
				values[j][POINT_ID] = -1.0f;
				continue;
			}

			// face center
			float imx, imy;
			ULS_GetFaceCenter(imx, imy, j);

			// with pose stability
			float pitch = ULS_GetStablePitchRadians(j);
			float yaw = ULS_GetStableYawRadians(j);
			float roll = ULS_GetStableRollRadians(j);

			cv::Point2f pupils[2];
			ULS_GetLeftPupilXY(pupils[0].x, pupils[0].y, j);
			ULS_GetRightPupilXY(pupils[1].x, pupils[1].y, j);

			cv::Point3f gazes[2];
			ULS_GetLeftGaze(gazes[0].x, gazes[0].y, gazes[0].z, j);
			ULS_GetRightGaze(gazes[1].x, gazes[1].y, gazes[1].z, j);

			//if (yaw > 45 || yaw < -45) return{};

			// project head center in world coordinate to image plane
			//	float nx, ny;
			float hx, hy, hz;
			ULS_GetHead3DPosition(hx, hy, hz, j);

			float hxCC, hyCC, hzCC;
			ULS_GetHead3DPositionCC(hxCC, hyCC, hzCC, j);

			float faceScale;
			faceScale = ULS_GetScaleInImage(j);

			int lX = pts[0];
			int ly = pts[1];
			int rX = pts[32];
			int ry = pts[33];

			// #1 = pts[2 * i], #2 = pts[2 * i + 1]
			// point # 28
			int noseX = pts[56];
			int noseY = pts[57];

			int lipX = pts[102];
			int lipY = pts[103];

			values[j][TOP_LIP_CENTER_X] = lipX;
			values[j][TOP_LIP_CENTER_Y] = lipY;

			values[j][FACE_CENTER_X] = noseX;
			values[j][FACE_CENTER_Y] = noseY;
			values[j][PITCH] = pitch;
			values[j][YAW] = yaw;
			values[j][ROLL] = roll;
			values[j][LEFT_PUPIL_X] = pupils[0].x;
			values[j][LEFT_PUPIL_Y] = pupils[0].y;
			values[j][RIGHT_PUPIL_X] = pupils[1].x;
			values[j][RIGHT_PUPIL_Y] = pupils[1].y;
			values[j][LEFT_GAZE_X] = gazes[0].x;
			values[j][LEFT_GAZE_Y] = gazes[0].y;
			values[j][LEFT_GAZE_Z] = gazes[0].z;
			values[j][RIGHT_GAZE_X] = gazes[1].x;
			values[j][RIGHT_GAZE_Y] = gazes[1].y;
			values[j][RIGHT_GAZE_Z] = gazes[1].z;
			values[j][FACE_WIDTH] = PointCalcUtil::distanceBetween(int(pts[0]), int(pts[1]), int(pts[32]), int(pts[33]));
			values[j][POINT_ID] = (float)j;
			values[j][POINT_TOTAL] = (float)n;
			values[j][HEAD_3D_POS_X] = hx;
			values[j][HEAD_3D_POS_Y] = hy;
			values[j][HEAD_3D_POS_Z] = hz;
			values[j][HEAD_3D_POS_CC_X] = hxCC;
			values[j][HEAD_3D_POS_CC_Y] = hyCC;
			values[j][HEAD_3D_POS_CC_Z] = hzCC;
			values[j][FACE_SCALE] = faceScale;

			values[j][FACE_RECT_X] = fx;
			values[j][FACE_RECT_Y] = fy;

			values[j][FACE_RECT_WIDTH] = fw;
			values[j][FACE_RECT_HEIGHT] = fh;

			values[j][LIP_BOTTOM_CENTER_X] = pts[114];
			values[j][LIP_BOTTOM_CENTER_Y] = pts[115];

			// rawPoints set--
			rawPoints[j][RAW_POINTS] = pts;
			rawPoints[j][RAW_3D_POINTS] = pts3d;
		}
	}

	result = keypoints(frame.cols, frame.rows, rawPoints, values, confidence);

	return result;
}

void FaceScanner2::scale(double newscale)
{
	activescale = newscale;
}

std::vector<TrackingTarget> FaceScanner2::keypoints(
	int frameCols,
	int frameRows,
	const float* rawPoints[MAX_TARGETS_COUNT][2],
	float trackerValues[MAX_TARGETS_COUNT][NUMBER_OF_PRESET_VALUES],
	const float* confidence[MAX_TARGETS_COUNT]
)
{
	std::vector<TrackingTarget> fxModelArr;

	for (int id = 0; id < MAX_TARGETS_COUNT; id++)
	{

		if (trackerValues[id][POINT_ID] >= 0)
		{
			float imgWidth = frameCols;
			float imgHeight = frameRows;
			float widthDiff = 1.0f;
			float heightDiff = 1.0f;

			if (imgWidth != Resolutions::OUTPUT_WIDTH) {
				widthDiff = Resolutions::OUTPUT_WIDTH / imgWidth;
			}

			if (imgHeight != Resolutions::OUTPUT_HEIGHT) {
				heightDiff = Resolutions::OUTPUT_HEIGHT / imgHeight;
			}

			// calculate x and y face center values.
			float xValue = (float)-(1 - ((trackerValues[id][FACE_CENTER_X]) / (imgWidth / 2)));
			float yValue = (float)(1 - ((trackerValues[id][FACE_CENTER_Y]) / (imgHeight / 2)));

			// calculate x and y face center values.
			float lipsXValue = (float)-(1 - ((trackerValues[id][TOP_LIP_CENTER_X]) / (imgWidth / 2)));
			float lipsYValue = (float)(1 - ((trackerValues[id][TOP_LIP_CENTER_Y]) / (imgHeight / 2)));

			float facewidth = trackerValues[id][FACE_WIDTH] / imgWidth;

			float viewerSize = (facewidth * 2) * widthDiff;

			int pitch = (int)trackerValues[id][PITCH];
			int yaw = 0 - (int)trackerValues[id][YAW];
			int roll = 0 - (int)trackerValues[id][ROLL];

			TrackingTarget fxModel = TrackingTarget();
			fxModel.pointId = id;
			fxModel.pointTotal = trackerValues[id][POINT_TOTAL];
			std::copy(&rawPoints[id][RAW_POINTS][0], &rawPoints[id][RAW_POINTS][66 * 2], &fxModel.pts[0]);
			std::copy(&rawPoints[id][RAW_POINTS][0], &rawPoints[id][RAW_POINTS][66 * 3], &fxModel.pts3d[0]);

			fxModel.width = facewidth;
			fxModel.widthRaw = trackerValues[id][FACE_WIDTH];
			fxModel.xCenter = xValue;
			fxModel.yCenter = yValue;

			fxModel.xCenterRaw = trackerValues[id][FACE_CENTER_X];
			fxModel.yCenterRaw = trackerValues[id][FACE_CENTER_Y];
			fxModel.pitch = pitch;
			fxModel.roll = roll;
			fxModel.yaw = yaw;
			fxModel.frameWidth = frameCols;
			fxModel.frameHeight = frameRows;
			fxModel.rect = cv::Rect(trackerValues[id][FACE_RECT_X], trackerValues[id][FACE_RECT_Y], trackerValues[id][FACE_RECT_WIDTH], trackerValues[id][FACE_RECT_HEIGHT]);
			fxModel.inited = true;

			fxModelArr.push_back(fxModel);
		}


	}

	return fxModelArr;

}
