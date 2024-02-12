#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#define _USE_MATH_DEFINES
#include <math.h>

#include <Graphics/GraphicsModel.h>
#include <FX.h>
#include "FxModDepthMask.h"

using namespace Eigen;

class FxWidget3DAnimated : public GraphicsModel
{
public:
	static const std::string TYPE_NAME;
	std::string getTypeName() override;

	bool SequenceLooped = true;
	bool backClipping;
	float Xclip;
	float Yclip;
	float Zclip;
	std::vector<std::string> materialList;
	std::vector<std::string> modelList;

	std::string modelPath;
	std::vector<std::string> sequenceModels;
	std::vector<std::string> sequenceMaterials;
	std::vector<bool> sequenceParticles;
	std::vector<Eigen::Matrix3f> sequenceRotation;
	std::vector<float> sequenceScale;
	std::vector<Eigen::Vector3f> sequenceShift;
	std::vector<float> sequenceDuration;

	std::string materialPath;

	std::array<float, 3> bmin = { 0, 0, 0 };
	std::array<float, 3> bmax = { 0, 0, 0 };

	Eigen::Matrix3f extraRotateMatrix = Eigen::Matrix3f::Identity();

	float modelScale = 1;

	Eigen::Vector3f modelShift = { 0, 0, 0 };

	std::array<std::vector<std::string>, MAX_TO_TRACK> texturesPaths;
	std::array<std::vector<GLuint>, MAX_TO_TRACK> texturesIDs;

	std::array<ValueSmoother, MAX_TO_TRACK> pivotX;
	std::array<ValueSmoother, MAX_TO_TRACK> pivotY;
	Eigen::Vector3d pivotOffset;

	FxWidget3DAnimated();

	void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately) override;

	void loadFromJSON(boost::property_tree::ptree& modelRecord) override;
	boost::property_tree::ptree getPTree(ExtraModelData &data) override;

	void prepareSuitForJSON(boost::property_tree::ptree &suit, ExtraModelData& modelData) override;

	void onInputFrameResized();

	bool load() override;
	void transform(TrackingTarget& fxModel, ExternalRenderParams &externalRenderParams) override;
	void draw(TrackingTarget& fxModel, ExternalRenderParams &externalRenderParams) override;
	void unload() override;
	std::shared_ptr<ObjectRenderParams> createDefaultObjectRenderParams() override;
	void setDefaultObjectRenderParams(ObjectRenderParams& params) override;

	bool loadModels();
	void sortMeshesInZAscending();

	int maxMovement = 9;
	void animateNextStep(size_t i);

	std::array<int, MAX_TO_TRACK> animationCounter;
	std::array<bool, MAX_TO_TRACK> animationDirection;
	std::array<bool, MAX_TO_TRACK> animationPaused;
	std::array<float, MAX_TO_TRACK> animationStepDuration;

	std::mutex animationMutex;
};
