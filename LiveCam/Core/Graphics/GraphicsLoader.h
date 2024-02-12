#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <array>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include <Eigen/Geometry>
#include <opencv2/imgcodecs.hpp>

#include <GLSL/GLSL.h>
#include <Common/CommonClasses.h>
#include <Graphics/GraphicsCommon.h>

namespace fs = boost::filesystem;

class GraphicsLoader {

public:

	static std::string GetBaseDir(const std::string &filepath);

	static bool FileExists(const std::string &abs_filename);

	static Eigen::Vector4f CalculateQuat(Eigen::Vector3f vecFrom, Eigen::Vector3f vecTo);

	static Eigen::Vector3f RotateByQuat(Eigen::Vector3f vec, Eigen::Vector4f quat);

	static void LoadModels(
		std::vector<std::string> &objFiles, 
		std::vector<GraphicsData>& drawObjects,
		std::array<float, 3>& bmin, 
		std::array<float, 3>& bmax, 
		float extraScale = 1.f, 
		Eigen::Vector3f extraShift = { 0, 0, 0 },
		Eigen::Matrix3f extraRotateMatrix = Eigen::Matrix3f::Identity(),
		std::vector<ObjectData>* outputBuffers = nullptr
	);

	static void LoadObj(std::vector<GraphicsData>& objects);

	static void LoadObjData(
		const std::string &file, 
		std::vector<ObjectData>& objDataArr, 
		std::array<float, 3>& bmin, 
		std::array<float, 3>& bmax, 
		float extraScale = 1.f, 
		Eigen::Vector3f extraShift = { 0, 0, 0 },
		Eigen::Matrix3f extraRotateMatrix = Eigen::Matrix3f::Identity()
	);

	static GraphicsData CreateQuadModel();

	static boost::property_tree::ptree convertMTLtoPTree(const std::string &path, const std::string &resourcesRoot);
	static void loadMaterial(std::string &objFile, GraphicsData& gData);
	static void loadMTLnames(std::vector<std::string> &objFiles, std::vector<GraphicsData>& drawObjects);
	static void loadObjectsNames(std::vector<std::string> &objFiles, std::vector<std::string>& objectsNames);
};
