#ifndef FACE_H
#define FACE_H

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/opencv.hpp>  

#include "facedetect-dll.h"
#include "FaceRecognize.h"

using namespace std;
using namespace cv;

vector<float> ExtractFeature_(Mat iptImage);

vector<Mat> FaceDetect_(Mat iptImage);//多人脸检测

vector<Rect> FaceDetect_(Mat iptImage, int flag);//多人脸检测

Mat FaceDetect_(Mat iptImage, int flag, int flags);//单人脸检测

Mat Vector2dToMat(vector<float> feature);

void SaveMat(Mat &saveFeature, const string& filename);

void SaveName(const string& name, const string& filename);

float cosine(const vector<float>& v1, const vector<float>& v2);

float dotProduct(const vector<float>& v1, const vector<float>& v2);

float module(const vector<float>& v);

vector<string> LoadName(const string& filename);

Mat LoadMat(const string& file);//文件名

vector<float> Mat2vector(Mat &FaceMatrix_mat);

vector<vector<float> > LoadFaceMatrix(vector<string> NameVector);

char*  FR_CreateGUID(int guid);

vector<vector<float> >  get_vector();


#endif