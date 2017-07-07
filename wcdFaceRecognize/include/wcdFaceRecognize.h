#ifndef WCDFACERECOGNIZE_H
#define WCDFACERECOGNIZE_H

#ifdef WCDFACERECOGNIZE_DLL_API_EXPORTS
#define DLL_API extern "C" __declspec(dllexport)
#else
#define DLL_API extern "C" __declspec(dllimport)
#endif

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <time.h>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/opencv.hpp>  

#include "facedetect-dll.h"
#include "FaceRecognize.h"


using namespace std;
using namespace cv;

#define DETECT_BUFFER_SIZE 0x20000

typedef unsigned char BYTE;

enum
{
	HIDE_IMAGE = 0,
	SHOW_IMAGE = 1
};



//每发现一个单独的人脸生成一个对应的FaceGUID
typedef bool(__stdcall * FR_FaceDetectCallBack)(const char* FaceGUID, BYTE* FacePhoto, int FacePhotoLen, int OutX, int OutY, int OutW, int OutH, int UserData);
//发现一个匹配的人脸时输出匹配源GUID和匹配源的图片，返回值是匹配度数值
typedef bool(__stdcall *FR_FaceVerifyCallBack)(const char* FaceGUID, const char* MatchGUID, BYTE* MatchPhoto, int MatchPhotoLen, int MatchDegree, int UserData);

//DLL初始化
DLL_API bool __stdcall FR_Init(FR_FaceDetectCallBack FFCB, FR_FaceVerifyCallBack MFCB, const string& path);
//DLL反初始化 
DLL_API bool __stdcall FR_Final();

//人脸校验参照表维护
DLL_API int __stdcall FR_AddVerifyTarget(const char* VTGUID, BYTE* VTPhoto, int VTPhotoLen);
DLL_API int __stdcall FR_DelVerifyTarget(const char* VTGUID);
DLL_API int __stdcall FR_ClearVerifyTarget();

//人脸检测：系统取得图像后调用FaceDetect输入RGB数据，FaceDetect分析发现人脸后返回人脸的数量。每发现一个人脸调用FaceDetectCallBack输出一次人脸数据。
DLL_API int __stdcall FR_FaceDetect(BYTE* AFrame, int PhotoLen, int UserData, int flag = HIDE_IMAGE);
//人脸检测：输入为BMP数据
DLL_API int __stdcall FR_FaceDetectSingle(BYTE* AFrame, int PhotoLen, BYTE* &AFace, int &FaceLen);
//人脸校验：将输入两张人脸图像匹配，返回
DLL_API int __stdcall FR_FaceVerify(BYTE* FaceAPhoto, int FaceAPhotoLen, BYTE* FaceBPhoto, int FaceBPhotoLen);
//人脸列表匹配：将输入的人脸图像与人脸校验参照表匹配，返回匹配中人脸的数量。每匹配中一个调用FaceVerifyCallBack输出一次匹配数据。
DLL_API int __stdcall FR_FaceListVerify(BYTE* FacePhoto, int FacePhotoLen, char* VTGUID, int UserData);

DLL_API int __stdcall FR_Multi_FaceListVerify(BYTE* FacePhoto, int FacePhotoLen, char* VTGUID, int UserData);//多人脸识别

//DLL测试
DLL_API bool __stdcall FR_Test();
//ps:所有函数中UserData为调用方输入值，触发回调时将此值原样输出。 

//初始化caffe框架
DLL_API void __stdcall FR_Caffe_init(const string& prototxt, const string& caffemodel);




#endif