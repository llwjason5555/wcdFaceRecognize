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



//ÿ����һ����������������һ����Ӧ��FaceGUID
typedef bool(__stdcall * FR_FaceDetectCallBack)(const char* FaceGUID, BYTE* FacePhoto, int FacePhotoLen, int OutX, int OutY, int OutW, int OutH, int UserData);
//����һ��ƥ�������ʱ���ƥ��ԴGUID��ƥ��Դ��ͼƬ������ֵ��ƥ�����ֵ
typedef bool(__stdcall *FR_FaceVerifyCallBack)(const char* FaceGUID, const char* MatchGUID, BYTE* MatchPhoto, int MatchPhotoLen, int MatchDegree, int UserData);

//DLL��ʼ��
DLL_API bool __stdcall FR_Init(FR_FaceDetectCallBack FFCB, FR_FaceVerifyCallBack MFCB, const string& path);
//DLL����ʼ�� 
DLL_API bool __stdcall FR_Final();

//����У����ձ�ά��
DLL_API int __stdcall FR_AddVerifyTarget(const char* VTGUID, BYTE* VTPhoto, int VTPhotoLen);
DLL_API int __stdcall FR_DelVerifyTarget(const char* VTGUID);
DLL_API int __stdcall FR_ClearVerifyTarget();

//������⣺ϵͳȡ��ͼ������FaceDetect����RGB���ݣ�FaceDetect�������������󷵻�������������ÿ����һ����������FaceDetectCallBack���һ���������ݡ�
DLL_API int __stdcall FR_FaceDetect(BYTE* AFrame, int PhotoLen, int UserData, int flag = HIDE_IMAGE);
//������⣺����ΪBMP����
DLL_API int __stdcall FR_FaceDetectSingle(BYTE* AFrame, int PhotoLen, BYTE* &AFace, int &FaceLen);
//����У�飺��������������ͼ��ƥ�䣬����
DLL_API int __stdcall FR_FaceVerify(BYTE* FaceAPhoto, int FaceAPhotoLen, BYTE* FaceBPhoto, int FaceBPhotoLen);
//�����б�ƥ�䣺�����������ͼ��������У����ձ�ƥ�䣬����ƥ����������������ÿƥ����һ������FaceVerifyCallBack���һ��ƥ�����ݡ�
DLL_API int __stdcall FR_FaceListVerify(BYTE* FacePhoto, int FacePhotoLen, char* VTGUID, int UserData);

DLL_API int __stdcall FR_Multi_FaceListVerify(BYTE* FacePhoto, int FacePhotoLen, char* VTGUID, int UserData);//������ʶ��

//DLL����
DLL_API bool __stdcall FR_Test();
//ps:���к�����UserDataΪ���÷�����ֵ�������ص�ʱ����ֵԭ������� 

//��ʼ��caffe���
DLL_API void __stdcall FR_Caffe_init(const string& prototxt, const string& caffemodel);




#endif