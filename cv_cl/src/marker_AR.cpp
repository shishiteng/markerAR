//-----------------------------------------
//AR 基于标识，by zhuzhu 2015.11.18
//-----------------------------------------
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>  
#include <stdio.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include "MarkerDetector.hpp"

using namespace cv;
using namespace std;

static GLint imagewidth;  
static GLint imageheight;  
static GLint pixellength;  
static GLubyte* pixeldata;  
#define GL_BGR_EXT 0x80E0  

Matrix44 projectionMatrix;  
vector<Marker> m_detectedMarkers;  
GLuint defaultFramebuffer, colorRenderbuffer;  

MarkerDetector markerDetector;

int cameraNumber = 0;

Mat_<float>  camMatrix;
Mat_<float>  distCoeff;

void initCamera(VideoCapture &VideoCapture,int cameraNumber);
void readCameraParameter();
void readCameraParameter1();

void build_projection(Mat_<float> cameraMatrix)  
{  
    float near = 0.01;  // Near clipping distance  
    float far = 100;  // Far clipping distance  
  
    // Camera parameters  
    //float f_x = cameraMatrix.data[0]; // Focal length in x axis  
    //float f_y = cameraMatrix.data[4]; // Focal length in y axis (usually the same?)  
    //float c_x = cameraMatrix.data[2]; // Camera primary point x  
    //float c_y = cameraMatrix.data[5]; // Camera primary point y  

    float f_x = cameraMatrix(0,0); // Focal length in x axis  
    float f_y = cameraMatrix(1,1); // Focal length in y axis (usually the same?)  
    float c_x = cameraMatrix(0,2); // Camera primary point x  
    float c_y = cameraMatrix(1,2); // Camera primary point y  
  
    projectionMatrix.data[0] =  - 2.0 * f_x / imagewidth;  
    projectionMatrix.data[1] = 0.0;  
    projectionMatrix.data[2] = 0.0;  
    projectionMatrix.data[3] = 0.0;  
  
    projectionMatrix.data[4] = 0.0;  
    projectionMatrix.data[5] = 2.0 * f_y / imageheight;  
    projectionMatrix.data[6] = 0.0;  
    projectionMatrix.data[7] = 0.0;  
  
    projectionMatrix.data[8] = 2.0 * c_x / imagewidth - 1.0;  
    projectionMatrix.data[9] = 2.0 * c_y / imageheight - 1.0;      
    projectionMatrix.data[10] = -( far+near ) / ( far - near );  
    projectionMatrix.data[11] = -1.0;  
  
    projectionMatrix.data[12] = 0.0;  
    projectionMatrix.data[13] = 0.0;  
    projectionMatrix.data[14] = -2.0 * far * near / ( far - near );          
    projectionMatrix.data[15] = 0.0;  
}  
  
void setMarker(const vector<Marker>& detectedMarkers)  
{  
    m_detectedMarkers = detectedMarkers;  
}  
  
void display(void)  
{  
      
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);  
    //绘制图片,第一、二、三、四个参数表示图象宽度、图象高度、像素数据内容、像素数据在内存中的格式,最后一个参数表示用于绘制的像素数据在内存中的位置  
    glDrawPixels(imagewidth,imageheight,GL_BGR_EXT,GL_UNSIGNED_BYTE,pixeldata);  
    
    /* 
     glMatrixMode - 指定哪一个矩阵是当前矩阵 
      
     mode 指定哪一个矩阵堆栈是下一个矩阵操作的目标,可选值: GL_MODELVIEW、GL_PROJECTION、GL_TEXTURE. 
     说明 
     glMatrixMode设置当前矩阵模式: 
     GL_MODELVIEW,对模型视景矩阵堆栈应用随后的矩阵操作. 
     GL_PROJECTION,对投影矩阵应用随后的矩阵操作. 
     GL_TEXTURE,对纹理矩阵堆栈应用随后的矩阵操作. 
     与glLoadIdentity()一同使用 
     glLoadIdentity():该函数的功能是重置当前指定的矩阵为单位矩阵。 
     在glLoadIdentity()之后我们为场景设置了透视图。glMatrixMode(GL_MODELVIEW)设置当前矩阵为模型视图矩阵，模型视图矩阵储存了有关物体的信息。 
     */  
    //绘制坐标  ，导入相机内参数矩阵模型
    glMatrixMode(GL_PROJECTION);  
    glLoadMatrixf(projectionMatrix.data);  
    glMatrixMode(GL_MODELVIEW);  
    glLoadIdentity();  
    
    glEnableClientState(GL_VERTEX_ARRAY);  //启用客户端的某项功能
    glEnableClientState(GL_NORMAL_ARRAY);  
  
    glPushMatrix();  
    glLineWidth(3.0f);  
  
    float lineX[] = {0,0,0,1,0,0};  
    float lineY[] = {0,0,0,0,1,0};  
    float lineZ[] = {0,0,0,0,0,1};  
  
    const GLfloat squareVertices[] = {  
        -0.5f, -0.5f,  
        0.5f,  -0.5f,  
        -0.5f,  0.5f,  
        0.5f,   0.5f,  
    };  
    const GLubyte squareColors[] = {  
        255, 255,   0, 255,  
        0,   255, 255, 255,  
        0,     0,   0,   0,  
        255,   0, 255, 255,  
    };  
  
    for (size_t transformationIndex=0; transformationIndex<m_detectedMarkers.size(); transformationIndex++)  
    {  
        const Transformation& transformation = m_detectedMarkers[transformationIndex].transformation;  
        Matrix44 glMatrix = transformation.getMat44();  
        
        //导入相机外参数矩阵模型
        glLoadMatrixf(reinterpret_cast<const GLfloat*>(&glMatrix.data[0]));  //reinterpret_cast:任何类型的指针之间都可以互相转换,修改了操作数类型,仅仅是重新解释了给出的对象的比特模型而没有进行二进制转换
  
        glVertexPointer(2, GL_FLOAT, 0, squareVertices);  //指定顶点数组的位置，2表示每个顶点由三个量构成（x, y），GL_FLOAT表示每个量都是一个GLfloat类型的值。第三个参数0。最后的squareVertices指明了数组实际的位置。这个squareVertices是由第一个参数和要画的图形有几个顶点决定大小，理解。
        glEnableClientState(GL_VERTEX_ARRAY);  //表示启用顶点数组
        glColorPointer(4, GL_UNSIGNED_BYTE, 0, squareColors);  //RGBA颜色，四个顶点
        glEnableClientState(GL_COLOR_ARRAY);  //启用颜色数组
  
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);  
        glDisableClientState(GL_COLOR_ARRAY);  
  
        float scale = 0.5;  
        glScalef(scale, scale, scale);  
  
        glColor4f(1.0f, 0.0f, 0.0f, 1.0f);  
        glVertexPointer(3, GL_FLOAT, 0, lineX);  
        glDrawArrays(GL_LINES, 0, 2);  
  
        glColor4f(0.0f, 1.0f, 0.0f, 1.0f);  
        glVertexPointer(3, GL_FLOAT, 0, lineY);  
        glDrawArrays(GL_LINES, 0, 2);  
  
        glColor4f(0.0f, 0.0f, 1.0f, 1.0f);  
        glVertexPointer(3, GL_FLOAT, 0, lineZ);  
        glDrawArrays(GL_LINES, 0, 2);  
    }     
    glFlush();  
    glPopMatrix();  
  
    glDisableClientState(GL_VERTEX_ARRAY);    
  
    glutSwapBuffers();  
}  
  
int show(const char* filename,int argc, char** argv,Mat_<float>& cameraMatrix, vector<Marker>& detectedMarkers)  
{  
    //打开文件  
    FILE* pfile=fopen(filename,"rb");  
    if(pfile == 0) exit(0);  
    //读取图像大小  
    fseek(pfile,0x0012,SEEK_SET);  
    fread(&imagewidth,sizeof(imagewidth),1,pfile);  
    fread(&imageheight,sizeof(imageheight),1,pfile);  
    //计算像素数据长度  
    pixellength=imagewidth*3;  
    while(pixellength%4 != 0)pixellength++;  
    pixellength *= imageheight;  
    //读取像素数据  
    pixeldata = (GLubyte*)malloc(pixellength);  
    if(pixeldata == 0) exit(0);  
    fseek(pfile,54,SEEK_SET);  
    fread(pixeldata,pixellength,1,pfile);  
    //以上是读取一个bmp图像宽高和图像数据的操作
    //关闭文件  
    fclose(pfile);  
  
    build_projection(cameraMatrix);  //这是建立摄像机内参数矩阵，就是相机矩阵，display函数开始导入的模型就是相机矩阵
    setMarker(detectedMarkers);  //导入找到的标识
    //初始化glut运行  
    glutInit(&argc,argv);  
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);  
    glutInitWindowPosition(100,100);  
    glutInitWindowSize(imagewidth,imageheight);  
    glutCreateWindow(filename);  
    glutDisplayFunc(&display);  
    glutMainLoop();  
    //-------------------------------------  
    free(pixeldata);  
    return 0;  
}

int main(int argc,char *argv[])
{ 
  Mat src,dst;
  Mat grayscale,threshImg;
  vector< vector<Point> > contours;
  vector<Marker> markers;

  //设置摄像头或视频
  if(argc > 1)
    cameraNumber = atoi(argv[1]);

  //打开摄像头，检测是否打开
  VideoCapture camera;
  initCamera(camera,cameraNumber); 
  //设置窗口长宽比
  camera.set(CV_CAP_PROP_FRAME_WIDTH,640);
  camera.set(CV_CAP_PROP_FRAME_HEIGHT,480);
  
  //导入相机内参数
  readCameraParameter1();
  cout << camMatrix << endl;
  //while(true)
     {
      //camera >> src;
      src = imread("1.jpg");
      if(src.empty()){
          cerr << "ERROR: could not grab a camera frame!" << endl;
          exit(1);
      }

      markerDetector.processFrame(src,camMatrix, distCoeff,markers);
      
      imwrite("test.bmp",src);  
      show("test.bmp",argc,argv,camMatrix, markers);  
  
      char keypress =  waitKey(0) & 0xff;
       if(keypress == 27){
          //break;
          exit(1);
      }
     }
   return 0;
}

void initCamera(VideoCapture &VideoCapture,int cameraNumber)
{
  VideoCapture.open(cameraNumber);
  if(!VideoCapture.isOpened()){
      cerr << "ERROR: Could not access the camera!!!" << endl;
      //exit(1);
      return;
  }
  cout << "Loaded camera" << cameraNumber << "." << endl;
}

void readCameraParameter()
{
  camMatrix = Mat::eye(3, 3, CV_64F);
  distCoeff = Mat::zeros(8, 1, CV_64F);

  FileStorage fs("/home/zhu/program_c/opencv_study/AR_CV_GL/mastering_opencv/marker_AR/camera_calibration/out_camera_data.yml",FileStorage::READ);
  if (!fs.isOpened())
    {
      cout << "Could not open the configuration file!" << endl;
      exit(1);
    }
  fs["Camera_Matrix"] >> camMatrix;
  fs["Distortion_Coefficients"] >> distCoeff;
  fs.release();
  cout << camMatrix << endl;
  cout << distCoeff << endl;
}

void readCameraParameter1()
{
  //calibratoin data for iPad 2 
    camMatrix = Mat::eye(3, 3, CV_64F);
    distCoeff = Mat::zeros(8, 1, CV_64F); 

    camMatrix(0,0) = 6.24860291e+02 * (640./352.);  
    camMatrix(1,1) = 6.24860291e+02 * (480./288.);  
    camMatrix(0,2) = 640 * 0.5f; //640  
    camMatrix(1,2) = 480 * 0.5f; //480,我改的！牛逼不？！  
  
    for (int i=0; i<4; i++)  
        distCoeff(i,0) = 0;
}






