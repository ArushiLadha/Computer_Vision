// this calibration programs finds the P matrix using least squares estimation using SVD using all available matches.

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "cv.h"

using namespace std;
using namespace cv;


void do_decompose(Mat P)
{
	Mat K, R, t;
	decomposeProjectionMatrix(P,K,R,t);
	for(int i=0; i<4; i++)
		t.at<float>(i,0) /= t.at<float>(3,0); 
	cout << "K = " << K << endl << endl;
	cout << "R = " << R << endl << endl;
	cout << "t = " << t << endl << endl;
}

int proj_error(Mat wld_pt, Mat img_pt, Mat P, int np){
   int j;
   Mat proj = P*wld_pt;

   for(j=0; j<np; j++){
      proj.at<float>(0, j) /= proj.at<float>(2, j);
      proj.at<float>(1, j) /= proj.at<float>(2, j);
      proj.at<float>(2, j) = 1;
   }

   Mat orig_X(1, 3, CV_32FC1), est_X(1, 3, CV_32FC1);
   float error;

   int i, count = 0;
   float max = 0, max_error = 230, total = 0;

   for (i=0; i<np; i++){
      orig_X.at<float>(0, 0) = img_pt.at<float>(0, i);
      orig_X.at<float>(0, 1) = img_pt.at<float>(1, i);
      orig_X.at<float>(0, 2) = img_pt.at<float>(2, i);

      est_X.at<float>(0, 0) = proj.at<float>(0, i);
      est_X.at<float>(0, 1) = proj.at<float>(1, i);
      est_X.at<float>(0, 2) = proj.at<float>(2, i);

      error = norm(orig_X - est_X);
      total += error;
      if (error <230)
         count++;
      if (max < error)
         max = error;
   } 
   cout << "max error : " << max<< endl <<endl;
   cout << "total error : "<<total<<endl <<"mean error : " <<total/(float)np << endl;
   return count;
}
void inv_reproj_error(Mat wld_pt, Mat img_pt, Mat P, int np){
   Mat inv_P;
   invert(P, inv_P, DECOMP_SVD);
   Mat reproj = inv_P*img_pt;

   for(int j=0; j<np; j++){
      reproj.at<float>(0, j) /= reproj.at<float>(3, j);
      reproj.at<float>(1, j) /= reproj.at<float>(3, j);
      reproj.at<float>(2, j) /= reproj.at<float>(3, j);
      reproj.at<float>(3, j) = 1;
   }

   Mat orig_X(1, 4, CV_32FC1), est_X(1, 4, CV_32FC1);
   float error;

   int i, count = 0;
   float max = 0, max_error = 230, total = 0;

   for (i=0; i<np; i++){
      orig_X.at<float>(0, 0) = wld_pt.at<float>(0, i);
      orig_X.at<float>(0, 1) = wld_pt.at<float>(1, i);
      orig_X.at<float>(0, 2) = wld_pt.at<float>(2, i);
      orig_X.at<float>(0, 3) = wld_pt.at<float>(3, i);

      est_X.at<float>(0, 0) = reproj.at<float>(0, i);
      est_X.at<float>(0, 1) = reproj.at<float>(1, i);
      est_X.at<float>(0, 2) = reproj.at<float>(2, i);
      est_X.at<float>(0, 3) = reproj.at<float>(3, i);

      error = norm(orig_X - est_X);
      total += error;
   } 
   cout << "total re error : "<<total<<endl <<"mean re error : " <<total/(float)np << endl; 
}

int main(){
   int i,j;
   int imgNo, npoint;
   cin >> imgNo >> npoint;

   Mat img_pt(3, npoint, CV_32FC1), wld_pt(4, npoint, CV_32FC1);
   Mat A(npoint*2,12,CV_32FC1);
   Mat D, U, Vt;


   for (i=0; i<npoint; i++){
      cin>>img_pt.at<float>(0, i) >>img_pt.at<float>(1, i);
      cin>> wld_pt.at<float>(0, i)>> wld_pt.at<float>(1, i)>> wld_pt.at<float>(2, i);
      wld_pt.at<float>(3, i) = 1;
      img_pt.at<float>(2, i) = 1;

      float x, y, z, u, v;
      x = wld_pt.at<float>(0, i);
      y = wld_pt.at<float>(1, i);
      z = wld_pt.at<float>(2, i);
      u = img_pt.at<float>(0, i);
      v = img_pt.at<float>(1, i);

      A.at<float>(i*2,0) = x; A.at<float>(i*2,1) = y; A.at<float>(i*2,2) = z; A.at<float>(i*2,3) = 1;
      A.at<float>(i*2,4) = 0; A.at<float>(i*2,5) = 0; A.at<float>(i*2,6) = 0; A.at<float>(i*2,7) = 0;
      A.at<float>(i*2,8) = -1*x*u; A.at<float>(i*2,9) = -1*y*u; A.at<float>(i*2,10) = -1*z*u; A.at<float>(i*2,11) = -1*u;

      A.at<float>(i*2+1,0) = 0; A.at<float>(i*2+1,1) = 0; A.at<float>(i*2+1,2) = 0; A.at<float>(i*2+1,3) = 0;
      A.at<float>(i*2+1,4) = x; A.at<float>(i*2+1,5) = y; A.at<float>(i*2+1,6) = z; A.at<float>(i*2+1,7) = 1;
      A.at<float>(i*2+1,8) = -1*x*v; A.at<float>(i*2+1,9) = -1*y*v; A.at<float>(i*2+1,10) = -1*z*v; A.at<float>(i*2+1,11) = -1*v;
   }
   cout << "Final or_mat:\n"  << endl;
   cout << "in SVD compute\n";
   SVD::compute(A, D, U, Vt);
   cout<<"done svd compute\n";
//   cout << endl<<D<<endl <<endl;

   Mat P(3, 4, CV_32FC1);
   for(i=0; i< 12; i++){
      if(D.at<float>(i, 0) == 0)
         break;
   }

   i = i-1;
   for(j=0; j<12; j++){
      P.at<float>(j/4, j%4) = Vt.at<float>(i, j);
   }

   Mat proj = P*wld_pt;
   for(i=0; i<npoint; i++){
      proj.at<float>(0, i) /= proj.at<float>(2, i);
      proj.at<float>(1, i) /= proj.at<float>(2, i);
      proj.at<float>(2, i) = 1;
   }

   cout<< "Answer of linear solver: \n" << P << endl;  
   
   proj_error(wld_pt, img_pt, P, npoint);
   inv_reproj_error(wld_pt, img_pt, P, npoint);
   
   do_decompose(P);
}
