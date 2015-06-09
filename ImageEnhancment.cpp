
//Sami Dalati
//to compile run /Applications/CMake.app/Contents/bin/cmake . to make the make file

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

Mat Y, Cr, Cb;

float _MAX(float a, float b, float c)
{
	if(a >= b && a >= c)
		return a;
	if(b >= a && b >= c)
		return b;
	if(c >= a && c >= b)
		return c;
	return 0;
}

float _MIN(float a, float b, float c)
{
	if(a <= b && a <= c)
		return a;
	if(b <= a && b <= c)
		return b;
	if(c <= a && c <= b)
		return c;
	return 0;
}

double threeway_max(double a, double b, double c) {
    return max(a, max(b, c));
}

double threeway_min(double a, double b, double c) {
    return min(a, min(b, c));
}

Mat show_histogram(std::string const& name, cv::Mat1b const& image)
{
    // Set histogram bins count
    int bins = 256;
    int histSize[] = {bins};
    // Set ranges for histogram bins
    float lranges[] = {0, 256};
    const float* ranges[] = {lranges};
    // create matrix for histogram
    cv::Mat hist;
    int channels[] = {0};

    // create matrix for histogram visualization
    int const hist_height = 256;
    cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);

    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

    double max_val=0;
    minMaxLoc(hist, 0, &max_val);

    // visualize each bin
    for(int b = 0; b < bins; b++) {
        float const binVal = hist.at<float>(b);
        int   const height = cvRound(binVal*hist_height/max_val);
        cv::line( hist_image, cv::Point(b, hist_height-height), cv::Point(b, hist_height), cv::Scalar::all(255));
    }
    cv::imshow(name, hist_image);

    return hist;
}



//compare the histogram of two color frame, frames must be in RGB but the comparison is done in HSV
double HistCompare(Mat f1, Mat f2, int compare_method)
{
	Mat src_base, hsv_base;
    Mat src_Next, hsv_Next;
    char buffer [50];

    src_base = f1;
    src_Next = f2;
    sprintf (buffer, "RGB_Frame");
    namedWindow(buffer,1);
    imshow(buffer, src_base);
    
    //should this be done in color?
    cvtColor( src_base, hsv_base, COLOR_BGR2HSV );
    cvtColor( src_Next, hsv_Next, COLOR_BGR2HSV );

    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };

    const float* ranges[] = { h_ranges, s_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0, 1 };


    /// Histograms
    MatND hist_Src;
    MatND hist_Next;

    /// Calculate the histograms for the HSV images
    calcHist( &hsv_base, 1, channels, Mat(), hist_Src, 2, histSize, ranges, true, false );
    normalize( hist_Src, hist_Src, 0, 1, NORM_MINMAX, -1, Mat() );

    calcHist( &hsv_Next, 1, channels, Mat(), hist_Next, 2, histSize, ranges, true, false );
    normalize( hist_Next, hist_Next, 0, 1, NORM_MINMAX, -1, Mat() );

    /// Apply the histogram comparison methods
    return compareHist( hist_Src, hist_Next, compare_method );

}

// this function was the main function at the first attempt. it takes 10 frames, build hisogram for those frames, 
// compare the histogram of fram n with n+1. display value
// note frames are gray.

int histAnalyses()
{
	VideoCapture cap("video/CIS-sample.MOV"); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    int fr_cnt = 0;
    int fr_start = 10;
    const int numOfFrame = 6;
    Mat gray[numOfFrame], edges[numOfFrame], RGB[numOfFrame];
    Mat grayTemp;
    Mat frame;
    char* window[numOfFrame];
	int fr_num = 0;
    namedWindow("original",1);
    for(int i=0;i<=fr_start + numOfFrame;i++)
    {
    	fr_cnt++;
    
        cap >> frame; // get a new frame from camera
        cvtColor(frame, grayTemp, CV_BGR2GRAY);
        imshow("original", grayTemp);
        if(fr_cnt > fr_start && fr_cnt < fr_start + numOfFrame)
        {
        	//fr_cnt - fr_start;
        	gray[fr_num] = grayTemp.clone();
        	RGB[fr_num] = frame.clone();
        	GaussianBlur(gray[fr_num], gray[fr_num], Size(7,7), 1.5, 1.5);
	        Canny(gray[fr_num], edges[fr_num], 0, 30, 3);
	        char buffer [50];
		    sprintf (buffer, "Frame %d", fr_num);
		    window[fr_num] = buffer;
		    //strcpy(window[fr_num], buffer);
	        namedWindow(window[fr_num],1);
	        imshow(window[fr_num], edges[fr_num]);

	        //CvHistogram* hist;
	        //cvCalcPGH(edges, hist);
	        sprintf (buffer, "HistFrame %d", fr_num);
	        show_histogram(buffer, gray[fr_num]);

	        fr_num++;
	        //cout<<fr_num<<endl;
 
        }

        if(waitKey(30) >= 0) break;
    }

    for(int i=0;i<numOfFrame-2;i++)
    	{
    		for( int j = 0; j < 4; j++ )
		    {
		        double compResult = HistCompare(RGB[i], RGB[i+1], j);
		        printf( " Method [%d] Result : %f \n", j, compResult);
		    }
		    cout<<endl;  
		}

	waitKey(0);   
	return 0;
}

///dont do edge map cause you loose the blur data
//use gradiant instead
//calc gradiant map 
//normalize gradiant values (0 to 255)
//histogram
//compare hsitogram
//this Bhattacharyya is promising
// look at te difference, when the difference gets large then we have a blur


//for the gradiant: should i do a sobel or laplace? try both, laplacian is cheaper.
//should i use color instead of gray? gray 
// should i perform gaussianblur before applying sobel? no need to gaussian blur 

int GradiantCompare()
{

	VideoCapture cap("video/CIS-sample.MOV"); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    int fr_cnt = 0;
    int fr_start = 10;
    const int numOfFrame = 10;
    Mat gray[numOfFrame], edges[numOfFrame], RGB[numOfFrame], grad[numOfFrame], hist[numOfFrame];
    Mat grayTemp;
    Mat frame;
    char* window[numOfFrame];
    double compResult[numOfFrame];
	int fr_num = 0;

	int scale = 1;
  	int delta = 0;
  	int ddepth = CV_16S;


    namedWindow("original",1);
    for(int i=0;i<=fr_start + numOfFrame;i++) //loop through video
    {
    	fr_cnt++;
    
        cap >> frame; // get a new frame from camera
        cvtColor(frame, grayTemp, CV_BGR2GRAY);
        imshow("original", grayTemp);
        if(fr_cnt > fr_start && fr_cnt <= fr_start + numOfFrame)
        {
        	//fr_cnt - fr_start;
        	gray[fr_num] = grayTemp.clone();
        	RGB[fr_num] = frame.clone();
        	//GaussianBlur(gray[fr_num], gray[fr_num], Size(7,7), 1.5, 1.5, BORDER_DEFAULT);
	        //GaussianBlur( gray[fr_num], gray[fr_num], 0, 0, BORDER_DEFAULT );

        	  /// Generate grad_x and grad_y
			Mat grad_x, grad_y;
			Mat abs_grad_x, abs_grad_y;

			/// Gradient X
			//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
			Sobel( gray[fr_num], grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
			convertScaleAbs( grad_x, abs_grad_x ); //convert our partial results back to CV_8U // yet the values range from - to +
			normalize(grad_x, grad_x, 0, 255, NORM_MINMAX); //map values from 0 to 255

			/// Gradient Y
			//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
			Sobel( gray[fr_num], grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
			convertScaleAbs( grad_y, abs_grad_y ); //convert our partial results back to CV_8U
			normalize(grad_y, grad_y, 0, 255, NORM_MINMAX);

			/// Total Gradient (approximate)
			addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad[fr_num] );

			//imshow( window_name, grad );

	        char buffer [50];
		    sprintf (buffer, "Frame %d", fr_num);
		    window[fr_num] = buffer;
		    //strcpy(window[fr_num], buffer);
	        namedWindow(window[fr_num],1);
	        imshow(window[fr_num], grad[fr_num]);

	        //CvHistogram* hist;
	        //cvCalcPGH(edges, hist);
	        sprintf (buffer, "HistFrame %d", fr_num);
	        hist[fr_num] = show_histogram(buffer, grad[fr_num]);

	        fr_num++;
	        //cout<<fr_num<<endl;
 
        }

        if(waitKey(30) >= 0) break;
    }


    int j = CV_COMP_BHATTACHARYYA;
	compResult[0] = compareHist(hist[0], hist[1], j);

    for(int i=1;i<numOfFrame-1;i++)
    {
		//compare histograms
		//for( int j = 0; j < 4; j++ )
	    //{	
	    	compResult[i] = compareHist(hist[i], hist[i+1], j);
	        printf( " Method [%d] Result : %f Diff: %f \n", j, compResult[i], abs(compResult[i] - compResult[i-1]));
	   // }
	}

	waitKey(0);
	return 0;
}


//count the number of edges, the highest edge number is the clear one.
//three frames were tampered to add motion blur
//frame 3: 30 degree, 10 pixel
//frame 5: 30 degree, 20 pixel
//frame 7: 30 degree, 30 pixel

int ContoursCnt(Mat frame, int fr_num, int displayContour = 0)
{

    Mat grayTemp;

    cvtColor(frame, grayTemp, CV_BGR2GRAY);
    Canny(grayTemp, grayTemp, 100, 300, 3);
    /*Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    where the arguments are:

    detected_edges: Source image, grayscale
    detected_edges: Output of the detector (can be the same as the input)
    lowThreshold: The value entered by the user moving the Trackbar
    highThreshold: Set in the program as three times the lower threshold (following Canny’s recommendation)
    kernel_size: We defined it to be 3 (the size of the Sobel kernel to be used internally)*/

     /// Find contours   
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    RNG rng(12345);

    findContours( grayTemp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Draw contours: //drawing is not necessary, it is here for visual aid
    //**********************************************************************
    if(displayContour){
         Mat drawing = Mat::zeros( grayTemp.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
        }     
        char buffer [50];
        sprintf (buffer, "ContoursCnt %d.tiff", fr_num);
        imshow( buffer, drawing );
        imwrite(buffer, drawing);

        //**********************************************************************
    }
   
    return contours.size();
}

int main(int argc, char** argv)
{

    const int numOfFrame = 10;
    Mat frame, prev_frame;
    int startFrame = 175;
    int currentContourSize = 0;
    int lastContourSize = 0;
    int bestFrameNum = 0;
    char* window[numOfFrame];
    //int fr_num = 0;

    prev_frame = Mat::zeros(1, 1, CV_64F);//not sure about Mat type?

    for(int fr_num=startFrame;fr_num<numOfFrame+startFrame;fr_num++)//looping through frames, the frames were extracted from the video using ffmpeg
        //ffmpeg -i Cisfinal5-2.m4v -vf fps=30 out%d.png
        //ffmpeg -i Cisfinal5-2.m4v -pix_fmt yuv420p -vframes 30 out%d.png //does not work
        //ffmpeg -i Cisfinal5-2.m4v -bsf:v mjpeg2jpeg out%d.jpg//still encoded
    {   
        char buffer [50];
        sprintf (buffer, "video/out%d.png", fr_num);
        window[fr_num-startFrame] = buffer;
        frame = imread(buffer, CV_LOAD_IMAGE_COLOR);   // Read the file
        //if(countNonZero(prev_frame) < 1)
         //   prev_frame = frame;

        if(! frame.data )                              // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }

        sprintf (buffer, "Original %d.tiff", fr_num);
        namedWindow( window[fr_num-startFrame], WINDOW_AUTOSIZE );// Create a window for display.
        imshow( window[fr_num-startFrame], frame ); 
        imwrite(buffer, frame);


        /*********************************************************/ //this block is used for contours counting
        currentContourSize = ContoursCnt(frame, fr_num, 1); //find best frame based on contour counting
        cout<<"frame "<<fr_num<<" # of contours: "<<currentContourSize;
        
        if(currentContourSize > lastContourSize)
        {
            bestFrameNum = fr_num;
            lastContourSize = currentContourSize;
        }
        /*********************************************************/


        Mat diff = frame - prev_frame; //take frame diff
        sprintf (buffer, "Diff %d.tiff", fr_num);
        namedWindow( buffer, WINDOW_AUTOSIZE );// Create a window for display.
        imshow( buffer, diff); 
        imwrite(buffer, diff);

        Scalar mean;
        Scalar stddev;

        meanStdDev ( diff, mean, stddev ); //compute mean and std div 

        cout<<" mean: "<<mean.val[0]<<" STDdiv: "<<stddev.val[0]<<endl; //display the mean and std div for the first channel, note that the values for all channels were simialr

        /*********************************************************/
        //cout<<" norm: "<<norm(frame, prev_frame, NORM_L2)<<endl;

        prev_frame = frame;


    }
    
    cout<<"best frame: "<<bestFrameNum<<endl;

    waitKey(0);
    return 0;
}

/*
//canny: any recommendations about the threshold value?
//canny: best kernel size (3)?
//check if doing seprably, then it does not matter,
//otherwise, the bigger the better

 
//2nd: difference of the frames, build edge map of the difference 
//start with a clear vifdeo, fake some motino blur.
//do frames instead of video.


Prev Finding:
Contour counting works great for motion blur (software added  using CC) but it performed pretty bad when trying to detect best frame
that have digital noise distortion, this kind of distortion added blocks of pixelation that add edges, thus 
the contour count is higher for worst frames.


//Attempted:
Objective: find the worst frame, run both algorithm to detect worst and best frames in parallel and compare results

subratct fr(n) from fr(n-1)
-compute mean and std div
-the highest mean / std div is the worst frame (since consecutive good frames wont be that much different from each other
this means that their difference would be minimal)
-comare result with the result from countor counting.

here are the results:

frame 720 # of contours: 1543 mean: 107.107 STDdiv: 51.7164
frame 721 # of contours: 1557 mean: 10.589 STDdiv: 21.2054
frame 722 # of contours: 1659 mean: 10.5724 STDdiv: 21.2322     //good
frame 723 # of contours: 212 mean: 10.6699 STDdiv: 20.1694      //fake motion blur is added to this fram
frame 724 # of contours: 1442 mean: 10.4136 STDdiv: 20.3576     //good
frame 725 # of contours: 1862 mean: 5.58842 STDdiv: 14.2062     //worst
frame 726 # of contours: 1741 mean: 13.3808 STDdiv: 25.091
frame 727 # of contours: 1640 mean: 10.9518 STDdiv: 21.7349
frame 728 # of contours: 2017 mean: 5.27885 STDdiv: 13.9512     //worst
frame 729 # of contours: 1993 mean: 12.9161 STDdiv: 24.6727
frame 730 # of contours: 1711 mean: 10.5785 STDdiv: 21.5917     //ok


frame 175 # of contours: 2139 mean: 103.648 STDdiv: 52.2474
frame 176 # of contours: 2097 mean: 3.72111 STDdiv: 9.84544     //good
frame 177 # of contours: 2181 mean: 3.83103 STDdiv: 10.4603
frame 178 # of contours: 2185 mean: 6.72734 STDdiv: 17.0688     //good
frame 179 # of contours: 2128 mean: 1.0092 STDdiv: 4.16677      //good
frame 180 # of contours: 2252 mean: 6.39944 STDdiv: 17.1274     //halve bad
frame 181 # of contours: 2741 mean: 5.12643 STDdiv: 13.6313     //worst
frame 182 # of contours: 2816 mean: 4.24444 STDdiv: 13.0369     //worst
frame 183 # of contours: 2591 mean: 10.1657 STDdiv: 22.6364     //ok
frame 184 # of contours: 2359 mean: 1.12543 STDdiv: 4.76311
best frame: 182




Todo:
- investigate sbtract() instead of -
- normalize after subtract or absolute of diff
- countcontour on diff 


- take edge map or (or take gradiant then threshold)
- difference of edge map
- contour cnt (higher count of diff means worst frame)

(if frame n is bad then frame n+1 is also bad if it shows low contour cnt for diff!!)


- plot the contour cnt over 200 frames (gaussian dist) use matlab!!










*/

