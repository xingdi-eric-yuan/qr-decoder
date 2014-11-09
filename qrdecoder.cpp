#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <iostream>
#define CONT vector<Point>

using namespace cv;
using namespace std;

struct FinderPattern{
    Point topleft;
    Point topright;
    Point bottomleft;
    FinderPattern(Point a, Point b, Point c) : topleft(a), topright(b), bottomleft(c) {}
};

bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i > j );
}

Point getContourCentre(CONT& vec){
    double tempx = 0.0, tempy = 0.0;
    for(int i=0; i<vec.size(); i++){
        tempx += vec[i].x;
        tempy += vec[i].y;
    }
    return Point(tempx / (double)vec.size(), tempy / (double)vec.size());
}

bool isContourInsideContour(CONT& in, CONT& out){
    for(int i = 0; i<in.size(); i++){
        if(pointPolygonTest(out, in[i], false) <= 0) return false;
    }
    return true;
}

vector<CONT > findLimitedConturs(Mat contour, float minPix, float maxPix){
    vector<CONT > contours;
    vector<Vec4i> hierarchy;
    findContours(contour, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    cout<<"contours.size = "<<contours.size()<<endl;
    int m = 0; 
    while(m < contours.size()){
        if(contourArea(contours[m]) <= minPix){
            contours.erase(contours.begin() + m);
        }else if(contourArea(contours[m]) > maxPix){
            contours.erase(contours.begin() + m);
        }else ++ m;
    }
    cout<<"contours.size = "<<contours.size()<<endl;
    return contours;
}

vector<vector<CONT > > getContourPair(vector<CONT > &contours){
    vector<vector<CONT > > vecpair;
    vector<bool> bflag(contours.size(), false);

    for(int i = 0; i<contours.size() - 1; i++){
        if(bflag[i]) continue;
        vector<CONT > temp;
        temp.push_back(contours[i]);
        for(int j = i + 1; j<contours.size(); j++){
            if(isContourInsideContour(contours[j], contours[i])){
                temp.push_back(contours[j]);
                bflag[j] = true;
            }
        }
        if(temp.size() > 1){
            vecpair.push_back(temp);
        }
    }
    bflag.clear();
    for(int i=0; i<vecpair.size(); i++){
        sort(vecpair[i].begin(), vecpair[i].end(), compareContourAreas);
    }
    return vecpair;
}

void eliminatePairs(vector<vector<CONT > >& vecpair, double minRatio, double maxRatio){
    cout<<"maxRatio = "<<maxRatio<<endl;
    int m = 0; 
    bool flag = false;
    while(m < vecpair.size()){
        flag = false;
        if(vecpair[m].size() < 3){
            vecpair.erase(vecpair.begin() + m);
            continue;
        }
        for(int i=0; i<vecpair[m].size() - 1; i++){
            double area1 = contourArea(vecpair[m][i]);
            double area2 = contourArea(vecpair[m][i + 1]);
            if(area1 / area2 < minRatio || area1 / area2 > maxRatio){
                vecpair.erase(vecpair.begin() + m);
                flag = true;
                break;
            }
        }
        if(!flag){
            ++ m;
        }
    }
    if(vecpair.size() > 3){
        eliminatePairs(vecpair, minRatio, maxRatio * 0.9);
    }
}


double getDistance(Point a, Point b){
    return sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2));
}

FinderPattern getFinderPattern(vector<vector<CONT > > &vecpair){
    Point pt1 = getContourCentre(vecpair[0][vecpair[0].size() - 1]);
    Point pt2 = getContourCentre(vecpair[1][vecpair[1].size() - 1]);
    Point pt3 = getContourCentre(vecpair[2][vecpair[2].size() - 1]);
    double d12 = getDistance(pt1, pt2);
    double d13 = getDistance(pt1, pt3);
    double d23 = getDistance(pt2, pt3);
    double x1, y1, x2, y2, x3, y3;
    double Max = max(d12, max(d13, d23));
    Point p1, p2, p3;
    if(Max == d12){
        p1 = pt1;
        p2 = pt2;
        p3 = pt3;
    }else if(Max == d13){
        p1 = pt1;
        p2 = pt3;
        p3 = pt2;
    }else if(Max == d23){
        p1 = pt2;
        p2 = pt3;
        p3 = pt1;
    }
    x1 = p1.x;
    y1 = p1.y;
    x2 = p2.x;
    y2 = p2.y;
    x3 = p3.x;
    y3 = p3.y;
    if(x1 == x2){
        if(y1 > y2){
            if(x3 < x1){
                return FinderPattern(p3, p2, p1);
            }else{
                return FinderPattern(p3, p1, p2);
            }
        }else{
            if(x3 < x1){
                return FinderPattern(p3, p1, p2);
            }else{
                return FinderPattern(p3, p2, p1);
            }
        }
    }else{
        double newy = (y2 - y1) / (x2 - x1) * x3 + y1 - (y2 - y1) / (x2 - x1) * x1;
        if(x1 > x2){
            if(newy < y3){
                return FinderPattern(p3, p2, p1);
            }else{
                return FinderPattern(p3, p1, p2);
            }
        }else{
            if(newy < y3){
                return FinderPattern(p3, p1, p2);
            }else{
                return FinderPattern(p3, p2, p1);
            }
        }
    }
}


int main()
{
    //Mat ori=imread("tshirt.png");
    Mat ori=imread("bigbook.jpg");
    Mat gray;
    cvtColor (ori,gray,CV_BGR2GRAY);

    Mat pcanny;
    gray.copyTo(pcanny);
    Canny( pcanny, pcanny, 50, 150, 3 );

    Mat bin;
    threshold(gray, bin, 0, 255, CV_THRESH_OTSU);
    Mat contour;
    bin.copyTo(contour);

    vector<CONT > contours;
    contours = findLimitedConturs(contour, 8.00, 0.2 * ori.cols * ori.rows);

/*
    Mat drawing;
    ori.copyTo(drawing);
    for( int i = 0; i< contours.size(); i++ ){

        int r = (rand() + 125)%255;
        int g = (rand() + 32)%255;
        int b = (rand() + 87)%255;
       drawContours( drawing, contours, i, CV_RGB(r, g, b), 1);
    }
    imshow("contours", drawing);
*/
    if(!contours.empty()) sort(contours.begin(), contours.end(), compareContourAreas);
    vector<vector<CONT > > vecpair = getContourPair(contours);
    eliminatePairs(vecpair, 1.0, 10.0);
    cout<<"there are "<<vecpair.size()<<" pairs left!!"<<endl;

    FinderPattern fPattern = getFinderPattern(vecpair);
    cout<<"topleft = "<<fPattern.topleft.x<<", "<<fPattern.topleft.y<<endl
        <<"topright = "<<fPattern.topright.x<<", "<<fPattern.topright.y<<endl
        <<"bottomleft = "<<fPattern.bottomleft.x<<", "<<fPattern.bottomleft.y<<endl;
    Mat drawing;
    ori.copyTo(drawing);

    circle(drawing, fPattern.topleft, 3, CV_RGB(255,0,0), 2, 8, 0);
    circle(drawing, fPattern.topright, 3, CV_RGB(0,255,0), 2, 8, 0);
    circle(drawing, fPattern.bottomleft, 3, CV_RGB(0,0,255), 2, 8, 0);

    vector<Point2f> vecsrc;
    vector<Point2f> vecdst;
    vecsrc.push_back(fPattern.topleft);
    vecsrc.push_back(fPattern.topright);
    vecsrc.push_back(fPattern.bottomleft);
    vecdst.push_back(Point2f(20, 20));
    vecdst.push_back(Point2f(120, 20));
    vecdst.push_back(Point2f(20, 120));
    Mat affineTrans = getAffineTransform(vecsrc, vecdst);
    Mat warped;
    warpAffine(ori, warped, affineTrans, ori.size());
    Mat qrcode_color = warped(Rect(0, 0, 140, 140));
    Mat qrcode_gray;
    cvtColor (qrcode_color,qrcode_gray,CV_BGR2GRAY);
    Mat qrcode_bin;
    threshold(qrcode_gray, qrcode_bin, 0, 255, CV_THRESH_OTSU);

    imshow("binary", bin);
    imshow("canny", pcanny);
    imshow("finder patterns", drawing);
    imshow("binaried qr code", qrcode_bin);

    waitKey();
    return 0;
}

