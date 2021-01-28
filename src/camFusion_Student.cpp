
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, string imgNumber, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = min(xwmin, xw);
            ywmin = min(ywmin, yw);
            ywmax = max(ywmax, yw);

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = min(top, y);
            left = min(left, x);
            bottom = max(bottom, y);
            right = max(right, x);

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-125, bottom+25), cv::FONT_ITALIC, 1, currColor);
        if (it1->lidarPoints.size() > 0)
            printf("xmin=%2.2f m, yw=%2.2f m\n", xwmin, ywmax-ywmin);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-125, bottom+63), cv::FONT_ITALIC, 1, currColor);  
    }

    // plot distance markers
    float lineSpacing = 0.1; //2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects " + imgNumber;
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Note, we've already associated keypoints and matches to bounding boxes when we were matching bounding boxes;
    // this just needs to go through those matches and remove outliers.
    // TODO: the plan is to 
    // 1. compute distances between pairs of keypoints,
    // 2. compute ratios of (current distance : previous distance) on matched pairs,
    // 3. find a "robust mean" (median) of those;
    // 4. remove points for which this is too far off from everything else.
    // The ideais that if we just have a translation + scaling (as we should), then a given point should get e.g. 2x as far
    // from every other point (except those other points that are themselves outliers, of course).
    // A non-outlier point will get 2x as far away from every other non-outlier point, so the median will be 2.
    // An outlier point will be a random distance away from other points, so the median will be something that's probably not 2.

    // early exit
    if (boundingBox.kptMatches.size() == 0)
        return;

    std::vector<double> distances;
    for (auto it = boundingBox.kptMatches.begin(); it != boundingBox.kptMatches.end(); ++it)
    {
        cv::Point2f ptPrev = kptsPrev[it->queryIdx].pt;
        cv::Point2f ptCurr = kptsCurr[it->queryIdx].pt;
        double dist = sqrt((ptCurr.x - ptPrev.x) * (ptCurr.x - ptPrev.x) + (ptCurr.y - ptPrev.y) * (ptCurr.y - ptPrev.y));
        distances.push_back(dist);
    }

    // compute quartiles and interquartile range
    // we don't care about the error introduced by even vs. odd sizes here
    sort(distances.begin(), distances.end());
    double quartile1 =  distances[floor(distances.size() / 4)], quartile3 = distances[floor(distances.size() * 3 / 4)];

    // we will use the definition that "outlier" means "1.5 interquartile ranges below the first quartile, or above the third quartile"
    // (see e.g. www.mathwords.com/o/outlier.htm)
    double outlierLowerBound = quartile1 - 1.5 * (quartile3 - quartile1);
    double outlierUpperBound = quartile3 + 1.5 * (quartile3 - quartile1);

    // iterating in reverse so that we can keep the same start pointer and not recreate every time
    auto start = boundingBox.kptMatches.begin();
    for (int i = distances.size() - 1; i >= 0; i--)
    {
        if (distances[i] < outlierLowerBound || distances[i] > outlierUpperBound)
        {
            boundingBox.kptMatches.erase(start + i);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // we will use the ratio of the means of pairwise distances between matched keypoints in the two frames
    // as an estimate of the relative distance to the vehicle
    std::vector<cv::Point2f> keypointsPrevBox, keypointsCurrBox;
    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        keypointsPrevBox.push_back(kptsPrev[it->queryIdx].pt);
        keypointsCurrBox.push_back(kptsCurr[it->trainIdx].pt);
    }

    double sumDistsPrev = 0, sumDistsCurr = 0;
    // note the number of keypoints is the same in previous box and in current, since they're matched
    int numDists = keypointsPrevBox.size() * (keypointsPrevBox.size() - 1) / 2;
    for (int i = 0; i < keypointsPrevBox.size() - 1; i++)
    {
        for (int j = i+1; j < keypointsPrevBox.size(); j++)
        {
            cv::Point2f pt1 = keypointsPrevBox[i];
            cv::Point2f pt2 = keypointsPrevBox[j];
            double dist = sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y));
            sumDistsPrev += dist;

            pt1 = keypointsCurrBox[i];
            pt2 = keypointsCurrBox[j];
            dist = sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y));
            sumDistsCurr += dist;
        }
    }

    double meanDistancePrev = sumDistsPrev / numDists;
    double meanDistanceCurr = sumDistsCurr / numDists;

    // Substitute into constant velocity model, see Lesson 3.3
    // Note, normally the distance between points in the Curr frame will be larger than in the Prev frame
    // (since the car is moving towards the vehicle and its apparent size is increasing).
    // If we detect that the apparent size is shrinking, the formula would have us report negative numbers;
    // it seems more reasonable to report a large positive value instead.
    if (meanDistanceCurr < meanDistancePrev)
    {
        TTC = 10000;
    }
    else
    {
        TTC = 1 / frameRate * 1 / (meanDistanceCurr / meanDistancePrev - 1);
    }
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // plan: discard closest 2 values, average the next 30 to get a distance estimate

    std::vector<double> prevXVals;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        // our projection should filter out points behind the camera, but let's make sure
        if (it->x > 0)
        {
            prevXVals.push_back(it->x);
        }
    }
    sort(prevXVals.begin(), prevXVals.end());

    std::vector<double> currXVals;
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        // our projection should filter out points behind the camera, but let's make sure
        if (it->x > 0)
        {
            currXVals.push_back(it->x);
        }
    }
    sort(currXVals.begin(), currXVals.end());

    // if we don't have enough points for estimation, return -1
    // this is a safe nonsense value
    if (prevXVals.size() < 2 || currXVals.size() < 2) {
        TTC = -1;
        return;
    }

    // previous distance will be the mean of points 2 through 31 in the previous frame
    // (or 2 through end if there's less than 32 points total)
    double prevDistSum = 0;
    int prevCount = 32 <= prevXVals.size() ? 32 : prevXVals.size();
    for (int i = 2; i < prevCount; i++)
    {
        prevDistSum += prevXVals[i];
    }
    double prevDistance = prevDistSum / prevCount;

    // current distance will be the mean of points 2 through 31 in the previous frame
    // (or 2 through end if there's less than 32 points total)
    double currDistSum = 0;
    int currCount = 32 <= currXVals.size() ? 32 : currXVals.size();
    for (int i = 2; i < currCount; i++)
    {
        currDistSum += currXVals[i];
    }
    double currDistance = currDistSum / currCount;

    double speed = (prevDistance - currDistance) * frameRate;
    // add a tiny amount to the speed to prevent division by 0
    // also, if speed is negative, replace it by 0
    TTC = currDistance / (max(speed,0.0) + 0.0001);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // early escapes
    if (matches.size() == 0 || prevFrame.boundingBoxes.size() == 0 || currFrame.boundingBoxes.size() == 0)
        return;

    // create and initialize the structure that will keep the counts of matches between pairs of bounding boxes
    // matchCounts[i][j] = number of matches where the first point is contained in prevFrame.boundingBoxes[i]
    // and the second point is contained in currFrame.boundingBoxes[j]
    std::vector<std::vector<int>> matchCounts;
    for (int i = 0; i < prevFrame.boundingBoxes.size(); ++i)
    {
        std::vector<int> row;
        for (int j = 0; j < currFrame.boundingBoxes.size(); ++j)
        {
            row.push_back(0);
        }
        matchCounts.push_back(row);
    }

    // Figure out which boxes contain the points from each match
    // Note, while we will eventually use the keypoint matches on the box, we're not going to use the keypoints
    // (the keypoint matches will continue to refer to the indices of points inside the entire frame, not inside the box)
    for (auto it = matches.begin(); it != matches.end(); ++it)
    {
        cv::KeyPoint prevPoint = prevFrame.keypoints[it->queryIdx];  // stackoverflow says that query is the previous and train is the current
        cv::KeyPoint currPoint = currFrame.keypoints[it->trainIdx];  // stackoverflow.com/questions/13318853

        // which boxes in the previous frame enclose prevPoint?
        std::vector<int> prevFrameEnclosingBoxes;
        for (int i = 0; i < prevFrame.boundingBoxes.size(); i++)
        {
            BoundingBox* prevBox = &(prevFrame.boundingBoxes[i]);
            if (prevBox->roi.contains(prevPoint.pt))
            {
                prevBox->keypoints.push_back(prevPoint);
                prevFrameEnclosingBoxes.push_back(i);
            }            
        }

        // which boxes in the current frame enclose currPoint?
        std::vector<int> currFrameEnclosingBoxes;
        for (int i = 0; i < currFrame.boundingBoxes.size(); i++)
        {
            BoundingBox* currBox = &(currFrame.boundingBoxes[i]);
            if (currBox->roi.contains(currPoint.pt))
            {
                currBox->keypoints.push_back(currPoint);
                currFrameEnclosingBoxes.push_back(i);
                // we will associate the match to only the current frame's box
                currBox->kptMatches.push_back(*it);
            }            
        }

        // increment the counts
        for (auto it1 = prevFrameEnclosingBoxes.begin(); it1 != prevFrameEnclosingBoxes.end(); ++it1)
        {
            for (auto it2 = currFrameEnclosingBoxes.begin(); it2 != currFrameEnclosingBoxes.end(); ++it2)
            {
                matchCounts[(*it1)][(*it2)]++;
            }
        }
    }

    // debug: print out the matrix of paired counts
    bool bVis = false;
    if (bVis)
    {
        printf("__|_");
        for (int j = 0; j < matchCounts[0].size(); ++j)
        {
            printf("%3d_", j);
        }
        printf("\n");
        for (int i = 0; i < matchCounts.size(); ++i)
        {
            printf("%d | ", i);
            for (int j = 0; j < matchCounts[i].size(); ++j)
            {
                printf("%3d ", matchCounts[i][j]);
            }
            printf("\n");
        }
    }

    // find the stable matches:
    // prev box A and curr box B are a stable match if most matches from A are to B, and most matches into B are from A
    // (note we may also need to add a threshold for "...and at least N points")
    std::vector<int> idxOfBestCurrentBox;
    std::vector<int> idxOfBestPrevBox;
    std::vector<int> maxMatchToCurrent;
    for (int i = 0; i < matchCounts.size(); ++i)
    {
        idxOfBestCurrentBox.push_back(-1);
        maxMatchToCurrent.push_back(0);
    }
    std::vector<int> maxMatchToPrev;
    for (int j = 0; j < matchCounts[0].size(); ++j)
    {
        idxOfBestPrevBox.push_back(-1);
        maxMatchToPrev.push_back(0);
    }
    for (int i = 0; i < matchCounts.size(); ++i)
    {
        for (int j = 0; j < matchCounts[i].size(); ++j)
        {
            if (maxMatchToCurrent[i] < matchCounts[i][j])
            {
                maxMatchToCurrent[i] = matchCounts[i][j];
                idxOfBestCurrentBox[i] = j;
            }
            if (maxMatchToPrev[j] < matchCounts[i][j])
            {
                maxMatchToPrev[j] = matchCounts[i][j];
                idxOfBestPrevBox[j] = i;
            }
        }
    }

    // finally, identify stable matches
    for (int i = 0; i < matchCounts.size(); ++i)
    {
        int j = idxOfBestCurrentBox[i];
        if (j > -1 && idxOfBestPrevBox[j] == i)
        {
            bbBestMatches.insert({i, j});
        }
    }

    if (bVis)
    {
        printf("Best matches for each previous box:\n");
        for (int i = 0; i < matchCounts.size(); ++i)
        {
            printf("    %d | %d with %d points\n", i, idxOfBestCurrentBox[i], maxMatchToCurrent[i]);
        }
        printf("Best matches for each current box:\n");
        for (int j = 0; j < matchCounts[0].size(); ++j)
        {
            printf("    %d | %d with %d points\n", j, idxOfBestPrevBox[j], maxMatchToPrev[j]);
        }
        printf("Stable matches:\n");
        for (auto it = bbBestMatches.begin(); it != bbBestMatches.end(); ++it)
        {
            printf("%d -> %d\n", it->first, it->second);
        }
    }
    
}
