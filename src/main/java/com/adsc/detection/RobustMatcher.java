package com.adsc.detection;

import static com.adsc.detection.utils.SerializableStructure.*;

import com.adsc.detection.utils.Util;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.features2d.BFMatcher;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Intern04 on 8/8/2014.
 * Implementation of robust matcher described in OpenCV Cookbook ch. 9
 */
public class RobustMatcher {

    /** Parameters for detection {@link com.adsc.detection.Parameters}*/
    private Parameters params;

    /** Bruteforce matcher which matches two sets of keypoints */
    private BFMatcher matcher;

    /** Lists of matches from template to patch, from patch to template, and matches they have in common.
     Used for symmetry test (see OpenCV cookbook Ch. 9)*/
    private List<MatOfDMatch> matches12, matches21, matches;

    /** The rectangle corresponding to detected logo */
    private SerializableRect foundRect;

    private SerializableMat extractedTemplate;

    /**
     * Finds homography between two sets of keypoints using <a href="http://en.wikipedia.org/wiki/RANSAC">RANSAC</a>
     * algorithm. Does two iterations: first separates the outliers from inliers,
     * the second iteration performs the RANSAC inliers only to achieve better result.
     * @param logoKeyPoints the key points of the logo template
     * @param frameRegionKeyPoints key points of the frame patch
     * @return 3x3 transformation matrix
     */
    private Mat getHomography(MatOfKeyPoint logoKeyPoints, MatOfKeyPoint frameRegionKeyPoints) {

        // First iteration: find homography matrix and outliers
        int size = matches.size();
        Mat _src = new Mat(size, 2, CvType.CV_32FC1);
        Mat _dst = new Mat(size, 2, CvType.CV_32FC1);
        for (int i = 0 ; i < size ; i ++) {
            int queryIndex = (int) matches.get(i).toList().get(0).queryIdx;
            int trainIndex = (int) matches.get(i).toList().get(0).trainIdx;
            Point logoPoint = logoKeyPoints.toList().get(queryIndex).pt;
            Point frameRegionPoint = frameRegionKeyPoints.toList().get(trainIndex).pt;
            //logoKeyPoints.position(0);
            //frameRegionKeyPoints.position(0);
            _src.put(i, 0, logoPoint.x);
            _src.put(i, 1, logoPoint.y);
            _dst.put(i, 0, frameRegionPoint.x);
            _dst.put(i, 1, frameRegionPoint.y);
        }
        MatOfPoint2f src = new MatOfPoint2f(_src);
        MatOfPoint2f dst = new MatOfPoint2f(_dst);
        Mat mask = new Mat(src.rows(), 1, CvType.CV_32FC1, new Scalar(0));

        // Information about outliers will be stored in mask
        Mat h = Calib3d.findHomography(src, dst, Calib3d.RANSAC, params.getRansacParameters().getReprojectionThreshold(), mask, 2000, 0.995);


        // Second iteration: using only inliers
        Mat __src = new Mat(0, 2, CvType.CV_32FC1);
        Mat __dst = new Mat(0, 2, CvType.CV_32FC1);

        for (int i = 0 ; i < mask.rows() ; i ++) {
            if (mask.get(i, 0)[0] == 1.0) { // discard outliers
                __src.push_back(src.row(i));
                __dst.push_back(dst.row(i));
            }
        }

        src = new MatOfPoint2f(__src);
        dst = new MatOfPoint2f(__dst);
        mask = new Mat(__src.rows(), 1, CvType.CV_32FC1, new Scalar(0));

        // Find a homography matrix based on only inliers
        h = Calib3d.findHomography(src, dst, Calib3d.RANSAC,
                params.getRansacParameters().getReprojectionThreshold(), mask, 2000, 0.995);

        // Force release
        __src.release();
        __dst.release();
        mask.release();
        return h;
    }

    /**
     * Performs symmetry test as described in OpenCV cookbook pp 239-246, given two sets of matches:
     * from logo to image and from image to logo.
     * @param matches12
     * @param matches21
     * @return vector of refined matches
     */
    private List<MatOfDMatch> symmetryTest(
            List<MatOfDMatch> matches12,
            List<MatOfDMatch> matches21)
    {
        List<MatOfDMatch> matches = new ArrayList<MatOfDMatch>();
        // Checking every pair of matches and choosing symmetric ones.
        for (int i = 0 ; i < matches12.size(); i ++) {
            for (int j = 0 ; j < matches21.size(); j ++) {
                MatOfDMatch matOfDMatch12 = matches12.get(i);
                double[] tmpDMatch = matOfDMatch12.get(0, 0);
                DMatch dMatch12 = new DMatch();
                dMatch12.queryIdx = (int) tmpDMatch[0];
                dMatch12.trainIdx = (int) tmpDMatch[1];
                dMatch12.imgIdx = (int) tmpDMatch[2];
                dMatch12.distance = (int) tmpDMatch[3];
                ;
                MatOfDMatch matOfDMatch21 = matches21.get(j);
                tmpDMatch = matOfDMatch21.get(0, 0);
                DMatch dMatch21 = new DMatch();
                dMatch21.queryIdx = (int) tmpDMatch[0];
                dMatch21.trainIdx = (int) tmpDMatch[1];
                dMatch21.imgIdx = (int) tmpDMatch[2];
                dMatch21.distance = (int) tmpDMatch[3];
                if (dMatch12.queryIdx == dMatch21.trainIdx && dMatch12.trainIdx == dMatch21.queryIdx) {
                    matches.add(new MatOfDMatch(dMatch21));
                    break;
                }
            }
        }
        return matches;
    }

    /**
     * Perform
     *  <a href = "http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf#page=20">Lowe's Ratio test</a>
     */
    private List<MatOfDMatch> refineMatches(List<MatOfDMatch> oldMatches) {
        // Ratio of Distances
        double RoD = params.getMatchingParameters().getRatioOfDistances();
        List<MatOfDMatch> newMatches = new ArrayList<MatOfDMatch>();

        // Refine results 1: Accept only those matches, where best dist is < RoD of 2nd best match.
        double maxDist = 0.0, minDist = 1e100; // infinity

        for (int i = 0 ; i < oldMatches.size(); i ++) {
            if (oldMatches.get(i).toList().get(0).distance < RoD * oldMatches.get(i).toList().get(0).distance) {
                newMatches.add(oldMatches.get(i));
                double distance = oldMatches.get(i).toList().get(0).distance; // [3] for distance
                if ( distance < minDist )
                    minDist = distance;
                if ( distance > maxDist )
                    maxDist = distance;
            }
        }

        // Refine results 2: accept only those matches which distance is no more than 3x greater than best match
        List<MatOfDMatch> brandNewMatches = new ArrayList<MatOfDMatch>();
        for (int i = 0 ; i < newMatches.size(); i ++) {
            // TODO: Move this weights into params
            // Since minDist may be equal to 0.0, add some non-zero value
            if (newMatches.get(i).toList().get(0).distance <= 3 * minDist + maxDist / 12) {
                brandNewMatches.add(newMatches.get(i));
            }
        }
        return brandNewMatches;
    }

    /**
     * Creates instance of RobustMatcher using parameters.
     * @param params
     */
    public RobustMatcher(Parameters params) {
        matcher = new BFMatcher();
        this.params = params;
    }

    /**
     * Performs matching of two images.
     * @param logoMat - the image matrix of the logo template
     * @param logoDescriptors the descriptor matrix of the logo template
     * @param logoKeyPoints the key points of the logo template
     * @param frameRegionMat the image matrix of the tested frame patch
     * @param frameRegionDescriptors the descriptor matrix of the tested frame patch
     * @param frameRegionKeyPoints the keypoints of the tested frame patch
     * @param roi the rectanlge corresponding to the tested patch
     * @return true if the logo is present on the patch, and false otherwise
     */
    public boolean matchImages(Mat logoMat, Mat logoDescriptors, MatOfKeyPoint logoKeyPoints,
                               Mat frameRegionMat, Mat frameRegionDescriptors, MatOfKeyPoint frameRegionKeyPoints, Rect roi) {

        // Find matches from the logo to the patch and vice versa
        matches12 = new ArrayList<MatOfDMatch>();
        matches21 = new ArrayList<MatOfDMatch>();
        // For each match we need also second best match to perform Ratio Test
        matcher.knnMatch(logoDescriptors, frameRegionDescriptors, matches12, 2); // Find only two best matches.
        matcher.knnMatch(frameRegionDescriptors, logoDescriptors, matches21, 2); // Find only two best matches.

        //TODO
        // Performing ratio test
        matches12 = refineMatches(matches12);
        matches21 = refineMatches(matches21);

        // rolling back the vector
        //matches12.position(0);
        //matches21.position(0);

        // performing symmetry test
        matches = symmetryTest(matches12, matches21); // updates matches using information from matches12 and matches 21

        // Return false if too small number of matches defined in params
        int size = matches.size();
        if (size < params.getMatchingParameters().getMinimalNumberOfMatches()) {
            return false;
        }

        // Getting homography and checking that it's found
        Mat homography = getHomography(logoKeyPoints, frameRegionKeyPoints);
        if (homography == null || homography.empty()) {
            if (Debuger.logoDetectionDebugOutput)
                System.out.println("No homography found");
            return false;
        }

        // Choose rectangle where 90% (this number defined in params.getMatchingParameters().getBoxAccuracy())
        // of all matches lie in a logo and extract this rectangle as a new logo template
        ArrayList<Point> kp = new ArrayList<Point>();
        for (int i = 0 ; i < matches.size(); i ++) {
            Point location = logoKeyPoints.toList().get(matches.get(i).toList().get(0).queryIdx).pt;
            kp.add(location);
            //logoKeyPoints.position(0);
        }
        double[][] corners = new double[4][];
        // Find this desired rectanlge
        Rect best = Util.bestBoundingBoxFast(kp, params.getMatchingParameters().getBoxAccuracy());
        double xMin = best.x, yMin = best.y, xMax = best.x + best.width, yMax = best.y + best.height;


        // adjust borders of the extracted template since some of keypoints lie on the boundaries of the rectanlge
        // TODO: adjust this borders
        double dx = logoMat.cols() / 10.0, dy = logoMat.rows() / 10.0;
        xMin = Math.max(xMin - dx, 0.0);
        xMax = Math.min(xMax + dx, logoMat.cols() - 1);
        yMin = Math.max(yMin - dy, 0.0);
        yMax = Math.min(yMax + dy, logoMat.rows() - 1);
        corners[0] = new double[]{xMin, yMin};
        corners[1] = new double[]{xMax, yMin};
        corners[2] = new double[]{xMax, yMax};
        corners[3] = new double[]{xMin, yMax};

        // map this rectangle to the image to obtain its coordinates relative to the patch
        Mat objCornersMat = new Mat(4, 2, CvType.CV_32FC2);
        Mat sceneCornersMat = new Mat(4, 2, CvType.CV_32FC2);
        objCornersMat.put(0, 0, corners[0][0]);
        objCornersMat.put(0, 1, corners[0][1]);
        objCornersMat.put(1, 0, corners[1][0]);
        objCornersMat.put(1, 1, corners[1][1]);
        objCornersMat.put(2, 0, corners[2][0]);
        objCornersMat.put(2, 1, corners[2][1]);
        objCornersMat.put(3, 0, corners[3][0]);
        objCornersMat.put(3, 1, corners[3][1]);
        Core.perspectiveTransform(objCornersMat, sceneCornersMat, homography);
        double[][] scene_corners = new double[4][2];
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 2; j++){
                scene_corners[i][j] = sceneCornersMat.get(i, j)[0];
            }
        }
        homography.release();

        // Checking obtained corners for 'regularity'
        if (Util.checkQuadrilateral(scene_corners, roi)) {

            xMin = 1e100; xMax = 0.0; yMin = 1e100; yMax = 0.0;
            for (int i = 0 ; i < 4; i ++) {
                double x = scene_corners[i][0] , y = scene_corners[i][1] ;
                if (xMin > x) xMin = x;
                if (xMax < x) xMax = x;
                if (yMin > y) yMin = y;
                if (yMax < y) yMax = y;
                // making coordinates relative to the frame
                scene_corners[i][0] += roi.x;
                scene_corners[i][1] += roi.y;
            }
            int xmn = Math.max(0, (int)xMin), xmx = Math.min(frameRegionMat.cols(), (int) xMax);
            int ymn = Math.max(0, (int)yMin), ymx =  Math.min(frameRegionMat.rows(), (int)yMax);

            // Prepare this rectangle as a result of matching
            foundRect = new SerializableRect(xmn + roi.x, ymn + roi.y, xmx - xmn, ymx - ymn);

            // Prepare new logo template to push update to other bolts
            Rect newRoi = new Rect(xmn, ymn, xmx - xmn, ymx - ymn);
            Mat _new = new Mat(frameRegionMat, newRoi);
            extractedTemplate = new SerializableMat(_new);

            // Deallocating
            /*
            for (int i = 0 ; i < matches.size() ; i++)
                matches.get(i).get(0).deallocate();
            matches.deallocate();
            */
            return true;
        }
        /*
        for (int i = 0 ; i < matches.size() ; i++)
            matches.get(i).get(0).deallocate();
        matches.deallocate();
        */
        return false;
    }

    /**
     * Obtain the image matrix of the extracted template if the logo was found on the patch
     * @return the Serializable matrix
     */
    public SerializableMat getExtractedTemplate() {
        return extractedTemplate;
    }

    /**
     * Obtain the rectangle with coordinates of detected logo
     * @return Serializable rectangle containing box enclosing the detected logo
     */
    public SerializableRect getFoundRect() {
        return foundRect;
    }
}
