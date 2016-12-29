package com.adsc.detection;

import org.opencv.core.*;
import org.opencv.xfeatures2d.SIFT;

public class SIFTfeatures {

    MatOfKeyPoint keyPoints; //= new opencv_features2d.KeyPoint();
    Mat testDescriptors;// = new opencv_core.Mat();
    Mat rr;// = subFrame.clone();
    Rect roi;

    public SIFTfeatures(SIFT sift, Mat frame, Rect rec, boolean isWholeFrame){

        keyPoints = new MatOfKeyPoint();
        testDescriptors = new Mat();
        roi = new Rect();
        roi.x = rec.x;
        roi.y = rec.y;
        roi.width = rec.width;
        roi.height = rec.height;

        if (isWholeFrame) {
            Mat r = new Mat(frame, roi);
            rr = r.clone(); // make r continuous
            r.release();
        }else {
            rr = frame.clone();//subframe
        }
        sift.detectAndCompute(rr, new Mat(), keyPoints, testDescriptors);
    }

    public void release(){
        // Manually force JVM to release this.
        rr.release();
        //keyPoints..deallocate();
        testDescriptors.release();
    }
}