package com.adsc.detection;

import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import com.adsc.detection.utils.ImageViewer;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.xfeatures2d.SIFT;

import static com.adsc.detection.utils.SerializableStructure.*;

import javax.imageio.ImageIO;
import javax.swing.*;

public class logoDetectorRunner {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        String sourceFolder = "/home/jiamingmai/Pictures";
        String outputFolder = "/home/jiamingmai/Pictures/output";

        String srcVideoFile = "/home/jiamingmai/Videos/1.MOV";

        //int startFrame = 34500;
        //int endFrame = 37000;

        int startFrame = 0;
        int endFrame = 2500;

        int minNumberOfMatches = 4;
        int maxAdditionTemp = 4;
        int sampleRate = 4;
        List<String> templateFiles = new ArrayList<>();
        templateFiles.add("/home/jiamingmai/Pictures/mc.png");
        templateFiles.add("/home/jiamingmai/Pictures/adidas.jpg");

        long startTime = System.currentTimeMillis();
        System.out.println("Video logo detection - single machine - offline, start at: " + startTime);

        //LogoDetectionByInputImages(sourceFolder, outputFolder, startFrame, endFrame, maxAdditionTemp, minNumberOfMatches, templateFiles, sampleRate, false);
        //LogoDetectionByInputImagesGamma(sourceFolder, outputFolder, startFrame, endFrame, maxAdditionTemp, minNumberOfMatches, templateFiles, sampleRate, false, true);

        LogoDetectionByInputVideo(srcVideoFile, outputFolder, startFrame, endFrame, maxAdditionTemp, minNumberOfMatches, templateFiles, sampleRate, false);
        //LogoDetectionByInputVideoGamma(srcVideoFile, outputFolder, startFrame, endFrame, maxAdditionTemp, minNumberOfMatches, templateFiles, sampleRate, true);

        long endTime = System.currentTimeMillis();
        System.out.println("finished with duration: " + (endTime - startTime));
    }

    public static void LogoDetectionByInputVideo
            (String sourceFile, String outputFolder, int startFrame, int endFrame, int maxAdditionTemp,
             int minNumberOfMatches, List<String> templateFiles, int sampleRate, boolean toFile) {

        VideoCapture capture = new VideoCapture(sourceFile);

        Parameters parameters = new Parameters().withMatchingParameters(
                new Parameters.MatchingParameters().withMinimalNumberOfMatches(minNumberOfMatches));

        List<logoDetectorBeta> detectors = new ArrayList<>();
        for (int logoIndex = 0; logoIndex < templateFiles.size(); logoIndex++) {
            detectors.add(new logoDetectorBeta(parameters, templateFiles.get(logoIndex), logoIndex, maxAdditionTemp));
        }

        List<Scalar> colorList = new ArrayList<>();
        colorList.add(new Scalar(255, 0, 255)); // magenta
        colorList.add(new Scalar(0, 255, 255)); // yellow
        colorList.add(new Scalar(255, 255, 0)); // cyan
        colorList.add(new Scalar(0, 0, 255)); // blue
        colorList.add(new Scalar(0, 255, 0)); // green
        colorList.add(new Scalar(255, 0, 0)); // red
        colorList.add(new Scalar(0, 0, 0)); // black

        //StormVideoLogoDetector detector = new StormVideoLogoDetector(parameters, templateFiles);

        int frameId = 0;
        long totalFrameUsed = 0;
        while (++frameId < startFrame)
            capture.grab();
        int diff = endFrame - startFrame + 1;
        frameId = 0;
        endFrame = frameId + diff;

        List<SerializablePatchIdentifier> patchIdentifierList = new ArrayList<>();
        double fx = .25, fy = .25;
        double fsx = .5, fsy = .5;

        ///int W = mat.cols(), H = mat.rows();
        int W = 640, H = 480;
        int w = (int) (W * fx + .5), h = (int) (H * fy + .5);
        int dx = (int) (w * fsx + .5), dy = (int) (h * fsy + .5);

        for (int x = 0; x + w <= W; x += dx) {
            for (int y = 0; y + h <= H; y += dy) {
                patchIdentifierList.add(new SerializablePatchIdentifier(frameId, new SerializableRect(x, y, w, h)));
            }
        }
        System.out.println("W: " + W + ", H: " + H + ", total patch: " + patchIdentifierList.size());

        List<List<SerializableRect>> foundedRectList = new ArrayList<>();
        for (int logoIndex = 0; logoIndex < detectors.size(); logoIndex++) {
            foundedRectList.add(new ArrayList<>());
        }

        long start = System.currentTimeMillis();

        while (frameId < endFrame) {
            long frameStart = System.currentTimeMillis();

            Mat matOrg = new Mat();
            capture.read(matOrg);
            Mat mat = new Mat();
            Imgproc.resize(matOrg, mat, new Size(W, H));

            if (frameId % sampleRate == 0) {
                foundedRectList = LogoDetectionForOneFrame(frameId, mat, detectors, patchIdentifierList);
            }

            for (int logoIndex = 0; logoIndex < foundedRectList.size(); logoIndex++) {
                Scalar color = colorList.get(logoIndex % colorList.size());
                if (foundedRectList.get(logoIndex) != null) {
                    for (SerializableRect rect : foundedRectList.get(logoIndex)) {
                        Imgproc.rectangle(mat, new Point(rect.x, rect.y), new Point(rect.x + rect.width - 1, rect.y + rect.height - 1), color);
                    }
                }
            }

            long frameSpend = System.currentTimeMillis() - frameStart;
            if (toFile) {
                String outputFileName = outputFolder + System.getProperty("file.separator")
                        + String.format("frame%06d.jpg", (frameId + 1));
                Imgcodecs.imwrite(outputFileName, mat);
                //System.out.println("finish draw frameID: " + frameId);
            }

            frameId++;
            long nowTime = System.currentTimeMillis();
            totalFrameUsed += frameSpend;
            System.out.println("Sendout: " + nowTime + ", " + frameId + ", used: " + (nowTime - start) + ", frameUsed: " + frameSpend + ", totalFrameUsed: " + totalFrameUsed);
        }
    }


    public static void LogoDetectionByInputVideoGamma
            (String sourceFile, String outputFolder, int startFrame, int endFrame, int maxAdditionTemp,
             int minNumberOfMatches, List<String> templateFiles, int sampleRate, boolean toFile) {

        VideoCapture capture = new VideoCapture(sourceFile);

        Parameters parameters = new Parameters().withMatchingParameters(
                new Parameters.MatchingParameters().withMinimalNumberOfMatches(minNumberOfMatches));

        SIFT sift = SIFT.create(0, 3, parameters.getSiftParameters().getContrastThreshold(),
                parameters.getSiftParameters().getEdgeThreshold(), parameters.getSiftParameters().getSigma());

        List<logoDetectorGamma> detectors = new ArrayList<>();
        for (int logoIndex = 0; logoIndex < templateFiles.size(); logoIndex++) {
            detectors.add(new logoDetectorGamma(parameters, templateFiles.get(logoIndex), logoIndex, maxAdditionTemp));
        }

        List<Scalar> colorList = new ArrayList<>();
        colorList.add(new Scalar(255, 0, 255)); // magenta
        colorList.add(new Scalar(0, 255, 255)); // yellow
        colorList.add(new Scalar(255, 255, 0)); // cyan
        colorList.add(new Scalar(0, 0, 255)); // blue
        colorList.add(new Scalar(0, 255, 0)); // green
        colorList.add(new Scalar(255, 0, 0)); // red
        colorList.add(new Scalar(0, 0, 0)); // black

        //StormVideoLogoDetector detector = new StormVideoLogoDetector(parameters, templateFiles);

        int frameId = 0;
        long totalFrameUsed = 0;
        while (++frameId < startFrame)
            capture.grab();
        int diff = endFrame - startFrame + 1;
        frameId = 0;
        endFrame = frameId + diff;

        List<SerializablePatchIdentifier> patchIdentifierList = new ArrayList<>();
        double fx = .25, fy = .25;
        double fsx = .5, fsy = .5;

        ///int W = mat.cols(), H = mat.rows();
        int W = 640, H = 480;
        int w = (int) (W * fx + .5), h = (int) (H * fy + .5);
        int dx = (int) (w * fsx + .5), dy = (int) (h * fsy + .5);

        for (int x = 0; x + w <= W; x += dx) {
            for (int y = 0; y + h <= H; y += dy) {
                patchIdentifierList.add(new SerializablePatchIdentifier(frameId, new SerializableRect(x, y, w, h)));
            }
        }
        System.out.println("W: " + W + ", H: " + H + ", total patch: " + patchIdentifierList.size());

        List<List<SerializableRect>> foundedRectList = new ArrayList<>();
        for (int logoIndex = 0; logoIndex < detectors.size(); logoIndex++) {
            foundedRectList.add(new ArrayList<>());
        }

        long start = System.currentTimeMillis();

        while (frameId < endFrame) {
            long frameStart = System.currentTimeMillis();
            Mat matOrg = new Mat();
            capture.read(matOrg);
            Mat mat = new Mat();
            Imgproc.resize(matOrg, mat, new Size(W, H));

            if (frameId % sampleRate == 0) {
                foundedRectList = LogoDetectionForOneFrameGama(frameId, mat, detectors, patchIdentifierList, sift);
            }

            for (int logoIndex = 0; logoIndex < foundedRectList.size(); logoIndex++) {
                Scalar color = colorList.get(logoIndex % colorList.size());
                if (foundedRectList.get(logoIndex) != null) {
                    for (SerializableRect rect : foundedRectList.get(logoIndex)) {
                        Imgproc.rectangle(mat, new Point(rect.x, rect.y), new Point(rect.x + rect.width - 1, rect.y + rect.height - 1), color);
                    }
                }
            }

            long frameSpend = System.currentTimeMillis() - frameStart;
            if (toFile) {
                String outputFileName = outputFolder + System.getProperty("file.separator")
                        + String.format("frame%06d.jpg", (frameId + 1));
                Imgcodecs.imwrite(outputFileName, mat);
                //System.out.println("finish draw frameID: " + frameId);
            }

            frameId++;
            long nowTime = System.currentTimeMillis();
            totalFrameUsed += frameSpend;
            System.out.println("Sendout: " + nowTime + ", " + frameId + ", used: " + (nowTime - start) + ", frameUsed: " + frameSpend + ", totalFrameUsed: " + totalFrameUsed);
        }
    }


    public static void LogoDetectionByInputImages
            (String sourceFolder, String outputFolder, int startFrame, int endFrame, int maxAdditionTemp,
             int minNumberOfMatches, List<String> templateFiles, int sampleRate, boolean toFile) {

        Parameters parameters = new Parameters().withMatchingParameters(
                new Parameters.MatchingParameters().withMinimalNumberOfMatches(minNumberOfMatches));

        List<logoDetectorBeta> detectors = new ArrayList<>();
        for (int logoIndex = 0; logoIndex < templateFiles.size(); logoIndex++) {
            detectors.add(new logoDetectorBeta(parameters, templateFiles.get(logoIndex), logoIndex, maxAdditionTemp));
        }

        List<Scalar> colorList = new ArrayList<>();
        colorList.add(new Scalar(255, 0, 255)); // magenta
        colorList.add(new Scalar(0, 255, 255)); // yellow
        colorList.add(new Scalar(255, 255, 0)); // cyan
        colorList.add(new Scalar(0, 0, 255)); // blue
        colorList.add(new Scalar(0, 255, 0)); // green
        colorList.add(new Scalar(255, 0, 0)); // red
        colorList.add(new Scalar(0, 0, 0)); // black

        //StormVideoLogoDetector detector = new StormVideoLogoDetector(parameters, templateFiles);

        int generatedFrames = startFrame;
        int targetCount = endFrame - startFrame;

        int frameId = 0;
        long totalFrameUsed = 0;

        List<SerializablePatchIdentifier> patchIdentifierList = new ArrayList<>();
        double fx = .25, fy = .25;
        double fsx = .5, fsy = .5;

        ///int W = mat.cols(), H = mat.rows();
        int W = 640, H = 480;
        int w = (int) (W * fx + .5), h = (int) (H * fy + .5);
        int dx = (int) (w * fsx + .5), dy = (int) (h * fsy + .5);

        for (int x = 0; x + w <= W; x += dx) {
            for (int y = 0; y + h <= H; y += dy) {
                patchIdentifierList.add(new SerializablePatchIdentifier(frameId, new SerializableRect(x, y, w, h)));
            }
        }
        System.out.println("W: " + W + ", H: " + H + ", total patch: " + patchIdentifierList.size());

        List<List<SerializableRect>> foundedRectList = new ArrayList<>();
        for (int logoIndex = 0; logoIndex < detectors.size(); logoIndex++) {
            foundedRectList.add(new ArrayList<>());
        }

        long start = System.currentTimeMillis();
        while (frameId < targetCount) {
            String fileName = sourceFolder + System.getProperty("file.separator")
                    + String.format("frame%06d.jpg", (generatedFrames + 1));
            File f = new File(fileName);
            if (f.exists() == false) {
                System.out.println("File not exist: " + fileName);
                continue;
            }
            long frameStart = System.currentTimeMillis();
            Mat mat = Imgcodecs.imread(fileName);

            if (frameId % sampleRate == 0) {
                foundedRectList = LogoDetectionForOneFrame(frameId, mat, detectors, patchIdentifierList);
            }

            for (int logoIndex = 0; logoIndex < foundedRectList.size(); logoIndex++) {
                Scalar color = colorList.get(logoIndex % colorList.size());
                if (foundedRectList.get(logoIndex) != null) {
                    for (SerializableRect rect : foundedRectList.get(logoIndex)) {
                        Imgproc.rectangle(mat, new Point(rect.x, rect.y), new Point(rect.x + rect.width - 1, rect.y + rect.height - 1), color);
                    }
                }
            }

            long frameSpend = System.currentTimeMillis() - frameStart;
            if (toFile) {
                String outputFileName = outputFolder + System.getProperty("file.separator")
                        + String.format("frame%06d.jpg", (frameId + 1));
                Imgcodecs.imwrite(outputFileName, mat);
                //System.out.println("finish draw frameID: " + frameId);
            }

            frameId++;
            generatedFrames++;

            long nowTime = System.currentTimeMillis();
            totalFrameUsed += frameSpend;
            System.out.println("Sendout: " + nowTime + ", " + frameId + ", used: " + (nowTime - start) + ", frameUsed: " + frameSpend + ", totalFrameUsed: " + totalFrameUsed);

        }
    }

    public static void LogoDetectionByInputImagesGamma
            (String sourceFolder, String outputFolder, int startFrame, int endFrame, int maxAdditionTemp,
             int minNumberOfMatches, List<String> templateFiles, int sampleRate, boolean toFile, boolean display) {

        JFrame outputJFrame = new JFrame("Video logo detection - single machine - offline");
        outputJFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        outputJFrame.setSize(400, 400);
        outputJFrame.setLocation(520,30);
        JLabel outputLabel = new JLabel();
        outputJFrame.add(outputLabel);

        ImageViewer imageViewer = new ImageViewer();

        Parameters parameters = new Parameters().withMatchingParameters(
                new Parameters.MatchingParameters().withMinimalNumberOfMatches(minNumberOfMatches));

        SIFT sift = SIFT.create(0, 3, parameters.getSiftParameters().getContrastThreshold(),
                parameters.getSiftParameters().getEdgeThreshold(), parameters.getSiftParameters().getSigma());

        List<logoDetectorGamma> detectors = new ArrayList<>();
        for (int logoIndex = 0; logoIndex < templateFiles.size(); logoIndex++) {
            detectors.add(new logoDetectorGamma(parameters, templateFiles.get(logoIndex), logoIndex, maxAdditionTemp));
        }

        List<Scalar> colorList = new ArrayList<>();
        colorList.add(new Scalar(255, 0, 255)); // magenta
        colorList.add(new Scalar(0, 255, 255)); // yellow
        colorList.add(new Scalar(255, 255, 0)); // cyan
        colorList.add(new Scalar(0, 0, 255)); // blue
        colorList.add(new Scalar(0, 255, 0)); // green
        colorList.add(new Scalar(255, 0, 0)); // red
        colorList.add(new Scalar(0, 0, 0)); // black

        //StormVideoLogoDetector detector = new StormVideoLogoDetector(parameters, templateFiles);

        if (display) {
            outputJFrame.setVisible(true);
        }

        int generatedFrames = startFrame;
        int targetCount = endFrame - startFrame;

        int frameId = 0;
        long totalFrameUsed = 0;

        List<SerializablePatchIdentifier> patchIdentifierList = new ArrayList<>();
        double fx = .25, fy = .25;
        double fsx = .5, fsy = .5;

        ///int W = mat.cols(), H = mat.rows();
        int W = 640, H = 480;
        int w = (int) (W * fx + .5), h = (int) (H * fy + .5);
        int dx = (int) (w * fsx + .5), dy = (int) (h * fsy + .5);

        for (int x = 0; x + w <= W; x += dx) {
            for (int y = 0; y + h <= H; y += dy) {
                patchIdentifierList.add(new SerializablePatchIdentifier(frameId, new SerializableRect(x, y, w, h)));
            }
        }
        //System.out.println("W: " + W + ", H: " + H + ", total patch: " + patchIdentifierList.size());

        List<List<SerializableRect>> foundedRectList = new ArrayList<>();
        for (int logoIndex = 0; logoIndex < detectors.size(); logoIndex++) {
            foundedRectList.add(new ArrayList<>());
        }

        long start = System.currentTimeMillis();
        while (frameId < targetCount) {
            String fileName = sourceFolder + System.getProperty("file.separator")
                    + String.format("frame%06d.jpg", (generatedFrames + 1));
            File f = new File(fileName);
            if (f.exists() == false) {
                System.out.println("File not exist: " + fileName);
                continue;
            }
            long frameStart = System.currentTimeMillis();
            Mat mat = Imgcodecs.imread(fileName);

            if (frameId % sampleRate == 0) {
                foundedRectList = LogoDetectionForOneFrameGama(frameId, mat, detectors, patchIdentifierList, sift);
            }

            for (int logoIndex = 0; logoIndex < foundedRectList.size(); logoIndex++) {
                Scalar color = colorList.get(logoIndex % colorList.size());
                if (foundedRectList.get(logoIndex) != null) {
                    for (SerializableRect rect : foundedRectList.get(logoIndex)) {
                        Imgproc.rectangle(mat, new Point(rect.x, rect.y), new Point(rect.x + rect.width - 1, rect.y + rect.height - 1), color);
                    }
                }
            }

            long frameSpend = System.currentTimeMillis() - frameStart;
            if (toFile) {
                String outputFileName = outputFolder + System.getProperty("file.separator")
                        + String.format("frame%06d.jpg", (frameId + 1));
                Imgcodecs.imwrite(outputFileName, mat);
                //System.out.println("finish draw frameID: " + frameId);
            }

            if (display) {
                Image outputImg = imageViewer.toBufferedImage(mat);
                ImageIcon outputImageIcon = new ImageIcon(outputImg, "Video logo detection - single machine - offline");
                outputLabel.setIcon(outputImageIcon);
                outputJFrame.pack();
                try{
                    Thread.sleep(1);
                }catch (InterruptedException e){
                    e.printStackTrace();
                }
            }

            frameId++;
            generatedFrames++;

            long nowTime = System.currentTimeMillis();
            totalFrameUsed += frameSpend;
            //System.out.println("Sendout: " + nowTime + ", " + frameId + ", time elapse (ms): " + (nowTime - start) + ", frameUsed: " + frameSpend + ", totalFrameUsed: " + totalFrameUsed);
            System.out.println("Sendout: " + nowTime + ", " + frameId + ", time elapse (ms): " + (nowTime - start));

        }
    }

    public static List<List<SerializableRect>> LogoDetectionForOneFrame(
            int frameId, Mat mat, List<logoDetectorBeta> detectors,
            List<SerializablePatchIdentifier> patchIdentifierList) {

        List<List<SerializableRect>> foundedRectList = new ArrayList<>();

        for (int logoIndex = 0; logoIndex < detectors.size(); logoIndex++) {
            foundedRectList.add(new ArrayList<>());
        }

        for (int logoIndex = 0; logoIndex < detectors.size(); logoIndex++) {
            logoDetectorBeta detector = detectors.get(logoIndex);
            //int totalFoundCnt = 0;
            for (SerializablePatchIdentifier hostPatch : patchIdentifierList) {
                detector.detectLogosInRoi(mat, hostPatch.roi.toJavaCVRect());
                SerializableRect detectedLogo = detector.getFoundRect();
                SerializableMat extractedTemplate = detector.getExtractedTemplate();

                if (detectedLogo != null) {
                    detector.addTemplate(hostPatch, extractedTemplate);
                    detector.incrementPriority(detector.getParentIdentifier(), 1);

                    foundedRectList.get(logoIndex).add(detectedLogo);
                    //totalFoundCnt++;
                }
            }
            //System.out.println("FrameID: " + frameId + ", logoIndex: " + logoIndex + ", totalfoundRect: " + totalFoundCnt);
        }
        return foundedRectList;
    }

    public static List<List<SerializableRect>> LogoDetectionForOneFrameGama(
            int frameId, Mat mat, List<logoDetectorGamma> detectors,
            List<SerializablePatchIdentifier> patchIdentifierList, SIFT sift) {

        List<List<SerializableRect>> foundedRectList = new ArrayList<>();

        for (int logoIndex = 0; logoIndex < detectors.size(); logoIndex++) {
            foundedRectList.add(new ArrayList<>());
        }


        int totalFoundCnt = 0;
        for (SerializablePatchIdentifier hostPatch : patchIdentifierList) {
            SIFTfeatures sifTfeatures = new SIFTfeatures(sift, mat, hostPatch.roi.toJavaCVRect(), true);

            for (int logoIndex = 0; logoIndex < detectors.size(); logoIndex++) {
                logoDetectorGamma detector = detectors.get(logoIndex);

                //detector.detectLogosInRoi(mat, hostPatch.roi.toJavaCVRect());
                detector.detectLogosByFeatures(sifTfeatures);

                SerializableRect detectedLogo = detector.getFoundRect();
                SerializableMat extractedTemplate = detector.getExtractedTemplate();

                if (detectedLogo != null) {
                    detector.addTemplate(hostPatch, extractedTemplate);
                    detector.incrementPriority(detector.getParentIdentifier(), 1);

                    foundedRectList.get(logoIndex).add(detectedLogo);
                    totalFoundCnt++;
                }
            }
            // System.out.println("FrameID: " + frameId + ", totalfoundRect: " + totalFoundCnt);
        }
        return foundedRectList;
    }
}