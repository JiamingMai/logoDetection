package com.adsc.detection;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import com.adsc.detection.utils.SerializableStructure.*;

/**
 *
 * <p>Descriptors and keypoints are precomputed before matching occurs, so that there is no need to extract features
 * and key points each time. Priority is used to compare the value of the logo templates. The higher the priority,
 * the earlier this logo is checked for presence on the patch. Also logo templates with higher priority have less
 * chance to be removed from the list of logo templates. Finally, each logo template has its own identifier, so
 * that we can distinguish between them.</p>
 * <p>How can we uniquely identify logo template? Since any logo template was
 * extracted from some frame's patch, each logo template is identified by patchIdentifier. patchIdentifier is the
 * identifier of the patch from which this template was extracted. This is only applied to dynamic list of templates
 * that is created during real-time detection. Those that were there from the beginning (original logo templates,
 * loaded during initialization) have not been extracted from the video, and hence their patch identifier have
 * negative frameId and null rectangle.</p>
 */
public class LogoTemplate implements Comparable<LogoTemplate> {
    Mat imageMat;
    Mat descriptor;
    MatOfKeyPoint keyPoints;

    SerializablePatchIdentifier identifier;
    public int priority;

    /* Creates template with given image, key points, descriptor, and identifier */
    public LogoTemplate(Mat mat, MatOfKeyPoint keyPoints, Mat descriptor, SerializablePatchIdentifier identifier)
    {
        this.imageMat = mat;
        this.descriptor = descriptor;
        this.keyPoints = keyPoints;
        this.identifier = identifier;
        priority = 0;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        LogoTemplate that = (LogoTemplate) o;

        if (!identifier.equals(that.identifier)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return identifier.hashCode();
    }

    public String toString() {
        return "" + priority;
    }

    @Override
    public int compareTo(LogoTemplate o) {
        if (this.priority > o.priority) return -1;
        return 1;
    }
}