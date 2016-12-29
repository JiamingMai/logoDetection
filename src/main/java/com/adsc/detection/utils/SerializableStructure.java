package com.adsc.detection.utils;

import java.io.*;

/**
 * This class provides kryo serialization for the JavaCV's Mat and Rect objects, so that Storm can wrap them in tuples.
 * Serializable.Mat - kryo serializable analog of opencv_core.Mat object.<p>
 * Serializable.Rect - kryo serializable analog of opencv_core.Rect object.<p>
 * Serializable.PatchIdentifier is also kryo serializable object,
 * which is used to identify each patch of the frame.<p>
 * <p>
 *
 * @author Nurlan Kanapin
 * @see SerializableMat
 * @see SerializableRect
 * @see SerializablePatchIdentifier
 */
public class SerializableStructure {

    /**
     * Kryo Serializable Mat class.
     * Essential fields are image data itself, rows and columns count and type of the data.
     */
    public static class SerializableMat implements java.io.Serializable {
        private byte[] data;
        private int rows, cols, type;

        public int getRows() {
            return rows;
        }

        public int getCols() {
            return cols;
        }

        public int getType() {
            return type;
        }

        public SerializableMat() {
        }

        /**
         * Creates new serializable Mat given its format and data.
         *
         * @param rows Number of rows in the Mat object
         * @param cols Number of columns in the Mat object
         * @param type OpenCV type of the data in the Mat object
         * @param data Byte data containing image.
         */
        public SerializableMat(int rows, int cols, int type, byte[] data) {
            this.rows = rows;
            this.cols = cols;
            this.type = type;
            this.data = data;
        }

        /**
         * Creates new serializable Mat given its format and data.
         *
         * @param input Byte data containing image.
         */
        public SerializableMat(byte[] input) {
            ByteArrayInputStream bis = new ByteArrayInputStream(input);
            ObjectInput in = null;
            try {
                in = new ObjectInputStream(bis);
                this.rows = in.readInt();
                this.cols = in.readInt();
                this.type = in.readInt();
                int size = in.readInt();
                this.data = new byte[size];
                int readed = 0;
                while (readed < size) {
                    readed += in.read(data, readed, size - readed);
                }
                System.out.println("in: " + this.rows + "-" + this.cols + "-" + this.type + "-" + input.length + "-" + size + "-" + readed);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        /**
         * Creates new serializable Mat from opencv_core.Mat
         *
         * @param mat The opencv_core.Mat
         */
        public SerializableMat(org.opencv.core.Mat mat) {
            if (!mat.isContinuous())
                mat = mat.clone();

            this.rows = mat.rows();
            this.cols = mat.cols();
            this.type = mat.type();
            int size = mat.rows() * mat.cols() * mat.channels();
            this.data = new byte[size];

            mat.get(0, 0, this.data);
//            ByteBuffer bb = mat.getByteBuffer();
//            bb.rewind();
//            this.data = new byte[size];
//            while (bb.hasRemaining())  // should happen only once
//                bb.get(this.data);
        }

        /**
         * @return Converts this Serializable Mat into JavaCV's Mat
         */
        public org.opencv.core.Mat toJavaCVMat() {
            org.opencv.core.Mat mat = new org.opencv.core.Mat(rows, cols, type);
            mat.put(0, 0, data);
            return mat;
        }


        public byte[] toByteArray() {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutput out = null;
            try {
                out = new ObjectOutputStream(bos);
                out.writeInt(this.rows);
                out.writeInt(this.cols);
                out.writeInt(this.type);
                out.writeInt(this.data.length);
                out.write(this.data);
                out.close();
                byte[] int_bytes = bos.toByteArray();
                bos.close();

                System.out.println("out: " + this.rows + "-" + this.cols + "-" + this.type + "-" + this.data.length + "-" + int_bytes.length);
                return int_bytes;
            } catch (IOException e) {
                e.printStackTrace();
            }
            return null;
        }

    }

    /**
     * Kryo Serializable Rect class.
     */
    public static class SerializableRect implements java.io.Serializable {
        /**
         * x, y, width, height - x and y coordinates of the left upper corner of the rectangle, its width and height
         */
        public int x, y, width, height;

        public SerializableRect() {
        }

        public SerializableRect(org.opencv.core.Rect rect) {
            x = rect.x;
            y = rect.y;
            width = rect.width;
            height = rect.height;
        }

        public SerializableRect(int x, int y, int width, int height) {
            this.x = x;
            this.y = y;
            this.height = height;
            this.width = width;
        }

        public org.opencv.core.Rect toJavaCVRect() {
            return new org.opencv.core.Rect(x, y, width, height);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            SerializableRect rect = (SerializableRect) o;

            if (height != rect.height) return false;
            if (width != rect.width) return false;
            if (x != rect.x) return false;
            if (y != rect.y) return false;

            return true;
        }

        @Override
        public int hashCode() {
            int result = x;
            result = 31 * result + y;
            result = 31 * result + width;
            result = 31 * result + height;
            return result;
        }
    }

    /**
     * This is a serializable class used for patch identification. Each patch needs to be distinguished form others.
     * Each patch is uniquely identified by the id of its frame and by the rectangle it corresponds to.
     */
    public static class SerializablePatchIdentifier implements java.io.Serializable {
        /**
         * Frame id of this patch
         */
        public int frameId;
        /**
         * Rectangle or Region of Interest of this patch.
         */
        public SerializableRect roi;

        public SerializablePatchIdentifier() {
        }

        /**
         * Creates PatchIdentifier with given frame id and rectangle.
         *
         * @param frameId
         * @param roi
         */
        public SerializablePatchIdentifier(int frameId, SerializableRect roi) {
            this.roi = roi;
            this.frameId = frameId;
        }

        /**
         * String representation of this patch identifier.
         *
         * @return the string in the format N%04d@%04d@%04d@%04d@%04d if roi is not null, and N%04d@null otherwise.
         */
        public String toString() {
            if (roi != null)
                return String.format("N%04d@%04d@%04d@%04d@%04d", frameId, roi.x, roi.y, roi.x + roi.width, roi.y + roi.height);
            return String.format("N%04d@null", frameId);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            SerializablePatchIdentifier that = (SerializablePatchIdentifier) o;

            if (frameId != that.frameId) return false;
            if (roi != null ? !roi.equals(that.roi) : that.roi != null) return false;

            return true;
        }

        @Override
        public int hashCode() {
            int result = frameId;
            result = 31 * result + (roi != null ? roi.hashCode() : 0);
            return result;
        }
    }

    public static class Point implements java.io.Serializable {
        double x;
        double y;

        public Point() {
        }

        public Point(Point p) {
            this.x = p.x;
            this.y = p.y;
        }

        public double x() {
            return this.x;
        }

        public double y() {
            return this.y;
        }

        public void x(float x) {
            this.x = x;
        }

        public void y(float y) {
            this.y = y;
        }
    }

    public static byte[] CvMat2ByteArray(org.opencv.core.Mat mat) {
        if (!mat.isContinuous()) {
            mat = mat.clone();
        }
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutput out = null;
        try {
            out = new ObjectOutputStream(bos);
            out.writeInt(mat.rows());
            out.writeInt(mat.cols());
            out.writeInt(mat.type());
            out.writeInt(mat.rows() * mat.cols() * mat.channels());

            byte[] data = new byte[mat.rows() * mat.cols() * mat.channels()];
            mat.get(0, 0, data);
            out.write(data);
            out.close();
            byte[] int_bytes = bos.toByteArray();
            bos.close();

            //System.out.println("out: " + this.rows + "-" + this.cols + "-" + this.type + "-" + this.data.length + "-" + int_bytes.length);
            return int_bytes;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static org.opencv.core.Mat ByteArray2CvMat(byte[] input){
        ByteArrayInputStream bis = new ByteArrayInputStream(input);
        ObjectInput in = null;
        try {
            in = new ObjectInputStream(bis);
            int rows = in.readInt();
            int cols = in.readInt();
            int type = in.readInt();
            int size = in.readInt();
            byte[] data = new byte[size];
            int readed = 0;
            while (readed < size) {
                readed += in.read(data, readed, size - readed);
            }
            org.opencv.core.Mat mat = new org.opencv.core.Mat(rows, cols, type);
            mat.put(0, 0, data);
            return mat;
//            System.out.println("in: " + this.rows + "-" + this.cols + "-" + this.type + "-" + input.length + "-" + size + "-" + readed);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

}