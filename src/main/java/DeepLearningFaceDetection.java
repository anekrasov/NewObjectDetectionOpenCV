import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromCaffe;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class DeepLearningFaceDetection {

    private static final String PROTO_FILE = "C:\\Users\\mark1\\Documents\\opencv\\caffe\\deploy.prototxt";
//    private static final String CAFFE_MODEL_FILE = "C:\\Users\\mark1\\Documents\\opencv\\caffe\\res10_300x300_ssd_iter_140000.caffemodel";
//    private static final String CAFFE_MODEL_FILE = "C:\\Users\\mark1\\Documents\\opencv\\caffe\\googlenet_finetune_web_car_iter_10000.caffemodel";
    private static final String CAFFE_MODEL_FILE = "C:\\Users\\mark1\\Documents\\opencv\\caffe\\SSD_MobileNet.caffemodel";
    private static final Net net;

    static {
        net = readNetFromCaffe(PROTO_FILE, CAFFE_MODEL_FILE);
    }

    public static void detectAndDraw(Mat image) {//detect faces and draw a blue rectangle arroung each face
        resize(image, image, new Size(300, 300));//resize the image to match the input size of the model
        Mat blob = blobFromImage(image, 1.0, new Size(300, 300), new Scalar(104.0, 177.0, 123.0, 0), false, false, CV_32F);

        net.setInput(blob);//set the input to network model
        Mat output = net.forward();//feed forward the input to the netwrok to get the output matrix

        Mat ne = new Mat(new Size(output.size(3), output.size(2)), CV_32F, output.ptr(0, 0));//extract a 2d matrix for 4d output matrix with form of (number of detections x 7)

        FloatIndexer srcIndexer = ne.createIndexer(); // create indexer to access elements of the matric

        for (int i = 0; i < output.size(3); i++) {//iterate to extract elements
            float confidence = srcIndexer.get(i, 2);
            float f1 = srcIndexer.get(i, 3);
            float f2 = srcIndexer.get(i, 4);
            float f3 = srcIndexer.get(i, 5);
            float f4 = srcIndexer.get(i, 6);
            if (confidence > .6) {
                float tx = f1 * 300;//top left point's x
                float ty = f2 * 300;//top left point's y
                float bx = f3 * 300;//bottom right point's x
                float by = f4 * 300;//bottom right point's y
                rectangle(image, new Rect(new Point((int) tx, (int) ty), new Point((int) bx, (int) by)), new Scalar(255, 0, 0, 0));//print blue rectangle
            }
        }
    }
}