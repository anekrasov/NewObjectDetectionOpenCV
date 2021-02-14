import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromDarknet;

public class BasicExample {

    public static void Yolo3(Mat image){
//        Mat img = imread("E:\\CS_OpenCvSharpYolo3\\5troadnd.jpg");

        //setting blob, size can be:320/416/608
        //opencv blob setting can check here https://github.com/opencv/opencv/tree/master/samples/dnn#object-detection
        Mat blob = blobFromImage(image, 1.0 / 255, new Size(416, 416), new Scalar(), true, false,CV_32F);
        //Mat blob = opencv_dnn.blobFromImage(img, 1.0, new Size(608, 608), new Scalar(), true, false,CV_8U);

        //load model and config, if you got error: "separator_index < line.size()", check your cfg file, must be something wrong.
        String cfg = "C:\\Users\\mark1\\Documents\\opencv\\yolo\\config.cfg";
        String model = "C:\\Users\\mark1\\Documents\\opencv\\yolo\\yolov3.weights";
        Net net = readNetFromDarknet(cfg, model);
        //set preferable
        net.setPreferableBackend(3);
            /*
            0:DNN_BACKEND_DEFAULT
            1:DNN_BACKEND_HALIDE
            2:DNN_BACKEND_INFERENCE_ENGINE
            3:DNN_BACKEND_OPENCV
             */
        net.setPreferableTarget(0);
            /*
            0:DNN_TARGET_CPU
            1:DNN_TARGET_OPENCL
            2:DNN_TARGET_OPENCL_FP16
            3:DNN_TARGET_MYRIAD
            4:DNN_TARGET_FPGA
             */

        //input data
        net.setInput(blob);

        //get output layer name
        StringVector outNames = net.getUnconnectedOutLayersNames();
        //create mats for output layer
        //MatVector outs = outNames.Select(_ => new Mat()).ToArray();

        MatVector outs = new MatVector();
        for(int i=0;i<outNames.size();i++){
            outs.put(new Mat());
        }

        //forward model
//        StopWatch  sw = StopWatch.createStarted();
        net.forward(outs, outNames);
//        sw.stop();

        //get result from all output
        float threshold = 0.5f;       //for confidence
        float nmsThreshold = 0.3f;    //threshold for nms
        GetResult(outs, image, threshold, nmsThreshold);
    }

    private static void GetResult(MatVector output, Mat image, float threshold, float nmsThreshold)
    {
        boolean nms = true;
        //for nms
        ArrayList<Integer> classIds = new ArrayList<>();
        ArrayList<Float> confidences = new ArrayList<>();
        ArrayList<Float> probabilities = new ArrayList<>();
        ArrayList<Rect2d> rect2ds = new ArrayList<>();
        //Rect2dVector boxes = new Rect2dVector();
        try{
            int w = image.cols();
            int h = image.rows();
            /*
             YOLO3 COCO trainval output
             0 1 : center                    2 3 : w/h
             4 : confidence                  5 ~ 84 : class probability
            */
            int prefix = 5;   //skip 0~4
            /**/
            int indiceNum = 0;
            long boxNum = 0;
            for(int k=0;k<output.size();k++)
            {
                Mat prob = output.get(k);
                final FloatRawIndexer probIdx = prob.createIndexer();
                for (int i = 0; i < probIdx.rows(); i++)
                {
                    float confidence = probIdx.get(i, 4);
                    if (confidence > threshold)
                    {
                        //get classes probability
                        //Cv2.MinMaxLoc(prob.Row[i].ColRange(prefix, prob.Cols), out _, out Point max);
                        //Point min;
                        //Point max;
                        //minMaxLoc(prob.rows(i).colRange(prefix, prob.cols()), null,null, min, max,null);
                        indiceNum++;
                        DoublePointer minVal= new DoublePointer();
                        DoublePointer maxVal= new DoublePointer();
                        Point min = new Point();
                        Point max = new Point();
//                        minMaxLoc(prob.rows(i).colRange(prefix, prob.cols()), minVal, maxVal, min, max, null);
                        int classes = max.x();
                        float probability = probIdx.get(i, classes + prefix);

                        if (probability > threshold) //more accuracy, you can cancel it
                        {
                            //get center and width/height
                            float centerX = probIdx.get(i, 0) * w;
                            float centerY = probIdx.get(i, 1) * h;
                            float width = probIdx.get(i, 2) * w;
                            float height = probIdx.get(i, 3) * h;

                            if (!nms)
                            {
                                // draw result (if don't use NMSBoxes)
                                //Draw(image, classes, confidence, probability, centerX, centerY, width, height);
                                continue;
                            }

                            //put data to list for NMSBoxes
                            boxNum++;
                            classIds.add(classes);
                            confidences.add(confidence);
                            probabilities.add(probability);
                            rect2ds.add(new Rect2d(centerX, centerY, width, height));
                            //boxes.put(new Rect2d(centerX, centerY, width, height));
                            //boxes.resize(boxNum);
                        }
                    }
                }
            }

            if (!nms) return;

            //using non-maximum suppression to reduce overlapping low confidence box
            //CvDnn.NMSBoxes(boxes, confidences, threshold, nmsThreshold, out int[] indices);
            //int[] indices = new int[]{8,8,8,8,8,8,8,8};
            IntPointer indices = new IntPointer(confidences.size());

            Rect2dVector boxes = new Rect2dVector();
            for (Rect2d rect2d : rect2ds) {
                boxes.push_back(rect2d);
            }

            FloatPointer con = new FloatPointer(confidences.size());
            for (Float confidence : confidences) {
                con.put(confidence);
            }
            opencv_dnn.NMSBoxes(boxes, con, threshold, nmsThreshold, indices);

            List<String> list = new ArrayList<String>();
            FileInputStream fis = new FileInputStream("C:\\Users\\mark1\\Documents\\opencv\\yolo\\coco.names");

            InputStreamReader isr = new InputStreamReader(fis, StandardCharsets.UTF_8);
            BufferedReader br = new BufferedReader(isr);
            String line;
            while ((line = br.readLine()) != null) {
                list.add(line);
            }
            String[] Labels = list.toArray(new String[0]);
            br.close();
            isr.close();
            fis.close();
            //Console.WriteLine($"NMSBoxes drop {confidences.Count - indices.Length} overlapping result.");

            for (int m=0;m<indices.sizeof();m++)
            {
                int i = indices.get(m);
                System.out.println(i);
                Rect2d box = boxes.get(i);
                //]]Draw(image, classIds[i], confidences[i], probabilities[i], box.x(), box.y(), box.width(), box.height());
                String res = "name="+Labels[classIds.get(i)]+" classIds="+classIds.get(i)+" confidences="+confidences.get(i)+" probabilities="+probabilities.get(i);
                res += " box.x="+box.x() + " box.y="+box.y() + " box.width="+box.width() + " box.height="+box.height();
                System.out.println(res);
            }
        }catch(Exception e){
            System.out.println("GetResult error:" + e.getMessage());
        }
    }
}