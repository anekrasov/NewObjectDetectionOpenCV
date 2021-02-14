import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

public class MainApp {
        public static void main(String[] args) {
            OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
//            String file = "C:\\Users\\mark1\\Documents\\opencv\\face-demographics-walking-and-pause.mp4";
            String file = "C:\\Users\\mark1\\Documents\\opencv\\yolo\\demo.mp4";
            VideoCapture capture = new VideoCapture();
            capture.set(CAP_PROP_FRAME_WIDTH, 1280);
            capture.set(CAP_PROP_FRAME_HEIGHT, 720);

            if (!capture.open(file)) {
                System.out.println("Can not open the cam !!!");
            }
            Mat colorimg = new Mat();
            CanvasFrame mainframe = new CanvasFrame("Car Detection", CanvasFrame.getDefaultGamma() / 2.2);
            mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
            mainframe.setCanvasSize(1280,720);
            mainframe.setLocationRelativeTo(null);
            mainframe.setVisible(true);

            while (true) {
                while (capture.read(colorimg) && mainframe.isVisible()) {
//                    DeepLearningFaceDetection.detectAndDraw(colorimg);
                    BasicExample.Yolo3(colorimg);
                    mainframe.showImage(converter.convert(colorimg));
//                    try {
//                        Thread.sleep(50);
//                    } catch (InterruptedException ex) {
//                        System.out.println(ex.getMessage());
//                    }
                }
            }
       }
}
