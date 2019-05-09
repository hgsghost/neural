import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class MnistRead {

  public static final String TRAIN_IMAGES_FILE =
      "src/main/resources/data/mnist/train-images.idx3-ubyte";
    public static final String TRAIN_LABELS_FILE = "src/main/resources/data/mnist/train-labels.idx1-ubyte";
    public static final String TEST_IMAGES_FILE = "src/main/resources/data/mnist/t10k-images.idx3-ubyte";
    public static final String TEST_LABELS_FILE = "src/main/resources/data/mnist/t10k-labels.idx1-ubyte";

    /**
     * change bytes into a hex string.
     *
     * @param bytes bytes
     * @return the returned hex string
     */
    public static String bytesToHex(byte[] bytes) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < bytes.length; i++) {
            String hex = Integer.toHexString(bytes[i] & 0xFF);
            if (hex.length() < 2) {
                sb.append(0);
            }
            sb.append(hex);
        }
        return sb.toString();
    }

    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static double[][] getImages(String fileName) {
        double[][] x = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000803".equals(bytesToHex(bytes))) {                        // 读取魔数
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);           // 读取样本总数
                bin.read(bytes, 0, 4);
                int xPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每行所含像素点数
                bin.read(bytes, 0, 4);
                int yPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每列所含像素点数
                x = new double[number][xPixel * yPixel];
                for (int i = 0; i < number; i++) {
                    double[] element = new double[xPixel * yPixel];
                    for (int j = 0; j < xPixel * yPixel; j++) {
                        element[j] = bin.read();                                // 逐一读取像素值
                        // normalization
//                        element[j] = bin.read() / 255.0;
                    }
                    x[i] = element;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return x;
    }

    /**
     * get labels of `train` or `test`
     *
     * @param fileName the file of 'train' or 'test' about label
     * @return
     */
    public static double[] getLabels(String fileName) {
        double[] y = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000801".equals(bytesToHex(bytes))) {
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);
                y = new double[number];
                for (int i = 0; i < number; i++) {
                    y[i] = bin.read();
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return y;
    }

    /*public static void main(String[] args) {

        File directory = new File("");//设定为当前文件夹

        try {
            System.out.println(directory.getCanonicalPath());//获取标准的路径
            System.out.println(directory.getAbsolutePath());//获取绝对路径
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[][] images = getImages(TRAIN_IMAGES_FILE);
        double[] labels = getLabels(TRAIN_LABELS_FILE);

        double[][] imagesTest = getImages(TEST_IMAGES_FILE);
        double[] labelsTest = getLabels(TEST_LABELS_FILE);

//        System.out.println(images.length);
//        System.out.println(images[0].length);
//        System.out.println(labels.length);
//        System.out.println(Arrays.toString(images));
        NeuralNet neuralNet=new NeuralNet(784,30,10);
        for(int k=0;k<100;k++){

            for(int i=0;i<images.length;i++){
                Random ran=new Random();
                int j=ran.nextInt(images.length);
                int a=(int)labels[j];
                double[] outputLayerExpect = new double[]{0,0,0,0,0,0,0,0,0,0};
                outputLayerExpect[a]=1;
                neuralNet.forward(images[j], outputLayerExpect);
                neuralNet.backPropagation();
            }

            int count=0;
            for(int i=0;i<imagesTest.length;i++){
                int a=(int)labelsTest[i];
                double[] outputLayerExpect = new double[]{0,0,0,0,0,0,0,0,0,0};
                //outputLayerExpect[a]=1;
                neuralNet.forward(imagesTest[i], outputLayerExpect);
               if(neuralNet.outputLayerA[a]>0.5){
                   count++;
                   System.out.println(Arrays.toString(neuralNet.outputLayerA)+"/"+a);
               }
            }
            System.out.println("共"+imagesTest.length+"测试数据/成功"+count+"数据");
        }
    }*/
}
