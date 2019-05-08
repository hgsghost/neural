import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class NeuralNet {
    double[] inputLayer;//输入层

    double[][] hiddenLayerWeight;//隐藏层权重
    double[] hiddenLayerB;//隐藏层偏置
    double[] hiddenLayerZ;//隐藏层z
    double[] hiddenLayerA;//隐藏层a
    double[] hiddenLayerErr;//隐藏层误差

    double[][] outputLayerWeight;//输出层权重
    double[] outputLayerB;//输出层偏置
    double[] outputLayerZ;//输出层z
    double[] outputLayerA;//输出层a
    double[] outputLayerExpect;//输出层期望
    double[] outputLayerErr;//输出误差



    double edu=3;//学习速率


    //初始化
    public NeuralNet(int inputCount, int hidderCount, int outputCount){

        inputLayer=new double[inputCount];
        hiddenLayerWeight=new double[hidderCount][inputCount];
        hiddenLayerB=new double[hidderCount];
        hiddenLayerZ=new double[hidderCount];
        hiddenLayerA=new double[hidderCount];
        hiddenLayerErr=new double[hidderCount];

        outputLayerWeight=new double[outputCount][hidderCount];
        outputLayerB=new double[outputCount];
        outputLayerZ=new double[outputCount];
        outputLayerA=new double[outputCount];
        outputLayerExpect=new double[outputCount];
        outputLayerErr=new double[outputCount];

        Random random = new java.util.Random();
        for(int i=0;i<hiddenLayerWeight.length;i++){
            for(int j=0;j<hiddenLayerWeight[i].length;j++){
                hiddenLayerWeight[i][j]=random.nextGaussian();
            }
        }
        for(int i=0;i<hiddenLayerB.length;i++){
            hiddenLayerB[i]=random.nextGaussian();
        }
        for(int i=0;i<outputLayerWeight.length;i++){
            for(int j=0;j<outputLayerWeight[i].length;j++){
                outputLayerWeight[i][j]=random.nextGaussian();
            }
        }
        for(int i=0;i<outputLayerB.length;i++){
            outputLayerB[i]=random.nextGaussian();
        }
    }
    //激活函数
    public static double sigmod(double z){
       return 1d /(1d+ Math.pow(Math.E,-z));
    }
    public  void forward(double[] inputLayer,double[] outputLayerExpect){
        this.inputLayer=inputLayer;
        this.outputLayerExpect=outputLayerExpect;
        for(int i=0; i<hiddenLayerZ.length;i++){
            hiddenLayerZ[i]=0.0;
            for(int j=0;j<inputLayer.length;j++){
                hiddenLayerZ[i]+=inputLayer[j]*hiddenLayerWeight[i][j];
            }
            hiddenLayerZ[i]=hiddenLayerZ[i]+hiddenLayerB[i];
            hiddenLayerA[i]=sigmod(hiddenLayerZ[i]);
        }
        for(int i=0; i<outputLayerZ.length;i++){
            outputLayerZ[i]=0.0;
            for(int j=0;j<hiddenLayerA.length;j++){
                outputLayerZ[i]+=hiddenLayerA[j]*outputLayerWeight[i][j];
            }
            outputLayerZ[i]=outputLayerZ[i]+outputLayerB[i];
            outputLayerA[i]=sigmod(outputLayerZ[i]);
            //outputLayerErr[i]=Math.pow(outputLayerExpect[i]-outputLayerA[i],2);
        }

    };
    public double sigmod_prime(double z){
        return  sigmod(z)*(1-sigmod(z));
    }

    public void backPropagation(){
        for(int i=0;i<outputLayerErr.length;i++){
            outputLayerErr[i]=(outputLayerA[i]-outputLayerExpect[i])*sigmod_prime(outputLayerZ[i]);
        }
        for(int i=0;i<hiddenLayerErr.length;i++){
            double sum=0;
            for(int j=0;j<outputLayerErr.length;j++){
                sum+=outputLayerErr[j]* outputLayerWeight[j][i];
            }
            hiddenLayerErr[i]=sum*sigmod_prime(hiddenLayerZ[i]);
        }

        for(int i=0;i<outputLayerWeight.length;i++){
            for(int j=0;j<outputLayerWeight[i].length;j++){
                outputLayerWeight[i][j]-=edu*outputLayerErr[i]*hiddenLayerA[j];
            }
        }
        for(int i=0;i<outputLayerB.length;i++){
            outputLayerB[i]-=edu*outputLayerErr[i];
        }
        for(int i=0;i<hiddenLayerWeight.length;i++){
            for(int j=0;j<hiddenLayerWeight[i].length;j++){
                hiddenLayerWeight[i][j]-=edu*hiddenLayerErr[i]*inputLayer[j];
            }
        }
        for(int i=0;i<hiddenLayerB.length;i++){
            hiddenLayerB[i]-=edu*hiddenLayerErr[i];
        }


    }

    public static void main(String[] args){

        NeuralNet neuralNet=new NeuralNet(2,10,1);

        for (int c = 0; c < 100000; c++) {
          int i = Math.random() > 0.5 ? 1 : 0;
          int j = Math.random() > 0.5 ? 1 : 0;
          double[] inputLayer = new double[] {i, j};
          double[] outputLayerExpect = new double[]{i==j?1:0};
          //outputLayerExpect[i*j+i+j]=1;
          neuralNet.forward(inputLayer, outputLayerExpect);
          neuralNet.backPropagation();
          System.out.println(c);
        }
        System.out.println("训练完成");
    {
      int i = 1;
      int j = 1;
      double[] inputLayer = new double[] {i, j};
        double[] outputLayerExpect = new double[1] ;

      neuralNet.forward(inputLayer, outputLayerExpect);

        System.out.println("输出为"+Arrays.toString(neuralNet.outputLayerA)+"/期望为"+(i==j));
        }
    {
      int i = 0;
      int j = 0;
      double[] inputLayer = new double[] {i, j};
        double[] outputLayerExpect = new double[1] ;
      neuralNet.forward(inputLayer, outputLayerExpect);
        System.out.println("输出为"+Arrays.toString(neuralNet.outputLayerA)+"/期望为"+(i==j));
        }
    {
      int i = 1;
      int j = 0;
      double[] inputLayer = new double[] {i, j};
        double[] outputLayerExpect = new double[1] ;
      neuralNet.forward(inputLayer, outputLayerExpect);
        System.out.println("输出为"+Arrays.toString(neuralNet.outputLayerA)+"/期望为"+(i==j));
        }
    {
      int i = 0;
      int j = 1;
      double[] inputLayer = new double[] {i, j};
        double[] outputLayerExpect = new double[1] ;
      neuralNet.forward(inputLayer, outputLayerExpect);
        System.out.println("输出为"+Arrays.toString(neuralNet.outputLayerA)+"/期望为"+(i==j));
        }
    }

}
