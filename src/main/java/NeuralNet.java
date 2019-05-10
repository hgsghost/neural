import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class NeuralNet {
    private double[] inputLayer;//输入层

    private double[][] hiddenLayerWeight;//隐藏层权重
    private double[][][] hiddenLayerWeightErr;//隐藏层权重实际误差值
    private double[] hiddenLayerB;//隐藏层偏置
    private double[][] hiddenLayerBErr;//隐藏层偏置实际误差值
    private double[] hiddenLayerZ;//隐藏层z
    private double[] hiddenLayerA;//隐藏层a
    private double[] hiddenLayerErr;//隐藏层误差


    private double[][] outputLayerWeight;//输出层权重
    private double[][][] outputLayerWeightErr;//输出层权重实际误差
    private double[] outputLayerB;//输出层偏置
    private double[][] outputLayerBErr;//输出层偏置实际误差
    private double[] outputLayerZ;//输出层z
    private double[] outputLayerA;//输出层a
    private double[] outputLayerExpect;//输出层期望
    private double[] outputLayerErr;//输出误差



    private double edu=2;//学习速率
    private int batchCount=1000;//批次数量


    public double[] getInputLayer() {
        return inputLayer;
    }

    public void setInputLayer(double[] inputLayer) {
        this.inputLayer = inputLayer;
    }

    public double[][] getHiddenLayerWeight() {
        return hiddenLayerWeight;
    }

    public void setHiddenLayerWeight(double[][] hiddenLayerWeight) {
        this.hiddenLayerWeight = hiddenLayerWeight;
    }

    public double[][][] getHiddenLayerWeightErr() {
        return hiddenLayerWeightErr;
    }

    public void setHiddenLayerWeightErr(double[][][] hiddenLayerWeightErr) {
        this.hiddenLayerWeightErr = hiddenLayerWeightErr;
    }

    public double[] getHiddenLayerB() {
        return hiddenLayerB;
    }

    public void setHiddenLayerB(double[] hiddenLayerB) {
        this.hiddenLayerB = hiddenLayerB;
    }

    public double[][] getHiddenLayerBErr() {
        return hiddenLayerBErr;
    }

    public void setHiddenLayerBErr(double[][] hiddenLayerBErr) {
        this.hiddenLayerBErr = hiddenLayerBErr;
    }

    public double[] getHiddenLayerZ() {
        return hiddenLayerZ;
    }

    public void setHiddenLayerZ(double[] hiddenLayerZ) {
        this.hiddenLayerZ = hiddenLayerZ;
    }

    public double[] getHiddenLayerA() {
        return hiddenLayerA;
    }

    public void setHiddenLayerA(double[] hiddenLayerA) {
        this.hiddenLayerA = hiddenLayerA;
    }

    public double[] getHiddenLayerErr() {
        return hiddenLayerErr;
    }

    public void setHiddenLayerErr(double[] hiddenLayerErr) {
        this.hiddenLayerErr = hiddenLayerErr;
    }

    public double[][] getOutputLayerWeight() {
        return outputLayerWeight;
    }

    public void setOutputLayerWeight(double[][] outputLayerWeight) {
        this.outputLayerWeight = outputLayerWeight;
    }

    public double[][][] getOutputLayerWeightErr() {
        return outputLayerWeightErr;
    }

    public void setOutputLayerWeightErr(double[][][] outputLayerWeightErr) {
        this.outputLayerWeightErr = outputLayerWeightErr;
    }

    public double[] getOutputLayerB() {
        return outputLayerB;
    }

    public void setOutputLayerB(double[] outputLayerB) {
        this.outputLayerB = outputLayerB;
    }

    public double[][] getOutputLayerBErr() {
        return outputLayerBErr;
    }

    public void setOutputLayerBErr(double[][] outputLayerBErr) {
        this.outputLayerBErr = outputLayerBErr;
    }

    public double[] getOutputLayerZ() {
        return outputLayerZ;
    }

    public void setOutputLayerZ(double[] outputLayerZ) {
        this.outputLayerZ = outputLayerZ;
    }

    public double[] getOutputLayerA() {
        return outputLayerA;
    }

    public void setOutputLayerA(double[] outputLayerA) {
        this.outputLayerA = outputLayerA;
    }

    public double[] getOutputLayerExpect() {
        return outputLayerExpect;
    }

    public void setOutputLayerExpect(double[] outputLayerExpect) {
        this.outputLayerExpect = outputLayerExpect;
    }

    public double[] getOutputLayerErr() {
        return outputLayerErr;
    }

    public void setOutputLayerErr(double[] outputLayerErr) {
        this.outputLayerErr = outputLayerErr;
    }

    public double getEdu() {
        return edu;
    }

    public void setEdu(double edu) {
        this.edu = edu;
    }

    public int getBatchCount() {
        return batchCount;
    }

    public void setBatchCount(int batchCount) {
        this.batchCount = batchCount;
    }

    //初始化
    public NeuralNet(int inputCount, int hidderCount, int outputCount){

        inputLayer=new double[inputCount];
        hiddenLayerWeight=new double[hidderCount][inputCount];
        hiddenLayerWeightErr=new double[batchCount][hidderCount][inputCount];
        hiddenLayerB=new double[hidderCount];
        hiddenLayerBErr=new double[batchCount][hidderCount];
        hiddenLayerZ=new double[hidderCount];
        hiddenLayerA=new double[hidderCount];
        hiddenLayerErr=new double[hidderCount];

        outputLayerWeight=new double[outputCount][hidderCount];
        outputLayerWeightErr=new double[batchCount][outputCount][hidderCount];
        outputLayerB=new double[outputCount];
        outputLayerBErr=new double[batchCount][outputCount];
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
    public void train(double[][] inputLayer,double[][] outputLayerExpect){

        List<Integer> list=new ArrayList<>();
        for(int i=0;i<inputLayer.length;i++){
            list.add(i);
        }
        for(int i=0;i<inputLayer.length;i++){
            Random random=new Random();
            int index= random.nextInt(list.size());
            int count=list.remove(index);
            forward(inputLayer[count],outputLayerExpect[count],i%batchCount);
            if(i!=0&&i%batchCount==0){
                backPropagation();
            }
        }
    }

    public  void forward(double[] inputLayer,double[] outputLayerExpect,int batchCount){
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

            outputLayerErr[i]=Math.pow(outputLayerExpect[i]-outputLayerA[i],2);
            //outputLayerErr[i]=Math.pow(outputLayerExpect[i]-outputLayerA[i],2);
        }

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
                outputLayerWeightErr[batchCount][i][j]=outputLayerErr[i]*hiddenLayerA[j];
            }
        }
        for(int i=0;i<outputLayerB.length;i++){
            outputLayerBErr[batchCount][i]=outputLayerErr[i];
        }
        for(int i=0;i<hiddenLayerWeight.length;i++){
            for(int j=0;j<hiddenLayerWeight[i].length;j++){
                hiddenLayerWeightErr[batchCount][i][j]=hiddenLayerErr[i]*inputLayer[j];
            }
        }
        for(int i=0;i<hiddenLayerB.length;i++){
            hiddenLayerBErr[batchCount][i]=hiddenLayerErr[i];
        }


    };
    public double sigmod_prime(double z){
        return  sigmod(z)*(1-sigmod(z));
    }

    public void backPropagation(){

        /*for(int i=0;i<outputLayerErr.length;i++){
            outputLayerErr[i]=(outputLayerA[i]-outputLayerExpect[i])*sigmod_prime(outputLayerZ[i]);
        }*/
        /*for(int i=0;i<hiddenLayerErr.length;i++){
            double sum=0;
            for(int j=0;j<outputLayerErr.length;j++){
                sum+=outputLayerErr[j]* outputLayerWeight[j][i];
            }
            hiddenLayerErr[i]=sum*sigmod_prime(hiddenLayerZ[i]);
        }*/
        outputLayerWeightErr[0]=avg(outputLayerWeightErr);
        outputLayerBErr[0]=avg(outputLayerBErr);
        hiddenLayerWeightErr[0]=avg(hiddenLayerWeightErr);
        hiddenLayerBErr[0]=avg(hiddenLayerBErr);

        for(int i=0;i<outputLayerWeight.length;i++){
            for(int j=0;j<outputLayerWeight[i].length;j++){

                outputLayerWeight[i][j]-=edu*outputLayerWeightErr[0][i][j];
            }
        }
        for(int i=0;i<outputLayerB.length;i++){
            outputLayerB[i]-=edu*outputLayerBErr[0][i];
        }
        for(int i=0;i<hiddenLayerWeight.length;i++){
            for(int j=0;j<hiddenLayerWeight[i].length;j++){
                hiddenLayerWeight[i][j]-=edu*hiddenLayerWeightErr[0][i][j];
            }
        }
        for(int i=0;i<hiddenLayerB.length;i++){
            hiddenLayerB[i]-=edu*hiddenLayerBErr[0][i];
        }


    }
    //三维数组求平均坍缩成二维数据
    public static double[][] avg(double[][][] matrix){
        for(int x=0;x<matrix[0].length;x++){
            for(int y=0;y<matrix[0][0].length;y++){
                double sum=0;
                for(int z=0;z<matrix.length;z++){
                    sum+=matrix[z][x][y];
                }
                matrix[0][x][y]=sum/matrix.length;
            }
        }
        return matrix[0];
    }
    //二维数组求平均坍缩成一维数据
    public static double[] avg(double[][] matrix){
        for(int x=0;x<matrix[0].length;x++){
                double sum=0;
                for(int z=0;z<matrix.length;z++){
                    sum+=matrix[z][x];
                }
                matrix[0][x]=sum/matrix.length;
        }
        return matrix[0];
    }

  public static void main(String[] args) throws IOException {
      File directory = new File("");//设定为当前文件夹

      try {
          System.out.println(directory.getCanonicalPath());//获取标准的路径
          System.out.println(directory.getAbsolutePath());//获取绝对路径
      } catch (IOException e) {
          e.printStackTrace();
      }

      double[][] images = MnistRead.getImages(MnistRead.TRAIN_IMAGES_FILE);
      double[] labels = MnistRead.getLabels(MnistRead.TRAIN_LABELS_FILE);
      double[][] labelArr=new double[labels.length][10];
      //System.out.println(JSONObject.toJSONString(images));
      for(int i=0;i<labels.length;i++){
          labelArr[i]=new double[]{0,0,0,0,0,0,0,0,0,0};
          int index=(int)labels[i];
          labelArr[i][index]=1;
      }


      double[][] imagesTest = MnistRead.getImages(MnistRead.TEST_IMAGES_FILE);
      double[] labelsTest = MnistRead.getLabels(MnistRead.TEST_LABELS_FILE);

        NeuralNet neuralNet=new NeuralNet(784,50,10);


        while(true){

            neuralNet.train(images,labelArr);



            int count=0;
            for(int i=0;i<imagesTest.length;i++){
                double[] outputLayerExpect = new double[]{0,0,0,0,0,0,0,0,0,0};
                neuralNet.forward(imagesTest[i], outputLayerExpect,0);
                int a=(int)labelsTest[i];
                if(neuralNet.outputLayerA[a]>0.5){
                    count++;
                    //System.out.println(Arrays.toString(neuralNet.outputLayerA)+"/"+a);
                }
            }
            System.out.println("共"+imagesTest.length+"测试数据/成功"+count+"数据");
            if(count>8500){
                output(neuralNet,"1");
                break;
            }
        }

      while(true){

          neuralNet.train(images,labelArr);



          int count=0;
          for(int i=0;i<imagesTest.length;i++){
              double[] outputLayerExpect = new double[]{0,0,0,0,0,0,0,0,0,0};
              neuralNet.forward(imagesTest[i], outputLayerExpect,0);
              int a=(int)labelsTest[i];
              if(neuralNet.outputLayerA[a]>0.5){
                  count++;
                  //System.out.println(Arrays.toString(neuralNet.outputLayerA)+"/"+a);
              }
          }
          System.out.println("共"+imagesTest.length+"测试数据/成功"+count+"数据");
          if(count>9000){
              output(neuralNet,"1");
              break;
          }
      }
      while(true){

          neuralNet.train(images,labelArr);



          int count=0;
          for(int i=0;i<imagesTest.length;i++){
              double[] outputLayerExpect = new double[]{0,0,0,0,0,0,0,0,0,0};
              neuralNet.forward(imagesTest[i], outputLayerExpect,0);
              int a=(int)labelsTest[i];
              if(neuralNet.outputLayerA[a]>0.5){
                  count++;
                  //System.out.println(Arrays.toString(neuralNet.outputLayerA)+"/"+a);
              }
          }
          System.out.println("共"+imagesTest.length+"测试数据/成功"+count+"数据");
          if(count>9500){
              output(neuralNet,"2");
              break;
          }
      }

      while(true){

          neuralNet.train(images,labelArr);



          int count=0;
          for(int i=0;i<imagesTest.length;i++){
              double[] outputLayerExpect = new double[]{0,0,0,0,0,0,0,0,0,0};
              neuralNet.forward(imagesTest[i], outputLayerExpect,0);
              int a=(int)labelsTest[i];
              if(neuralNet.outputLayerA[a]>0.5){
                  count++;
                  //System.out.println(Arrays.toString(neuralNet.outputLayerA)+"/"+a);
              }
          }
          System.out.println("共"+imagesTest.length+"测试数据/成功"+count+"数据");
          if(count>9800){
              output(neuralNet,"3");
              break;
          }
      }





  }
  public static void writeFile(String name,String data) throws IOException {
      File hiddenwf = new File(name);
      if(!hiddenwf.exists()){
          hiddenwf.createNewFile();
      }
      FileWriter fileWritter = new FileWriter(hiddenwf.getName(),false);
      fileWritter.write(data);
      fileWritter.close();
  }
    public static void output(NeuralNet neuralNet,String name) throws IOException {
        List<List<Double>> hiddenwlist=new ArrayList<>();
        double[][] hiddenw=neuralNet.getHiddenLayerWeight();
        for(int i=0;i<hiddenw.length;i++){
            List<Double> row=new ArrayList<>();
            for(int j=0;j<hiddenw[i].length;j++){
                row.add(hiddenw[i][j]);
            }
            hiddenwlist.add(row);
        }
        writeFile("hiddenw"+name,JSONArray.toJSONString(hiddenwlist));

        List<Double> hiddenblist=new ArrayList<>();
        double[] hiddenb=neuralNet.getHiddenLayerB();
        for(int i=0;i<hiddenb.length;i++){
            hiddenblist.add(hiddenb[i]);
        }
        writeFile("hiddenb"+name,JSONArray.toJSONString(hiddenblist));


        List<List<Double>> outputwlist=new ArrayList<>();
        double[][] outputw=neuralNet.getOutputLayerWeight();
        for(int i=0;i<outputw.length;i++){
            List<Double> row=new ArrayList<>();
            for(int j=0;j<outputw[i].length;j++){
                row.add(outputw[i][j]);
            }
            outputwlist.add(row);
        }
        writeFile("ouputw"+name,JSONArray.toJSONString(outputwlist));

        List<Double> outputblist=new ArrayList<>();
        double[] outputb=neuralNet.getHiddenLayerB();
        for(int i=0;i<outputb.length;i++){
            outputblist.add(outputb[i]);
        }
        writeFile("outputb"+name,JSONArray.toJSONString(outputblist));
    }
}
