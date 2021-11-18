package com.sparkml.skuc;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;

/**
 * 给奶粉类目标文件打标
 * 具体标签见 getLabelMap 方法
 * 具体训练见scala工程
 */
public class DrinkPreProcess {

    public static void main(String[] args) {
        //给到商品库四级目录打标
        File file = new File("D:\\logs\\food.csv");
        //列出所有种类的文件夹
        StringBuffer lineText=new StringBuffer();
        HashSet<String> brandSet = new HashSet<String>();

        HashMap<String, String> labelMap = getLabelMap();


        try {
            InputStreamReader read=new InputStreamReader(new FileInputStream(file));
            BufferedReader bufferedreader=new BufferedReader(read);
            File f = new File("D:\\logs\\labelbrand4.txt");
            OutputStream os = null;
            String contentLine;
            int i=0;
            while ((contentLine = bufferedreader.readLine())!=null){
                String line = contentLine;
                String[] drinkArr = line.split("\t");
                if (drinkArr.length>7){
                    String title=drinkArr[1];
                    String brandName=drinkArr[2];
                    String cateName1=drinkArr[3];
                    String cateName2=drinkArr[4];
                    String cateName3=drinkArr[5];
                    String cateName4=drinkArr[6].toLowerCase(Locale.ROOT).trim();
                    String bDesc=drinkArr[7].toLowerCase(Locale.ROOT).replace("null","");
                    String totalDec =  title+bDesc;
                    String label="0";
                    if(labelMap.containsKey(cateName4)){
                        label=labelMap.get(cateName4);
                    }
                    if(!cateName3.equals("null")){
                        lineText.append(totalDec);
                        lineText.append("\t");
                        lineText.append(label);
                        lineText.append("\n");
                        os = new FileOutputStream(f);
                        os.write((lineText.toString()).getBytes());
                        os.flush();
                    }

                }else{
                    System.out.println(line);
                }

            }


        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 给到商品库二级目录打标，正确率较高
     */
    public  static void getCateNameTwo(){
        File file = new File("D:\\logs\\food.csv");
        //列出所有种类的文件夹
        StringBuffer lineText=new StringBuffer();
        HashSet<String> brandSet = new HashSet<String>();



        try {
            InputStreamReader read=new InputStreamReader(new FileInputStream(file));
            BufferedReader bufferedreader=new BufferedReader(read);
            File f = new File("D:\\logs\\labelbrand2.txt");
            OutputStream os = null;
            String contentLine;
            int i=0;
            while ((contentLine = bufferedreader.readLine())!=null){
                String line = contentLine;
                String[] drinkArr = line.split("\t");
                if (drinkArr.length>7){
                    String title=drinkArr[1];
                    String brandName=drinkArr[2];
                    String cateName1=drinkArr[3];
                    String cateName2=drinkArr[4];
                    String cateName3=drinkArr[5];
                    String cateName4=drinkArr[6];
                    String bDesc=drinkArr[7].toLowerCase(Locale.ROOT).replace("null","");
                    String totalDec =  title+bDesc;
                    String label="0";
                    if(cateName2.equals("儿童奶粉")){
                        label="1";
                    }else if(cateName2.equals("婴幼奶粉")){
                        label="2";
                    }else if(cateName2.equals("孕妇成人奶粉")){
                        label="3";
                    }
                    lineText.append(totalDec);
                    lineText.append("\t");
                    lineText.append(label);
                    lineText.append("\n");
                    os = new FileOutputStream(f);
                    os.write((lineText.toString()).getBytes());
                    os.flush();
                }else{
                    System.out.println(line);
                }

            }


        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static HashMap<String,String> getLabelMap(){
        HashMap<String, String> LabelMap =new HashMap<>();
        LabelMap.put("儿童牛奶粉","18");
        LabelMap.put("儿童羊奶粉","19");
        LabelMap.put("1段牛奶粉","1");
        LabelMap.put("2段牛奶粉","2");
        LabelMap.put("3段牛奶粉","3");
        LabelMap.put("4段牛奶粉","4");
        LabelMap.put("偏食厌食奶粉","5");
        LabelMap.put("早产儿奶粉","6");
        LabelMap.put("防腹泻奶粉","7");
        LabelMap.put("防过敏奶粉","8");
        LabelMap.put("1段羊奶粉","9");
        LabelMap.put("2段羊奶粉","10");
        LabelMap.put("3段羊奶粉","11");
        LabelMap.put("4段羊奶粉","12");
        LabelMap.put("孕妇牛奶粉","13");
        LabelMap.put("孕妇羊奶粉","14");
        LabelMap.put("成人牛奶粉","15");
        LabelMap.put("成人羊奶粉","16");
        LabelMap.put("骆驼奶粉","17");
        return LabelMap;
    }
}
