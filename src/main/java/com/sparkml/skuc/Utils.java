package com.sparkml.skuc;

import org.wltea.analyzer.core.IKSegmenter;
import org.wltea.analyzer.core.Lexeme;

import java.io.*;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;

public class Utils {

    public static String ikAnalyzer(String novelWord){

        if(novelWord == null || novelWord.trim().length() == 0){
            return "";
        }

        StringReader stringReader = new StringReader(novelWord);
        //使用 StringBuffer 而不是 String 来存储字符串
        StringBuffer stringBuffer = new StringBuffer();

        IKSegmenter ikSegmenter = new IKSegmenter(stringReader, true);
        Lexeme lex;
        try {
            while ((lex =ikSegmenter.next())!= null){
                //分词结果用空格符分隔开
                stringBuffer.append(lex.getLexemeText()+" ");
            }
        }catch (Exception e){
            e.printStackTrace();
        }
        return stringBuffer.toString();

    }

    public static void main(String[] args) {
        File file = new File("D:\\logs\\allbrand.csv");
        HashSet<String> brandSet = new HashSet<>();
        StringBuilder lineText=new StringBuilder();
        try {
            InputStreamReader read=new InputStreamReader(new FileInputStream(file));
            BufferedReader bufferedreader=new BufferedReader(read);
            File f = new File("D:\\logs\\allbrand.txt");
            OutputStream os = null;
            String contentLine;
            int i=0;
            while ((contentLine = bufferedreader.readLine())!=null){
                String[] brandArray ;
                if(contentLine.contains("、")){
                    brandArray = contentLine.split("、");
                }else{
                    brandArray = contentLine.split("/");
                }

                brandSet.addAll(Arrays.asList(brandArray));

            }

            for(String eachBrand:brandSet){
                lineText.append(eachBrand);
                lineText.append("\n");
                os = new FileOutputStream(f);
                os.write((lineText.toString()).getBytes());
                os.flush();
            }




        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
