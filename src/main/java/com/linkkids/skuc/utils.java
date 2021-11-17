package com.linkkids.skuc;

import org.wltea.analyzer.core.IKSegmenter;
import org.wltea.analyzer.core.Lexeme;

import java.io.StringReader;

public class utils {

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
}
