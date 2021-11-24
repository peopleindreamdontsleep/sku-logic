package com.sparkml.sku

import com.sparkml.sku.SkuClassFPro.wordFearture
import com.sparkml.skuc.Utils
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.sql.SparkSession

/**
 * 加载训练好的模型进行预测
 * 使用pipline训练保存模型，加载的时候只需要输入RDD[wordFearture]即可
 * 如果正常训练，则需要将TF-IDF等流程全部走一遍
 */
object SkuLoadProModel {

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "E:\\soft\\hadoop-2.7.0\\hadoop-2.7.0")
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("SkuLoadModel")
      .getOrCreate()


    //设置 spark 的日志级别为 ERROR，过滤不必要的信息
    spark.sparkContext.setLogLevel("ERROR")

    //加载保存好的模型
    //pro用TrainValidationSplitModel，普通的用NaiveBayesModel
    val skuModel = TrainValidationSplitModel.load("/tmp/spark-bayes-allSku-model")


    val fileRDD = spark.sparkContext.textFile("D:\\logs\\nfpre.txt")

    val splitRDD = fileRDD.map(line => {
      try {
        //按照之前处理文本添加的奇异分隔符进行分割
        val splitwords = line.split("\t")
        //对文本中存在的一些无用符号作一些清除
        val filterWords = splitwords(0).trim
        //调用 IK 的分词方法对文本进行分词，返回一段空格符分割的字符串词组
        val ikWord = Utils.ikAnalyzer(filterWords)
        //预测模型，这里的label可以是商品的id，到时候可以将预测结果与商品行对应起来
        wordFearture(10, ikWord)
      } catch {
        //如果解析报错赋予空值
        case e: Exception => wordFearture(0, "")
      }
    })

    val wordset = spark.createDataFrame(splitRDD.filter(!_.wordsFearture.equals("")))

    val predictions=skuModel.transform(wordset)

    predictions.createOrReplaceTempView("predictions")
    import spark.implicits._
    //将标签和预测结果不相等的导出探究原因
    val unResult = predictions.sparkSession.sql("select label,prediction,wordsFearture from predictions where prediction=2")
    unResult.write.csv("D:\\logs\\null2Result")


  }

}
