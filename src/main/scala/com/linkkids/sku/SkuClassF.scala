package com.linkkids.sku

import com.linkkids.skuc.utils
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

object SkuClassF {

  case class wordFearture(label:Int,wordsFearture:String)
  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "E:\\soft\\hadoop-2.7.0\\hadoop-2.7.0")
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("skuclassficar")
      .getOrCreate()


    //设置 spark 的日志级别为 ERROR，过滤不必要的信息
    spark.sparkContext.setLogLevel("ERROR")
    val fileRDD = spark.sparkContext.textFile("D:\\logs\\labelbrand4.txt")

    val splitRDD = fileRDD.map(line => {
      try {
        //按照之前处理文本添加的奇异分隔符进行分割
        val splitwords = line.split("\t")
        //对文本中存在的一些无用符号作一些清除
        val filterWords = splitwords(0).trim
        //调用 IK 的分词方法对文本进行分词，返回一段空格符分割的字符串词组
        val ikWord = utils.ikAnalyzer(filterWords)
        wordFearture(splitwords(1).toInt, ikWord)
      } catch {
        //如果解析报错赋予空值
        case e: Exception => wordFearture(0, "")
      }
    })

    //将前一步中文本为空的内容给过滤掉，再转化为 DataFrame
    val wordset = spark.createDataFrame(splitRDD.filter(!_.wordsFearture.equals("")))
    //wordset.show()

    //运用 spark MLlib 的 Tokenizer 自带的按空格切割的方法转化上一步的 DataFrame 中的 wordsFearture
    val tokenizer = new Tokenizer().setInputCol("wordsFearture").setOutputCol("words")
    val wordsData = tokenizer.transform(wordset)

    //使用HashTF 方法计算词频，默认是2^18   大概25万左右
    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures")
    val featurizedData = hashingTF.transform(wordsData)

    //将上一步的 TF 计算 IDF
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    //rescaledData.show()

    //首先将文本分为两份，一份训练集用来训练模型，一份测试集用来验证模型的准确性
    val Array(trainingData, testData) = rescaledData.randomSplit(Array(0.7, 0.3), seed = 1234L)

    // 开始训练贝叶斯模型
    //贝叶斯默认的列名是  label   features
    val model = new NaiveBayes()
      .fit(trainingData)

    //使用训练好的模型预测测试集
    val predictions = model.transform(testData)

    predictions.show()
    predictions.createOrReplaceTempView("predictions")
    import spark.implicits._
    //将标签和预测结果不相等的导出探究原因
    val unResult = predictions.sparkSession.sql("select label,prediction,wordsFearture from predictions where label<>prediction")
    unResult.write.csv("D:\\logs\\unResult4")

    //
    //    //也可以使用 spark MLlib 自带的衡量指标的方法来计算正确率
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println(s"Test set accuracy = $accuracy")
  }


}
