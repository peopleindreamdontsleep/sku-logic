package com.sparkml.sku

import com.sparkml.skuc.Utils
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

/**
 * 使用pipline来进行调参训练，寻找最优模型
 */
object SkuClassFPro {

  case class wordFearture(label:Int,wordsFearture:String)

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "E:\\soft\\hadoop-2.7.0\\hadoop-2.7.0")
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("skuclassficarpro")
      .getOrCreate()


    //设置 spark 的日志级别为 ERROR，过滤不必要的信息
    spark.sparkContext.setLogLevel("ERROR")
    val brandRDD = spark.sparkContext.textFile("D:\\logs\\labelbrand4.txt")

    val otherRDD = spark.sparkContext.textFile("D:\\logs\\other.txt")

    val fileRDD= brandRDD++otherRDD


    val splitRDD = fileRDD.map(line => {
      try {
        //按照之前处理文本添加的奇异分隔符进行分割
        val splitWords = line.split("\t")
        //对文本中存在的一些无用符号作一些清除
        val filterWords = splitWords(0).trim
        //调用 IK 的分词方法对文本进行分词，返回一段空格符分割的字符串词组
        val ikWord = Utils.ikAnalyzer(filterWords)
        wordFearture(splitWords(1).toInt, ikWord)
      } catch {
        //如果解析报错赋予空值
        case e: Exception => wordFearture(0, "")
      }
    })


    //将前一步中文本为空的内容给过滤掉，再转化为 DataFrame
    val wordset = spark.createDataFrame(splitRDD.filter(!_.wordsFearture.equals("")))
    val tokenizer = new Tokenizer().setInputCol("wordsFearture").setOutputCol("words")

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures")

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    val nb = new NaiveBayes()

    //这里和之前不同的是所有的步骤都在 pipline 中进行转化，它的 stages 使得每一步都很清晰，处理步骤一目了然
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF,idf,nb))

    //网格参数使得超参数调优更加的方便，只需要在网格中加入可能的参数，最终按照你添加的参数训练出一个最优的模型
    //经过几轮验证，发现参数为1.0的时候最优，所以将参数直接固定，后面每一轮跑的时候弄能快点
    val paramGrid = new ParamGridBuilder()
      .addGrid(nb.smoothing, Array(1.0))
      .build()

    //将所有的步骤加入到 TrainValidationSplit 中，包括 训练器、评估方法、模型的网格参数、并行度等
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
      .setParallelism(4)
    val Array(training, test) = wordset.randomSplit(Array(0.7, 0.3), seed = 12345)
    // 运行上面的TrainValidationSplit，最终会选择一个最优的模型进行输出，大大方便了调优的部分
    val model = trainValidationSplit.fit(training)

    //保存模型到本地
    //注意，保存的时候是TrainValidationSplitModel，所以在load的时候要用同意的model去load
    model.write.overwrite().save("/tmp/spark-bayes-sku-model")


    val predictions = model.transform(test)

    predictions.createOrReplaceTempView("predictions")
    import spark.implicits._
    //将标签和预测结果不相等的导出探究原因
    val unResult = predictions.sparkSession.sql("select label,prediction,wordsFearture from predictions where label<>prediction")
    unResult.write.csv("D:\\logs\\onebest")

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")
  }

}
