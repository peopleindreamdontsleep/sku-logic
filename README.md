### 一、商品类目分类

#### 0. 背景

自有平台的商品会有品牌和四级类目，例如：

   ```
商品名为：蓝河绵羊奶粉800g
它所属的品牌是blue river/蓝河，
一级品类是食品
二级品类是儿童奶粉
三级品类是儿童羊奶粉
四级品类儿童羊奶粉
   ```

   由于是人工维护的，再加上商品数多大585万，就存在一定的问题，许多的类目、品牌都没有分类或者分错了，目的就是想找到哪些分错了，以及将未分类的分好，说白了，文本分类

#### 1. 基本概念

文本分类是 NLP 的一种，目的是为了给文本分类，即把文本归类到某一个类别。按照我们的日常经验，一个文本讲述伊拉克战争等内容，我们很自然就把它归类到军事，文本讲述一个青年拾金不昧，将捡到的钱包送到警察叔叔手中，那么自然就是社会类的文章，至于我们的大脑为什么能这么快地分出文章的类别？大概很多人都没有细细的思考过。

如果非要从理论上来解释的话，这里提供一种“自以为是”的解释：经过几十年的生活经验的累积，我们自然而然的形成了自己对于外界事物的概念，从文本方面来看，一篇文章里面包含“坦克”、“飞机”等字样，按照经验，肯定和军事离不开关系，反面来看，一篇正常的经济、政治类文章也不大可能出现这些字眼，从正面和反面论证结果来看，文章出现“坦克”、“飞机”等字眼大概率就是军事类文章，这大概就是文本分类基本的介绍和解释。

#### 2. 技术介绍

互联网时代到来，数据以指数级增长，自媒体的兴起，让文本的增长更是突飞猛进，文档作为一种非结构化的数据（MySQL 中存放的是结构化数据），对于它的分析本来就存在一定的难度，再加上数据量的猛增，让原本 Python 的单机机器学习也压力倍增，显得力不从心。

很多的机器学习算法在数据量少时可能效果还不错，但在大数据背景下，单机的训练慢影响到部署、超参数调优、数据结构调优等等方面，这时 Spark“全家桶”中的 MLlib 提供了一种很好的解决方案——分布式机器学习。

Spark 在离线、实时等方面性能明显，DataFrame、DataSet 让代码写起来更加方便、性能快捷，让我们专注于业务层面，而不必死磕性能，老是想着怎么优化。Spark MLlib 不仅结合了它的代码、性能方面的优点，更加丰富的外接数据源可以让机器学习和大数据更加紧密的结合起来。

**MLlib**

在我们的机器学习任务中，传入的参数大多是向量即都是数字，而中文文本必定都是汉字，如何把汉字转化为机器学习训练所需要的向量格式就是特征提取所要解决的问题。

下面列举一些中文特征提取的方法和局限：

![文本特征提取方法的优和劣](https://images.gitbook.cn/ded8b740-c28e-11e9-9538-55a0b92bba2a)

（图片来源于 [Text Classification Algorithms: A Survey](https://arxiv.org/abs/1904.08067)）

翻译如下：

![在这里插入图片描述](https://images.gitbook.cn/ed58e670-caf7-11e9-8b48-2f3e78c99db0)

从实现的复杂性和解释性等方面考虑，我们选择 TF-IDF 作为特征提取的方法。

TF-IDF 从字面意思来看分为 TF 和 IDF，TF 的意思是 Term Frequency，也就是词在文章中出现的频率，可以简单的认为是：一个词在文章中出现的频率越高，代表这个词越重要。比如：“坦克”这个词在军事类文章中出现了很多次，那么这个词对这类文章就会很重要，可能经济类的文章也会偶尔出现“坦克”，但肯定不会出现很多，那么这个词对经济类文章相对而言就不是那么重要。

IDF 的意思是 Inverse Document Frequency，也就是逆文本频率，可以认为是：一些词在一类文章中出现很多，如“坦克”，但在其他经济、政治类文章中很少出现，那么这个词就具有很好的分类能力，但相反，一些词在很多文章中都出现，如“有的”、“很多”等，它们虽然在很多文章中都出现了，但并没有很好的分类的能力，这个时候逆词频就发挥作用了，你出现的越多，你的比重反而下降了。

TF-IDF 的基本思想是：

> 一个词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

举个例子，“坦克”在 10000 篇文章共 100000 个词中总共出现了 100 次，那么“坦克”的 TF 就是 100/100000 = 0.001，“坦克”在 10000 篇文章中只在 10 篇里面出现过，那么“坦克”的 IDF 就是lg(10000/10) = 3，那么“坦克”的 TF-IDF 值就是 0.001*3 = 0.003。

同样的，假设“有的”这个词在 10000 篇中出现了 50000 词，那么它的 TF 就是 0.5，这个词因为非常常见，所以在 9900 篇文章中都出现了，那么它的 IDF 就是 lg(10000/9900) = 0.0044，“有的”这个词的 TF-IDF 值就是 0.0044*0.5 = 0.0022。

从上面的两个例子可以看出，虽然“有的”这个词的词频远高于“坦克”，但是它的逆词频大大的拉低了它的分值，这个就是 TF-IDF 对于一个词权重的计算，即把汉字能转化为向量。

#### 4. TF-IDF Spark 源码分析

TF-IDF 的理论并不难理解，实现起来也并不困难，无非就是通过计算词频和在多少个文档中出现来进行实现。

TF 的实现就是计算词频，一般的实现方法是，把所有的文档所有词遍历一遍，形成一个 Distinct List 去重词汇列表，然后再次遍历，这次遍历就是进行计数了，这种普通的实现方法有一个缺点，就是需要遍历两次才行，接下来我们看看 Spark 的实现。

```scala
  /**
   * 输入的文档计算词频
   * Transforms the input document into a sparse term frequency vector.
   */
  @Since("1.1.0")
  def transform(document: Iterable[_]): Vector = {
   //生成一个 HashMap 用来存储词汇和词汇的频率
    val termFrequencies = mutable.HashMap.empty[Int, Double]
    //如果生成的 Hash 值在 HashMap 中已经存在，则自增 1
    val setTF = if (binary) (i: Int) => 1.0 else (i: Int) => termFrequencies.getOrElse(i, 0.0) + 1.0
    // Hash 值的生成算法，有 murmur3Hash 和 nativeHash 两种实现方式，2.0之后默认是 murmur3Hash算法
    val hashFunc: Any => Int = getHashFunction
    document.foreach { term =>
      //为每一个词汇生成 Hash
      val i = Utils.nonNegativeMod(hashFunc(term), numFeatures)
      termFrequencies.put(i, setTF(i))
    }
    Vectors.sparse(numFeatures, termFrequencies.toSeq)
  }

   /* 
    * 根据生成的 Hash 值和给定的 HashMap 大小生成对应的值  
	* Calculates 'x' modulo 'mod', takes to consideration sign of x,
  	* i.e. if 'x' is negative, than 'x' % 'mod' is negative too
  	* so function return (x % mod) + mod in that case.
  */
  def nonNegativeMod(x: Int, mod: Int): Int = {
    val rawMod = x % mod
    rawMod + (if (rawMod < 0) mod else 0)
  }
```

代码使用 Scala 编写，看起来非常的简洁，实现思路就是通过 Hash，定义一定长度的 Hash，再为每一个词生成一个 Hash 值，存进 HashMap 中，相同的词用同一个算法生成的 Hash值肯定相同，如果 HashMap 存在，value 自增 1，不存在则设置为 1，整个过程只需要遍历一次词汇列表，比之前的方案更优。

```scala
class HashingTF(val numFeatures: Int) extends Serializable {
  //默认 Hash 算法
  private var hashAlgorithm = HashingTF.Murmur3

  /**RDD Based TF 算法特征默认值
   */
  @Since("1.1.0")
  def this() = this(1 << 20)
  
 /**DataFrame Based TF 算法特征默认值
   * Number of features. Should be greater than 0.
   * (default = 2^18^)
   * @group param
   */
  @Since("1.2.0")
  val numFeatures = new IntParam(this, "numFeatures", "number of features (> 0)",
    ParamValidators.gt(0))
```

HashMap 大小的设定也不是随机选取的，需要做一定的评估，Spark RDD 实现的 TF 默认大小是 1 << 20，大约是 100 万左右，而 DataFrame 的默认大小是 1 << 18，大约是 25 万左右，根据中文常用词汇的数量评估，HashMap 的大小取在 25 万比较合理，它的大小尽量能包含所有的词汇，如果过大，则词汇量小于 HashMap 大小，造成许多未知为空值，如果过小，则词汇量大于 HashMap 大小，造成很多词汇的 Hash 值计算重叠，使得效果打折扣，所以事先评估也是重要的一环。

```scala
 /**计算 IDF 即逆文档词频
   * Computes the inverse document frequency.
   * @param dataset an RDD of term frequency vectors
   */
  @Since("1.1.0")
  def fit(dataset: RDD[Vector]): IDFModel = {
    val idf = dataset.treeAggregate(new IDF.DocumentFrequencyAggregator(
          minDocFreq = minDocFreq))(
      seqOp = (df, v) => df.add(v),
      combOp = (df1, df2) => df1.merge(df2)
    ).idf()
    new IDFModel(idf)
  }

//实际计算词汇在多少个文档中出现
class DocumentFrequencyAggregator(val minDocFreq: Int) extends Serializable {

    /** number of documents */
    //总共的文档数
    private var m = 0L
    /** document frequency vector */
    //词汇在多少个文档中出现，全部包含在一个稠密向量里
    private var df: BDV[Long] = _
    /** Adds a new document. */
    def add(doc: Vector): this.type = {
      if (isEmpty) {
        df = BDV.zeros(doc.size)
      }
      doc match {
      	//TF 计算完成，传给 IDF 的值，传入的如果是稠密向量
        case SparseVector(size, indices, values) =>
          val nnz = indices.length
          var k = 0
          while (k < nnz) {
          	//如果有值，计数 +1，代表在多少个文档中出现
            if (values(k) > 0) {
              df(indices(k)) += 1L
            }
            k += 1
          }
         ...
      }
      //文档的总数 +1
      m += 1L
      this
    }

  //计算 IDF 的值
  def idf(): Vector = {
      ...
      while (j < n) {
        /*
         * minDocFreq，定义的最小在文档中出现的频次
         * log((m+1)/df(j)+1)中的m就是总的文档数，df(j)是在多少个文档中出现了，+1的目的是为了防止出现 0 的情况
         */
        if (df(j) >= minDocFreq) {
          inv(j) = math.log((m + 1.0) / (df(j) + 1.0))
        }
        j += 1
      }
      Vectors.dense(inv)
    }
  /**计算 TF * IDF 
   * Transforms a term frequency (TF) vector to a TF-IDF vector with a IDF vector
   *
   * @param idf an IDF vector
   * @param v a term frequency vector
   * @return a TF-IDF vector
   */
  def transform(idf: Vector, v: Vector): Vector = {
    val n = v.size
    v match {
      case SparseVector(size, indices, values) =>
        val nnz = indices.length
        val newValues = new Array[Double](nnz)
        var k = 0
        while (k < nnz) {
          //遍历整个 TF 向量，如果存在值则对应的 TF 和 IDF 相乘
          newValues(k) = values(k) * idf(indices(k))
          k += 1
        }
        Vectors.sparse(n, indices, newValues)
        ...
  }
```

#### 5. 中文分词

分词就是将连续的字序列按照一定的规范重新组合成词序列的过程。相对中文每个词之间以空格符分割而言，中文一段话语句之间没有明显的分隔符来进行分割，因为没有表面意义上的分隔符，所以中文的分词相对英文而言要难的多。

大部分分类效果好、分类结果优秀的中文分词都会在一定基础上进行收费，而免费开源的则存在词库不足的一些缺点，所以在使用开源工具是要注意词库的丰富。

本次选用的是开源中文分词库  IK Analyzer，它是基于 Java 语言开发的轻量级的中文分词工具包，在 2006 年 12 月推出 1.0 版。最初，它是以开源项目 Luence 为应用主体的，结合词典分词和文法分析算法的中文分词组件。从 3.0 版本开始，IK 发展为面向 Java 的公用分词组件，独立于 Lucene 项目，同时提供了对 Lucene 的默认优化实现。这里选取的是经典版本 IK Analyzer 2012。

从网上下载 IK Analyzer 2012，将它打包进本地的 Maven 仓库。

```shell
将第三方jar打包进本次 maven 仓库
mvn install:install-file -Dfile= D:\Tools\IKAnalyzer_2012 -DgroupId=com.ikanalyzer -DartifactId=analyzer -Dversion=1.0 -Dpackaging=jar

打进仓库之后使用下面依赖就可以使用IK分词了
<dependency>
   <groupId>com.ikanalyzer</groupId>
   <artifactId>analyzer</artifactId>
   <version>1.0</version>
</dependency>
```

可以在 Maven 仓库 .m2 目录下看见 jar 包：

![ik 在本次 maven 仓库](https://images.gitbook.cn/f848dbf0-c618-11e9-b5ad-0533828aa6f0)

在 Java 中写一个测试类：

```java
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
     System.out.println(ikAnalyzer("今天不是一个好天气，因为下了暴雨"));
}
分词结果：今天 不是 一个 好天气 因为 下了 暴雨 
```

分词结果会自动过滤一些停用词，如标点符号之类。

IK 分词的原理主要还是根据它自带的词库中的大约 27 万的词汇来进行匹配。

![ik 源码中的词库](https://images.gitbook.cn/2e519a60-c61a-11e9-8c18-3b7665bbaeb4)

如果在一些专业领域使用 IK 分词效果不好的话就要自己添加一些字典来进行补充。

本次由于是基于平台的奶粉类商品进行分词的，所以，必要的分词是必须的，首先将所有已有完整类目的品牌名字全部拉取下来，作为品牌词库，再加上一些奶粉类的专属名词作为扩展词库，扩展词库如下：

```tex
羊奶粉
牛奶粉
儿童
奶粉
牦牛奶
富硒
中老年
1段
2段
3段
4段
金装
一段
二段
高钙
高锌
三段
小听
四段
婴幼儿
婴儿
幼儿
孕妇牛奶粉
0-6
6-12
12-36
1-3
1-3岁
```

接着在项目的resources目录下新建一个IKAnalyzer.cfg.xml文件，作为IKAnalyzer的扩展词库配置，配置如下：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE properties SYSTEM "http://java.sun.com/dtd/properties.dtd">  
<properties>  
	<comment>IK Analyzer 扩展配置</comment>
	<!--用户可以在这里配置自己的扩展字典 ext.dic是扩展词库，brand.dic是品牌词库，food.dic是自己添加的其他词库-->
	<entry key="ext_dict">ext.dic;brand.dic;food.dic;</entry>

	<!--用户可以在这里配置自己的扩展停止词字典-->
	<entry key="ext_stopwords">stopword.dic;</entry>
	
</properties>
```

这样，在分词的时候，就会将自己新增的词库加载进去

#### 6. 分类算法选择

前面讲解的是数据处理，这里是最重要的环节，分类算法的选择，下面列举了机器学习任务以及算法选择的示意图。

![机器学习算法选择](https://images.gitbook.cn/a9bf47e0-c61c-11e9-8c18-3b7665bbaeb4)

（图片来源于 [scikit learn 官网](https://scikit-learn.org/stable/index.html#)）

从 start 开始，数据量 >50 所以走到 category 路口，新闻分类数据的打好标签的即每一个新闻属于哪个类别是分好了的，所以选择 labeled data 上面的部分，数据量小于 10 万并且是文本分类，所以走到左边分类模块的下方，最终到达的终点是贝叶斯分类模型，根据这张图，我们很方便的就找到了所需要的算法。

强行汉化之后的图片是：

![在这里插入图片描述](https://images.gitbook.cn/9beb0a10-caee-11e9-b421-bb73d778c438)

（图片来源于[博客](https://blog.csdn.net/a790209714/article/details/52708464)）

#### 7. 数据介绍

数据集采用的是自有品牌的数据，本次首先匹配的是奶粉类的商品，所以奶粉类的分类类目是作为主要标签，其他不属于奶粉的归为其他，

由于目前的四级品类都是唯一的，所以这里一步到位，直接选取最末端的品类进行匹配

```
"儿童牛奶粉","18"
"儿童羊奶粉","19"
"1段牛奶粉","1"
"2段牛奶粉","2"
"3段牛奶粉","3"
"4段牛奶粉","4"
"偏食厌食奶粉","5"
"早产儿奶粉","6"
"防腹泻奶粉","7"
"防过敏奶粉","8"
"1段羊奶粉","9"
"2段羊奶粉","10"
"3段羊奶粉","11"
"4段羊奶粉","12"
"孕妇牛奶粉","13"
"孕妇羊奶粉","14"
"成人牛奶粉","15"
"成人羊奶粉","16"
"骆驼奶粉","17"
"其他","0"
```

#### 

第一步就是对文本数据做一些处理，将商品的title和小b商户对这个商品的描述一起作为特征，自己定义的品类作为标签，为了完成有监督学习贝叶斯分类任务，需要把类别信息添加进去，目前只选取的食品类大概5万条商品数据，其中目标分类的奶粉类的大概有1万条

文本处理代码

```java
//给到商品库四级目录打标
File file = new File("D:\\logs\\food.csv");
//列出所有种类的文件夹
StringBuffer lineText=new StringBuffer();
HashSet<String> brandSet = new HashSet<String>();

HashMap<String, String> labelMap = getLabelMap();


try {
    InputStreamReader read=new InputStreamReader(new FileInputStream(file));
    BufferedReader bufferedreader=new BufferedReader(read);
    //生成的结果打标文件
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
```

上面是对原始的新闻文本作了一些处理，将一条新闻变成一条数据，并为它加上一个标签。

接来下就是开始分析、向量转化、贝叶斯分类的实践。

```scala
//每一条数据可分为 label 和 wordsFearture 两个属性
case class wordFearture(label:Int,wordsFearture:String)
def main(args: Array[String]) {
 val spark = SparkSession
   .builder()
   .appName("NewSclassficar")
   .getOrCreate()

//设置 spark 的日志级别为 ERROR，过滤不必要的信息
 spark.sparkContext.setLogLevel("ERROR")
 val fileRDD = spark.sparkContext.textFile("D:\\logs\\labelbrand4.txt")

 val splitRDD = fileRDD.map(line=>{
   try {
     //按照之前处理文本添加的奇异分隔符进行分割
     val splitwords = line.split("AL:")
     //对文本中存在的一些无用符号作一些清除
     val filterWords = splitwords(1).trim
     //调用 IK 的分词方法对文本进行分词，返回一段空格符分割的字符串词组
     val ikWord = JavaUtil.getSplitWords(filterWords)
     wordFearture(splitwords(0).toInt,ikWord)
   }catch {
   	 //如果解析报错赋予空值
     case e:Exception=>wordFearture(0,"")
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
 //predictions.show()

//    predictions.createOrReplaceTempView("predictions")
//    //将标签和预测结果不相等的导出探究原因
//    val unResult = predictions.sparkSession.sql("select label,prediction,wordsFearture from predictions where label<>prediction")
//    unResult.write.csv("D:\\logs\\unResult4")
 
 //也可以使用 spark MLlib 自带的衡量指标的方法来计算正确率
 val evaluator = new MulticlassClassificationEvaluator()
   .setLabelCol("label")
   .setPredictionCol("prediction")
   .setMetricName("accuracy")
 val accuracy = evaluator.evaluate(predictions)
 println(s"Test set accuracy = $accuracy")
```

accuracy = 0.8818012504988693

正确率挺高的

（另外，使用一级品类进行分类试验，奶粉的只有三个，正确率高达0.9558334441931622，还发现了很多平台维护错的品类信息）

以上是基于普通的 Spark MLlib 来实现文本分类，下面提供 Spark MLlib 一个重要的特性——Pipline 来进行实现。

```scala
//前面的文本处理部分一致
...
val tokenizer = new Tokenizer().setInputCol("wordsFearture").setOutputCol("words")

val hashingTF = new HashingTF()
  .setInputCol("words").setOutputCol("rawFeatures")

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

val nb = new NaiveBayes()

//这里和之前不同的是所有的步骤都在 pipline 中进行转化，它的 stages 使得每一步都很清晰，处理步骤一目了然
val pipeline = new Pipeline()
  .setStages(Array(tokenizer, hashingTF,idf,nb))

//网格参数使得超参数调优更加的方便，只需要在网格中加入可能的参数，最终按照你添加的参数训练出一个最优的模型
val paramGrid = new ParamGridBuilder()
  .addGrid(nb.smoothing, Array(0.5, 1,1.5))
  .build()

//将所有的步骤加入到 TrainValidationSplit 中，包括 训练器、评估方法、模型的网格参数、并行度等
val trainValidationSplit = new TrainValidationSplit()
  .setEstimator(pipeline)
  .setEvaluator(new MulticlassClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setTrainRatio(0.8)
  .setParallelism(2)
val Array(training, test) = wordset.randomSplit(Array(0.8, 0.2), seed = 12345)
// 运行上面的TrainValidationSplit，最终会选择一个最优的模型进行输出，大大方便了调优的部分
val model = trainValidationSplit.fit(training)

val predictions = model.transform(test)

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test set accuracy = $accuracy")
```



