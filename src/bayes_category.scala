import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.apache.log4j.{Level, Logger}
import java.io._


object bayes_category{

  Logger.getLogger("org").setLevel(Level.WARN)


  case class RawDataRecord(category: String, text: String)

  def main(args : Array[String]) {

    val conf = new SparkConf().setMaster("local[4]").setAppName("app")

    val sc = new SparkContext(conf)

    sc.setLogLevel("WARN")

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    //建立训练rdd， 测试rdd， 预测rdd
    var srcRDD = sc.textFile("E:\\spark-test-py\\machinespark\\spark-createTag\\sougou_all").map {
      x =>
        var data = x.split(",")
        RawDataRecord(data(0),data(1))
    }

    var videoRDD = sc.textFile("./file/0.txt").map { x=> var data = x.split(",")
      RawDataRecord(data(0), data(1))}


    //70%作为训练数据，30%作为测试数据
    val splits = srcRDD.randomSplit(Array(0.7, 0.3))
    var trainingDF = splits(0).toDF()
    var testDF = splits(1).toDF()
    var videoDF = videoRDD.toDF()


    //将词语转换成数组
    var tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    var wordsData = tokenizer.transform(trainingDF)
    println("output1：词转数组")
    var wordsData_result = wordsData.select($"category", $"text", $"words").take(1)
    println(wordsData_result.head)

    //计算每个词在文档中的词频
    var hashingTF = new HashingTF().setNumFeatures(500000).setInputCol("words").setOutputCol("rawFeatures")
    var featurizedData = hashingTF.transform(wordsData)
    println("output2：计算词频")
    println(featurizedData.select($"category", $"words", $"rawFeatures").take(1).head)


    //计算每个词的TF-IDF
    var idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    var idfModel = idf.fit(featurizedData)
    // idfModel.save("./model/IDFModel") 如果没有保存模型，可以使用save保存IDF模型
    var rescaledData = idfModel.transform(featurizedData)
    println("output3：计算TF-IDF")
    println(rescaledData.select($"category", $"features").take(1).head)

    //转换成Bayes的输入格式
    var trainDataRdd = rescaledData.select($"category",$"features").map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }


    //训练模型
    val model = NaiveBayes.train(trainDataRdd, lambda = 1.0, modelType = "multinomial")
    // model.save(sc, "./model/BayesModel") 如果没有保存模型， 可以使用save保存Bayes模型

    //测试数据集，做同样的特征表示及格式转换
    var testwordsData = tokenizer.transform(testDF)
    var testfeaturizedData = hashingTF.transform(testwordsData)
    var testrescaledData = idfModel.transform(testfeaturizedData)
    var testDataRdd = testrescaledData.select($"category",$"features").map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }

    //video测试集 同样做特征转换
    var videowordsData = tokenizer.transform(videoDF)
    var videofeaturizedData = hashingTF.transform(videowordsData)
    var videorescaledData = idfModel.transform(videofeaturizedData)
    var videoDataRdd = videorescaledData.select($"category", $"features").map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))

    }


    //对测试数据集使用训练模型进行分类预测
    val testpredictionAndLabel = testDataRdd.map(p => (model.predict(p.features), p.label))
    //对video测试集数据使用训练模型进行分类预测
    val videopredictionAndLabel = videoDataRdd.map(p => (model.predict(p.features), p.label))

    println("video测试集情况")
    println(videopredictionAndLabel.collect().foreach(println))

    //统计分类准确率
    var testaccuracy = 1.0 * testpredictionAndLabel.filter(x => x._1 == x._2).count() / testDataRdd.count()
    println("准确率是:")
    println(testaccuracy)



  }
}