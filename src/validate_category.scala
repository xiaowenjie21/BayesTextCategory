/**
  * Created by qiniu on 2017/7/10.
  */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.ml.feature.IDFModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.log4j.{Level, Logger}
import java.io._


object validate_category {
  // 类别的预测
  Logger.getLogger("org").setLevel(Level.WARN)

  case class RawDataRecord(category: String, text: String)

  def main(args: Array[String]) = {
    val conf2 = new SparkConf().setMaster("local").setAppName("app")
    val sc = new SparkContext(conf2)

    sc.setLogLevel("WARN")
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    var videoRDD = sc.textFile("./file/0.txt").map {
      x => var data = x.split(",")
        RawDataRecord(data(0), data(1))
    }

    var videoDF = videoRDD.toDF()
    val model = NaiveBayesModel.load(sc, "./model/BayesModel")


    //特征转换
    var tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    var hashingTF = new HashingTF().setNumFeatures(500000).setInputCol("words").setOutputCol("rawFeatures")
    var idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    var idf_model = IDFModel.load("./model/IDFModel")
    var videoData = tokenizer.transform(videoDF)
    var videofeaturizedData = hashingTF.transform(videoData)
    var videorescaledData = idf_model.transform(videofeaturizedData)
    var videoDataRdd = videorescaledData.select($"category", $"features").map{
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }

    val videopredictionAndLabel = videoDataRdd.map(p => (model.predict(p.features),
    p.label))

    println("video预测情况")

    // 将预测结果写入output.txt
    val writer = new PrintWriter(new File("output.txt"))
    var lines = videopredictionAndLabel.collect()
    println("lines")
    println(lines)
    for (i <- lines) {
      println(i)
      writer.println(i)
    }
    writer.close()
  }

}
