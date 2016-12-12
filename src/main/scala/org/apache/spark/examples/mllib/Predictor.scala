package org.apache.spark.examples.mllib

import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by v-cuili on 10/17/2016.
  */
class Predictor {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("GradientBoostedTreesRegressionExample")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc, "data/lambdaMart/outdata/testData.txt")


  }

}
