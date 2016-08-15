package org.apache.spark.examples.mllib

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.newtree.GradientBoostedTrees
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

// $example on$
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.configuration.Algo._
// $example off$

object GradientBoostingClassificationExample {
  case class Params( trainingData: String = null,
                     testData: String = null,
                     maxDepth: Int = 5,
                     numIterations: Int = 10,
                     numBins: Int = 64,
                     samplePercent: Double = 1.0,
                     numFeature: Int = 0,
                     numPartitions: Int = 160) extends AbstractParams[Params]

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("GradientBoostedTrees") {
      head("GradientBoostedTrees: an example decision tree app.")

      opt[String]("trainingData")
        .text("input path to labeled examples")
        .required()
        .action((x, c) => c.copy(trainingData = x))
      opt[String]("testData")
        .text("input path to labeled examples")
        .action((x, c) => c.copy(testData = x))
      opt[Int]("maxDepth")
        .text(s"max depth of the tree, default: ${defaultParams.maxDepth}")
        .action((x, c) => c.copy(maxDepth = x))
      opt[Int]("numIterations")
        .text(s"number of iterations of boosting," + s" default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("numBins")
        .text(s"number of bins," + s" default: ${defaultParams.numBins}")
        .action((x, c) => c.copy(numBins = x))
      opt[Double]("samplePercent")
        .text(s"sample percent," + s" default: ${defaultParams.samplePercent}")
        .action((x, c) => c.copy(samplePercent = x))
      opt[Int]("numFeature")
        .text(s"number of feature used for training," + s" default: ${defaultParams.numFeature}")
        .action((x, c) => c.copy(numFeature = x))
      opt[Int]("numPartitions")
        .text(s"number of partitions," + s" default: ${defaultParams.numPartitions}")
        .action((x, c) => c.copy(numPartitions = x))

    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      sys.exit(1)
    }
  }
  def run(params: Params) {
    println(s"mlLib classification with parameters:\n${params.toString}")


    val conf = new SparkConf().setAppName(s"mlLib Classification with ${params.toString}")
    val sc = new SparkContext(conf)
    // $example on$
    // Load and parse the data file.

    val trainingData = if (params.numFeature == 0 && params.samplePercent == 1.0)
      MLUtils.loadLibSVMFile(sc, params.trainingData).coalesce(params.numPartitions)
    else if(params.numFeature==0 && params.samplePercent!=1.0)
      MLUtils.loadLibSVMFile(sc, params.trainingData).sample(true, params.samplePercent).coalesce(params.numPartitions)
    else if(params.numFeature!=0 && params.samplePercent == 1.0)
      loadLibSVMFile(sc, params.trainingData, params.numFeature, sc.defaultMinPartitions).coalesce(params.numPartitions)
    else
      loadLibSVMFile(sc, params.trainingData, params.numFeature, sc.defaultMinPartitions)
      .sample(true, params.samplePercent).coalesce(params.numPartitions)


    println(s"numSamples: ${trainingData.count()}")
    // Split the data into training and test sets (30% held out for testing)

    // Train a GradientBoostedTrees model.
    // The defaultParams for Classification use LogLoss by default.
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = params.numIterations // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.numClasses = 2

    boostingStrategy.treeStrategy.maxDepth = params.maxDepth

    boostingStrategy.treeStrategy.maxBins = params.numBins
    boostingStrategy.treeStrategy.maxMemoryInMB = 4096
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    boostingStrategy.learningRate=1.0


    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    if(params.testData!=null) {
      val testData = MLUtils.loadLibSVMFile(sc, params.testData)
      println(s"numSamples: ${testData.count()}")
      val scoreAndLabels = testData.map { point =>
        val predictions =  model.trees.map(_.predict(point.features))
        val claPred = new Array[Double](predictions.length)
        Range(1,predictions.length).foreach{it=>
          predictions(it)+=predictions(it-1)
          if(predictions(it)>=0) claPred(it)=1.0
          else claPred(it)=0.0
        }

        (claPred, point.label)
      }

      Range(0,model.trees.length, 1).foreach{it=>
        val scoreLabel = scoreAndLabels.map{case(claPred, lb)=>(claPred(it),lb)}
        val metrics = new BinaryClassificationMetrics(scoreLabel)
        val accuracy = metrics.areaUnderROC()
        println(s"test Accuracy $it = $accuracy")
      }

    }
  }

  def loadLibSVMFile(sc: SparkContext,
                     path: String,
                     numFeatures: Int,
                     minPartitions: Int): RDD[LabeledPoint] = {
    val parsed = sc.textFile(path, minPartitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
        val items = line.split(' ')
        val label = items.head.toDouble
        val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
          val indexAndValue = item.split(':')
          val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
        val value = indexAndValue(1).toDouble
          (index, value)
        }.filter(_._1 < numFeatures).unzip

        // check if indices are one-based and in ascending order
        var previous = -1
        var i = 0
        val indicesLength = indices.length
        while (i < indicesLength) {
          val current = indices(i)
          require(current > previous, "indices should be one-based and in ascending order")
          previous = current
          i += 1
        }

        (label, indices.toArray, values.toArray)
      }

    // Determine number of features.
    val d = numFeatures

    parsed.map { case (label, indices, values) =>
      LabeledPoint(label, Vectors.sparse(d, indices, values))
    }
  }
}
// scalastyle:on println