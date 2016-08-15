/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark.examples.mllib

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.impl.{TimeTracker, DecisionTreeMetadata}
import org.apache.spark.mllib.tree.newtree.DecisionTree
import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.util.MLUtils
// $example off$

object GBDT {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("BinTimeMLlib")
    val sc = new SparkContext(conf)
    // $example on$
    // Load and parse the data file.
    val input = MLUtils.loadLibSVMFile(sc, "/data/lambdaMart/outdata/test10_libsvm")
    println("finish load")

    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.maxDepth = 5
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    boostingStrategy.treeStrategy.minInstancesPerNode = 2000

    boostingStrategy.treeStrategy.maxBins = 64
    while (boostingStrategy.treeStrategy.maxBins <= 1024) {
      val strategy = boostingStrategy.treeStrategy.copy
      println(s"numBins: ${boostingStrategy.treeStrategy.maxBins}")
      val retaggedInput = input.retag(classOf[LabeledPoint])
      val metadata =
        DecisionTreeMetadata.buildMetadata(retaggedInput, strategy, 1, "all")

      val timer = new TimeTracker()
      timer.start("findSplitsBins")
      val (splits, bins) = DecisionTree.findSplitsBins(retaggedInput, metadata)
      timer.stop("findSplitsBins")

      println(s"$timer")

      boostingStrategy.treeStrategy.maxBins= boostingStrategy.treeStrategy.maxBins * 2
    }
  }
}
// scalastyle:on println