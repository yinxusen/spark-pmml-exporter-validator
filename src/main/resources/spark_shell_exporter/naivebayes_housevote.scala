import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

// Load and parse the data
val data = sc.textFile("../datasets/house-votes-84.data")
val parsedData = data.map { line =>
  val parts = line.split(',')
  val label = if (parts.head == "democrat") 0.0 else 1.0
  LabeledPoint(label, Vectors.dense(parts.tail.map { x =>
    if (x == "y") 1.0
    else 0.0
  }))
}

// Build naive bayes model
val model = NaiveBayes.train(parsedData, lambda = 1.0, modelType = "bernoulli")

// Export naive bayes model to PMML
model.toPMML("../exported_pmml_models/naivebayes_housevote84.xml")
