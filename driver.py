import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._
import org.apache.spark.mllib.clustering._

val text = sc.textFile("input")

val header = text.first()
val filterData = text.filter(x => x != header)

val parseData = filterData.map(s => Vector.dense(s.split(',').drop(1).map(x => x.toDouble)))

val kmeams = new KMeans()
kmeans.setK(2)
val model = kmeans.run(parseData)
model.predict(parseData)
model.predict(parseData).foreach(println)
val testdata = model.predict(Vectors.dense(70, 25))
println(testdata)