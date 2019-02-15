import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

class Preprocessor (val path: String) {

    private val rawData = Nd4j.readNumpy(ClassLoader.getSystemResource("wine.data").path, ",")
    private val dataset = DataSet(rawData.get(NDArrayIndex.all(), NDArrayIndex.interval(1, rawData.columns())), rawData.getColumn(0))

    fun foo() {

        // One-hot encode labels
        val nClass = dataset.labels.maxNumber().toInt()
        for (i in 1..nClass) {
            println(Nd4j.hstack(dataset.labels, dataset.labels.getColumn(0).eq(i)))
            dataset.labels = Nd4j.hstack(dataset.labels, dataset.labels.getColumn(0).eq(i))
        }
        dataset.labels = dataset.labels.getColumns(*IntArray(nClass) {it + 1})


        // Shuffle dataset
        dataset.shuffle()

        // Split into train and test sets
        val testAndTrain = dataset.splitTestAndTrain(0.70)
        val trainSet = testAndTrain.train
        val testSet = testAndTrain.test

        // Normalize data
        val normalizer = NormalizerStandardize()
        normalizer.fit(trainSet)
        normalizer.transform(trainSet)
        normalizer.transform(testSet)

    }
}

fun main() {
    val preprocessor = Preprocessor("")
    preprocessor.foo()
}