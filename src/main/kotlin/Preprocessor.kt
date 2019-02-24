import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

class Preprocessor (val file: String) {

    init { Nd4j.setDataType(DataBuffer.Type.DOUBLE) }

    private val rawData = Nd4j.readNumpy(ClassLoader.getSystemResource(file).path, ",")
    private val dataset = DataSet(rawData.get(NDArrayIndex.all(), NDArrayIndex.interval(1, rawData.columns())), rawData.getColumn(0))

    fun getPreprocessedData(): SplitTestAndTrain {

        // One-hot encode labels
        val nClass = dataset.labels.maxNumber().toInt()
        for (i in 1..nClass) {
            dataset.labels = Nd4j.hstack(dataset.labels, dataset.labels.getColumn(0).eq(i))
        }
        dataset.labels = dataset.labels.getColumns(*IntArray(nClass) {it + 1})


        // Shuffle dataset
        dataset.shuffle()

        // Split into train and test sets
        val testAndTrain = dataset.splitTestAndTrain(0.70)

        // Normalize data
        val normalizer = NormalizerStandardize()
        normalizer.fit(testAndTrain.train)
        normalizer.transform(testAndTrain.train)
        normalizer.transform(testAndTrain.test)

        return testAndTrain
    }
}