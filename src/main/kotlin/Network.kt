import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.factory.Nd4j as nd
import org.nd4j.linalg.ops.transforms.Transforms.*

class Network(private vararg val dimensions: Int) {
    private val activations = Array<INDArray>(dimensions.size) { nd.zeros(1, dimensions[it]) }
    private val weights = Array<INDArray>(dimensions.size - 1) { nd.rand(dimensions[it + 1], dimensions[it]) }
    private val biases = Array<INDArray>(dimensions.size - 1) {nd.rand(dimensions[it + 1], 1)}

    fun feedForward(input: INDArray): INDArray {
        activations[0] = input
        for (i in 0 until dimensions.size - 1) {
            activations[i + 1] = sigmoid(weights[i].mmul(activations[i]).add(biases[i]))
        }
        softmax(activations.last(), false)
        return activations.last()
    }

    fun mse(pred: INDArray, label: INDArray): INDArray {
        val diff = pred.sub(label)
        val result = pow(diff, 2)
        return result
    }

    fun dMse(pred: INDArray, label: INDArray): INDArray {
        return pred.sub(label)
    }

    fun dSoftmax(a: INDArray): INDArray {
        // TODO: Implement this!
        return nd.empty()
    }

    fun dSigmoid(a: INDArray): INDArray {
        return sigmoid(a).mul(sigmoid(a).rsub(1))
    }

    fun backprop(learningRate: Double) {
        for (i in dimensions.size - 1 downTo 0) {

        }
    }

    fun train(trainSet: DataSet) {

    }


}

fun main() {
    nd.getRandom().setSeed(42)

//    val m1 = nd.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0))
//    println(pow(m1, 2))

    // Fetch preprocessed data
    val testAndTrain = Preprocessor("wine.data").getPreprocessedData()
    val trainSet = testAndTrain.train
    val testSet = testAndTrain.test


    // Instantiate network
    val network = Network(2, 3, 2)



}

//[0.8053,
//0.8020]