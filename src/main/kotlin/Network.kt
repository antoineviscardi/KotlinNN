import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.factory.Nd4j as nd
import org.nd4j.linalg.ops.transforms.Transforms.*
import java.lang.IllegalArgumentException
import java.util.HashMap
import kotlin.math.pow



/*
M refers to the size of the batch
N refers to the dimensionality of the layer
 */
class Network(private vararg val dimensions: Int) {

    private val weights = Array<INDArray>(dimensions.size - 1) { nd.rand(dimensions[it + 1], dimensions[it]) }
    private val biases = Array<INDArray>(dimensions.size - 1) {nd.rand(dimensions[it + 1], 1)}
    private val weightedInputs = Array<INDArray>(dimensions.size - 1) { nd.zeros(1, dimensions[it]) }
    private val activations = Array<INDArray>(dimensions.size) { nd.zeros(1, dimensions[it]) }

    /*
    input: M x N matrix
     */
    fun feedForward(input: INDArray): INDArray {
        if (input.columns() != dimensions[0]) {
            throw IllegalArgumentException("Was expecting input of dimension ${dimensions[0]} but got ${input.columns()} instead")
        }
        activations[0] = input
        for (i in 0 until dimensions.size - 1) {
            weightedInputs[i] = activations[i].mmul(weights[i].transpose()).addRowVector(biases[i].transpose())
            activations[i + 1] = sigmoid(weightedInputs[i])
        }
        return activations.last()
    }

    fun mse(preds: INDArray, labels: INDArray) : INDArray {
        val diff = preds.sub(labels)
        val sqr = pow(diff, 2)
        return sqr.sum(0).divi(labels.rows())
    }

    fun dMse(preds: INDArray, labels: INDArray): INDArray {
        return preds.sub(labels)
    }

    fun dSigmoid(a: INDArray): INDArray {
        return sigmoid(a).mul(sigmoid(a).rsub(1))
    }

    fun computeGradientsChecking() : Gradients {
        return Gradients(arrayOf(nd.empty()), arrayOf(nd.empty()))
    }

    fun computeGradients(preds: INDArray, labels: INDArray) : Gradients {

        // Compute the output error signal
        var errorSignal = dMse(preds, labels).mul(dSigmoid(activations.last()))

        var gradWeights = Array<INDArray>(dimensions.size - 1) { nd.empty() }
        var gradBiases = Array<INDArray>(dimensions.size - 1) { nd.empty() }

        // Iteratively compute each layer's parameters gradients while propagating the error signal
        for (i in dimensions.size - 2 downTo 0) {
            gradWeights.set(i, activations[i].transpose().mmul(errorSignal).div(errorSignal.rows()))
            gradBiases.set(i, errorSignal.sum(0).div(errorSignal.rows()))
            errorSignal = errorSignal.mmul(weights[i]).mul(dSigmoid(activations[i]))
        }

        return Gradients(gradWeights, gradBiases)
    }

    /*
    preds and labels should be a M x N matrices where M is the size of the batch and N the dimensionality of the output.
     */
    fun backprop(preds: INDArray, labels: INDArray, learningRate: Double) {
        
    }

    fun train(trainSet: DataSet) {

    }

}


fun main() {
    nd.getRandom().setSeed(42)

    // Fetch preprocessed data
    val testAndTrain = Preprocessor("wine.data").getPreprocessedData()
    val trainSet = testAndTrain.train
    val testSet = testAndTrain.test


    // Instantiate network
    val network = Network(2, 3, 2)


    // TESTING
    val input = nd.create(doubleArrayOf(1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0)).reshape(-1, 2)
    val labels = nd.create(doubleArrayOf(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)).reshape(-1, 2)

    val preds = network.feedForward(input)
    val grads = network.computeGradients(preds, labels)
    println(grads)



}


/*
Data class that holds the gradients for the different parameters.
 */
data class Gradients(val weights: Array<INDArray>, val biases: Array<INDArray>) {

    /*
    Utility function that returns true if gradients are approximately equal. That is, the difference between each
    corresponding element is not more than 10^-4
     */
    fun isAproxEqual(other: Gradients): Boolean {

        val eps = 10.0.pow(-4)

        // Compare weights
        for (i in 0 until weights.size) {
            if (!weights[i].equalsWithEps(other.weights[i], eps)) return false
        }

        // Compare biases
        for (i in 0 until biases.size) {
            if (!biases[i].equalsWithEps(other.biases[i], eps)) return false
        }

        return true
    }
}
