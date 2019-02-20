import org.nd4j.linalg.api.iter.NdIndexIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.indexing.INDArrayIndex
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.factory.Nd4j as nd
import org.nd4j.linalg.ops.transforms.Transforms.*
import java.lang.IllegalArgumentException
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

    private fun mse(preds: INDArray, labels: INDArray) : Double {
        val diff = preds.sub(labels)
        val sqr = pow(diff, 2)
        return sqr.sum(1).divi(labels.columns()).sum(0).divi(labels.rows()).getDouble(0, 0)
    }

    private fun dMse(preds: INDArray, labels: INDArray): INDArray = preds.sub(labels)

    private fun dSigmoid(a: INDArray): INDArray = sigmoid(a).mul(sigmoid(a).rsub(1))

    private fun computeGradients(preds: INDArray, labels: INDArray) : Gradients {

        // Compute the output error signal
        var errorSignal = dMse(preds, labels).mul(dSigmoid(activations.last()))

        var gradWeights = Array<INDArray>(dimensions.size - 1) { nd.empty() }
        var gradBiases = Array<INDArray>(dimensions.size - 1) { nd.empty() }

        // Iteratively compute each layer's parameters gradients while propagating the error signal
        for (i in dimensions.size - 2 downTo 0) {
            gradWeights[i] = errorSignal.transpose().mmul(activations[i]).div(errorSignal.rows())
            gradBiases[i] = errorSignal.sum(0).div(errorSignal.rows()).transpose()
            errorSignal = errorSignal.mmul(weights[i]).mul(dSigmoid(activations[i]))
        }

        return Gradients(gradWeights, gradBiases)
    }

    fun performGradientChecking(input: INDArray, label: INDArray, eps: Double = 1E-7) : Boolean {

        // Compute gradient analytically
        val grads = computeGradients(feedForward(input), label)

        // Initialize gradients arrays
        val gradWeights = grads.weights.map { it.dup() }.toTypedArray()
        val gradBiases = grads.biases.map { it.dup() }.toTypedArray()

        // Compute gradients for weights using two sided method
        for (i in 0 until weights.size) {
            val iter = NdIndexIterator(weights[i].rows(), weights[i].columns())
            while (iter.hasNext()) {
                val index = iter.next()
                val initialValue = weights[i].getFloat(index)

                // Compute cost at parameter + eps
                weights[i].putScalar(index, initialValue + eps)
                val costPlus = mse(feedForward(input), label)

                // Compute cost at parameter - eps
                weights[i].putScalar(index, initialValue - eps)
                val costMinus = mse(feedForward(input), label)

                // Reset weight to initial value
                weights[i].putScalar(index, initialValue + eps)

                // Compute gradient
                val grad = (costPlus - costMinus) / (2 * eps)
                gradWeights[i].putScalar(index, grad)
            }
        }

        // Compute gradients of biases using two sided method
        for (i in 0 until biases.size) {
            val iter = NdIndexIterator(biases[i].rows(), biases[i].columns())
            while (iter.hasNext()) {
                val index = iter.next()
                val initialValue = biases[i].getFloat(index)

                // Compute cost at parameter + eps
                biases[i].putScalar(index, initialValue + eps)
                val costPlus = mse(feedForward(input), label)

                // Compute cost at parameter - eps
                biases[i].putScalar(index, initialValue - eps)
                val costMinus = mse(feedForward(input), label)

                // Reset weight to initial value
                biases[i].putScalar(index, initialValue)

                // Compute gradient
                val grad = (costPlus - costMinus) / (2 * eps)
                gradBiases[i].putScalar(index, grad)
            }
        }

        // Compare and return whether or not the test passed
        return grads.isAproxEqual(Gradients(gradWeights, gradBiases), eps)
    }

    fun train(trainSet: DataSet, nEpoch: Int, batchSize: Int, learningRate: Double) {
        for (epoch in 0 until nEpoch) {
            val cost = mse(feedForward(trainSet.features), trainSet.labels)
            println("Epoch: $epoch, Cost: $cost")
            for (batchRange in 0 until trainSet.count() step batchSize) {
                val batch = trainSet.getRange(batchRange, batchRange + batchSize)
                val preds = this.feedForward(batch.features)
                val grads  = this.computeGradients(preds, batch.labels)
                for (i in 0 until dimensions.size - 1) {
                    this.weights[i].subi(grads.weights[i].mul(learningRate))
                    this.biases[i].subi(grads.biases[i].mul(learningRate))
                }
            }
        }
    }

    fun test(testSet: DataSet): Double {
        return mse(feedForward(testSet.features), testSet.labels)
    }

    fun testAccuracy(testSet: DataSet): Double {
        val preds = feedForward(testSet.features)
        return preds.argMax(1).eq(testSet.labels.argMax(1)).sum(0).getDouble(0, 0) / testSet.count()
    }
}


/*
Data class that holds the gradients for the different parameters.
 */
data class Gradients(val weights: Array<INDArray>, val biases: Array<INDArray>) {

    /*
    Utility function that returns true if gradients are approximately equal. That is, the difference between each
    corresponding element is not more than 10^-4
     */
    fun isAproxEqual(other: Gradients, eps: Double = 10.0.pow(-7)): Boolean {

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


fun main() {
    nd.getRandom().setSeed(42)

    // Fetch preprocessed data
    val testAndTrain = Preprocessor("wine.data").getPreprocessedData()
    val trainSet = testAndTrain.train
    val testSet = testAndTrain.test


    // Instantiate network
    val network = Network(13, 35, 3)

    // TESTING
    println(network.performGradientChecking(trainSet.get(0).features, trainSet.get(0).labels))

    // Train network
    network.train(trainSet, 10000, 25, 0.6)

    // Test against test set
    println("Test mse: ${network.test(testSet)}")
    println("Test accuract: ${network.testAccuracy(testSet)}")

}
