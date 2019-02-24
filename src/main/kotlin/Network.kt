import org.nd4j.linalg.api.iter.NdIndexIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j as nd
import MathUtils.sigmoid
import org.nd4j.linalg.api.buffer.DataBuffer
import java.lang.IllegalArgumentException
import kotlin.math.pow
import kotlin.random.Random

// TODO: Test on other dataset (maybe MNIST?)
// TODO: Write documentation

/*
M refers to the size of the batch
N refers to the dimensionality of the layer
 */
class Network(private vararg val dimensions: Int) {

    init { nd.setDataType(DataBuffer.Type.DOUBLE) }

    var weights = Array<INDArray>(dimensions.size - 1) { nd.rand(dimensions[it + 1], dimensions[it]) }
        set(value) {
            if (value.size != weights.size)
                throw IllegalArgumentException("The list of weight matrices has the wrong size")
            for ((a, b) in weights.zip(value)) {
                if (!a.equalShapes(b))
                    throw IllegalArgumentException("The weight matrices do not conform to the network dimensions")
            }
            field = value
        }

    var biases = Array<INDArray>(dimensions.size - 1) { nd.rand(dimensions[it + 1], 1) }
        set(value) {
            if (value.size != biases.size)
                throw IllegalArgumentException("The list of biases vectors has the wrong size")
            for ((a, b) in biases.zip(value)) {
                if (!a.equalShapes(b))
                    throw IllegalArgumentException("The biases vectors do not conform to the network dimensions")
            }
            field = value
        }

    private val weightedInputs = Array<INDArray>(dimensions.size - 1) { nd.zeros(1, dimensions[it]) }
    private val activations = Array<INDArray>(dimensions.size) { nd.zeros(1, dimensions[it]) }

    /*
    input: M x N matrix
     */
    fun feedForward(input: INDArray): INDArray {
        if (input.columns() != dimensions[0]) {
            throw IllegalArgumentException("Was expecting input of dimension ${dimensions[0]} but got ${input.columns()} instead")
        }
        activations[0] = input.convertToDoubles()
        for (i in 0 until dimensions.size - 1) {
            weightedInputs[i] = activations[i].mmul(weights[i].transpose()).addRowVector(biases[i].transpose())
            activations[i + 1] = weightedInputs[i].sigmoid()
        }
        return activations.last()
    }

    fun computeGradients(preds: INDArray, labels: INDArray) : Gradients {

        // Compute the output error signal
        var errorSignal = MathUtils.dMse(preds, labels).mul(MathUtils.dSigmoid(weightedInputs.last()))

        val gradWeights = Array<INDArray>(dimensions.size - 1) { nd.empty(DataBuffer.Type.DOUBLE) }
        val gradBiases = Array<INDArray>(dimensions.size - 1) { nd.empty(DataBuffer.Type.DOUBLE) }

        // Iteratively compute each layer's parameters gradients while propagating the error signal
        for (i in dimensions.size - 2 downTo 0) {
            gradWeights[i] = errorSignal.transpose().mmul(activations[i]).div(errorSignal.rows())
            gradBiases[i] = errorSignal.sum(0).div(errorSignal.rows()).transpose()
            if (i == 0) break
            errorSignal = errorSignal.mmul(weights[i]).mul(MathUtils.dSigmoid(weightedInputs[i-1]))
        }

        return Gradients(gradWeights, gradBiases)
    }

    fun performGradientChecking(input: INDArray, label: INDArray, eps: Double = 1E-3) : Boolean {

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
                val costPlus = MathUtils.avgMse(feedForward(input), label)

                // Compute cost at parameter - eps
                weights[i].putScalar(index, initialValue - eps)
                val costMinus = MathUtils.avgMse(feedForward(input), label)

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
                val costPlus = MathUtils.avgMse(feedForward(input), label)

                // Compute cost at parameter - eps
                biases[i].putScalar(index, initialValue - eps)
                val costMinus = MathUtils.avgMse(feedForward(input), label)

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
            val cost = MathUtils.avgMse(feedForward(trainSet.features), trainSet.labels)
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
        return MathUtils.avgMse(feedForward(testSet.features), testSet.labels)
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

    // Gradient checking
    println(network.performGradientChecking(trainSet.get(0).features.convertToDoubles(), trainSet.get(0).labels))

    // Train network
    network.train(trainSet, 1500, 25, 0.1)

    // Test against test set
    println("\nTest mse: ${network.test(testSet)}")
    println("Test accuract: ${network.testAccuracy(testSet)}")

    // Prediction example
    val data = testSet.get(Random.nextInt(testSet.count()))
    val pred = network.feedForward(data.features)
    println("""
        |Input: ${data.features}
        |Prediction: $pred
        |Label: ${data.labels}
    """.trimMargin())

}
