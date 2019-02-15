import org.nd4j.linalg.api.ndarray.INDArray
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


}

fun main() {
    nd.getRandom().setSeed(42)

    val data = nd.readNumpy("data/wine.data", ",")
    println(data.get(NDArrayIndex.interval(0, 10)))

    val network = Network(2, 3, 2)
    val input = nd.ones(2, 1)
    val out = network.feedForward(input)
    print(out)
}

//[0.8053,
//0.8020]