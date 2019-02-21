import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

/*
Utility object containing mathematical functions
 */

object MathUtils {

    fun INDArray.pow(e: Number) = Transforms.pow(this, e)
    fun INDArray.sigmoid() = Transforms.sigmoid(this)

    fun mse(preds: INDArray, labels: INDArray) : INDArray = preds.sub(labels).pow(2).sum(1).div(2)

    fun avgMse(preds: INDArray, labels: INDArray) : Double = mse(preds, labels).sum(0).div(labels.rows()).getDouble(0, 0)

    fun dMse(preds: INDArray, labels: INDArray): INDArray = preds.sub(labels)

    fun dSigmoid(a: INDArray): INDArray = a.sigmoid().mul(a.sigmoid().rsub(1))
}