import org.junit.jupiter.api.Assertions.assertEquals
import org.nd4j.linalg.factory.Nd4j
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe

object TestMathUtils : Spek ({
    val preds = Nd4j.create(doubleArrayOf(12.0, 7.0, 8.0, 4.0))
    val labels = Nd4j.create(doubleArrayOf(1.0, 0.0, 1.0, 1.0))
    val cost = MathUtils.mse(preds, labels)
    val dCost = MathUtils.dMse(preds, labels)


    describe("Compute cost") {
        it("Should correctly compute the MSE") {
            assertEquals(114.0, cost.getDouble(0, 0))
        }
    }

    describe("Compute cost derivative") {
        it("Should correctly compute the MSE derivative") {
            val dCostReal = Nd4j.create(doubleArrayOf(11.0, 7.0, 7.0, 3.0))
            assertEquals(dCostReal, dCost)
        }
    }
})