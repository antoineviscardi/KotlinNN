import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.nd4j.linalg.factory.Nd4j
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe

object TestMathUtils : Spek ({
    val preds = Nd4j.create(doubleArrayOf(12.0, 7.0, 8.0, 4.0))
    val labels = Nd4j.create(doubleArrayOf(1.0, 0.0, 1.0, 1.0))
    val cost = MathUtils.mse(preds, labels)
    val dCost = MathUtils.dMse(preds, labels)

    val batchPreds = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)).reshape(3, 3)
    val batchLabels = Nd4j.create(doubleArrayOf(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)).reshape(3, 3)
    val batchCost = MathUtils.mse(batchPreds, batchLabels)
    val batchDCost = MathUtils.dMse(batchPreds, batchLabels)

    describe("Compute cost") {
        it("Should correctly compute the MSE") {
            assertEquals(114.0, cost.getDouble(0, 0))
        }

        it("Should correctly compute the MSE for a batch") {
            val ans = Nd4j.create(doubleArrayOf(58.0, 4.0, 58.0)).reshape(-1, 1)
            assertTrue(ans.equals(batchCost))
            assertTrue(ans.equalShapes(batchCost))
        }
    }

    describe("Compute cost derivative") {
        it("Should correctly compute the MSE derivative") {
            val dCostReal = Nd4j.create(doubleArrayOf(11.0, 7.0, 7.0, 3.0))
            assertEquals(dCostReal, dCost)
        }

        it("Should correctly compute the MSE derivatives for a batch") {
            val ans = Nd4j.create(doubleArrayOf(-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)).reshape(3, 3)
            assertTrue(ans.equals(batchDCost))
            assertTrue(ans.equalShapes(batchDCost))
        }
    }

    describe("Compute the sigmoid derivative") {
        it("Should correctly compute the element wise sigmoid derivative of a matrix") {
            val input = Nd4j.zeros(3, 3)
            val expected = Nd4j.zeros(3, 3).addi(0.25)
            val output = MathUtils.dSigmoid(input)
            assertTrue(output.equals(expected))
            assertTrue(output.equalShapes(expected))
        }
    }
})