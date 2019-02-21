import org.nd4j.linalg.factory.Nd4j
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue

object TestNetwork : Spek({

    val net = Network(2, 3, 2)
    net.weights = arrayOf(Nd4j.ones(3, 2), Nd4j.ones(2, 3))
    net.biases = arrayOf(Nd4j.zeros(3, 1), Nd4j.zeros(2, 1).sub(1.5))

    describe("Forward pass") {

        it("Should be able to feed forward") {
            val output = net.feedForward(Nd4j.zeros(1, 2))
            val expected = Nd4j.zeros(1, 2).add(0.5)
            assertTrue(expected.equals(output))
            assertTrue(expected.equalShapes(output))
        }

        it("Should be able to do backward propagation") {

        }
    }
})