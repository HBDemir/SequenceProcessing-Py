import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class GeLU(Function):
    """
    Gaussian Error Linear Unit (GeLU) activation function.
    GeLU(x) = x * Phi(x), where Phi is the standard normal CDF.
    """

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Computes GeLU element-wise over the input tensor.

        :param tensor: Input tensor.
        :return: GeLU-transformed tensor.
        """
        values = []
        shape = tensor.getShape()
        sqrt2 = math.sqrt(2.0)

        if len(shape) == 1:
            for i in range(shape[0]):
                x = tensor.getValue((i,))
                values.append(0.5 * x * (1.0 + math.erf(x / sqrt2)))
        else:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    x = tensor.getValue((i, j))
                    values.append(0.5 * x * (1.0 + math.erf(x / sqrt2)))

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the element-wise gradient of GeLU.

        :param value: Input tensor (pre-activation values).
        :param backward: Upstream gradient tensor.
        :return: Gradient tensor after applying chain rule.
        """
        values = []
        shape = value.getShape()
        sqrt2 = math.sqrt(2.0)
        sqrt2pi = math.sqrt(2.0 * math.pi)

        if len(shape) == 1:
            for i in range(shape[0]):
                x = value.getValue((i,))
                gelu_prime = (0.5 * (1.0 + math.erf(x / sqrt2))
                              + x * math.exp(-0.5 * x * x) / sqrt2pi)
                values.append(gelu_prime)
        else:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    x = value.getValue((i, j))
                    gelu_prime = (0.5 * (1.0 + math.erf(x / sqrt2))
                                  + x * math.exp(-0.5 * x * x) / sqrt2pi)
                    values.append(gelu_prime)

        return backward.hadamardProduct(Tensor(values, shape))

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.

        :param input_nodes: Input computational nodes.
        :param is_biased: Indicates whether the edge is biased.
        :return: Newly created computational node.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node
