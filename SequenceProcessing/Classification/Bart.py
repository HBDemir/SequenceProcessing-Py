from typing import List
import math
import random

from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from Math.Tensor import Tensor
from SequenceProcessing.Classification.Transformer import Transformer
from SequenceProcessing.Functions.MultiplyByConstant import MultiplyByConstant
from SequenceProcessing.Functions.Transpose import Transpose
from SequenceProcessing.Parameters.TransformerParameter import TransformerParameter


class Bart(Transformer):
    """
    BART (Bidirectional and Auto-Regressive Transformers) model.

    Encoder: num_layers × [Bidirectional Self-Attention + Add & Norm →
                            Feed Forward (Linear → GeLU → Linear) + Add & Norm]
             → Layer Norm → Encoder Output (Memory)

    Decoder: num_layers × [Masked Self-Attention + Add & Norm →
                            Cross-Attention (K, V from encoder) + Add & Norm →
                            Feed Forward + Add & Norm]
             → Layer Norm → Linear + Softmax → Output Token
    """

    def __feedforwardBlock(self,
                           current: ComputationalNode,
                           current_layer_size: int,
                           parameter: TransformerParameter,
                           random_generator: random.Random,
                           is_input: bool) -> ComputationalNode:
        """
        Builds the feed-forward block within an encoder or decoder layer.
        Applies hidden projections with activation functions, then a final
        linear projection back to L — without a Softmax at the end.

        :param current: Current input node.
        :param current_layer_size: Dimensionality of the current input.
        :param parameter: Transformer parameters.
        :param random_generator: Random generator for weight initialization.
        :param is_input: True for encoder-side, False for decoder-side.
        :return: Output node after linear projection back to L.
        """
        if is_input:
            size = parameter.getInputSize()
        else:
            size = parameter.getOutputSize()

        for i in range(size):
            if is_input:
                hidden_weight = MultiplicationNode(
                    Tensor(
                        parameter.initializeWeights(
                            current_layer_size,
                            parameter.getInputHiddenLayer(i),
                            random_generator
                        ),
                        (current_layer_size, parameter.getInputHiddenLayer(i))
                    )
                )
                hidden_layer = self.addEdge(current, hidden_weight)
                current = self.addEdge(hidden_layer, parameter.getInputActivationFunction(i), True)
                current_layer_size = parameter.getInputHiddenLayer(i) + 1
            else:
                hidden_weight = MultiplicationNode(
                    Tensor(
                        parameter.initializeWeights(
                            current_layer_size,
                            parameter.getOutputHiddenLayer(i),
                            random_generator
                        ),
                        (current_layer_size, parameter.getOutputHiddenLayer(i))
                    )
                )
                hidden_layer = self.addEdge(current, hidden_weight)
                current = self.addEdge(hidden_layer, parameter.getOutputActivationFunction(i), True)
                current_layer_size = parameter.getOutputHiddenLayer(i) + 1

        output_weight = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(current_layer_size, parameter.getL(), random_generator),
                (current_layer_size, parameter.getL())
            )
        )
        return self.addEdge(current, output_weight)

    def train(self, train_set: List[Tensor]) -> None:
        """
        Builds the BART computational graph and trains the model.

        Expected gamma/beta parameter counts:
          - gamma_input_values / beta_input_values : 2 * num_layers + 1 entries
          - gamma_output_values / beta_output_values: 3 * num_layers + 1 entries

        :param train_set: List of training tensors.
        """
        parameter = self.parameters
        ln_size = [0, 0, 0, 0]
        random_generator = random.Random(parameter.getSeed())

        # ── Encoder Stack ─────────────────────────────────────────────────────
        input1 = MultiplicationNode(False, True)
        self.input_nodes.append(input1)

        current_enc = input1
        for _ in range(parameter.getNumLayers()):
            # Bidirectional Multi-Head Self-Attention + Add & Norm
            concatenated_enc = self.concatEdges(
                self.multiHeadAttention(current_enc, parameter, False, random_generator),
                1
            )
            we = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getL(), random_generator),
                    (parameter.getL(), parameter.getL())
                )
            )
            c_enc = self.addEdge(concatenated_enc, we)
            input_c_enc = self.addAdditionEdge(current_enc, c_enc, False)
            y_enc = self.layerNormalization(input_c_enc, parameter, True, ln_size)

            # Feed Forward (Linear → GeLU → Linear) + Add & Norm
            ff_enc = self.__feedforwardBlock(y_enc, parameter.getL(), parameter, random_generator, True)
            oe = self.addAdditionEdge(ff_enc, y_enc, False)
            current_enc = self.layerNormalization(oe, parameter, True, ln_size)

        # Final Layer Norm after the full encoder stack
        encoder = self.layerNormalization(current_enc, parameter, True, ln_size)

        # ── Decoder Stack ─────────────────────────────────────────────────────
        input2 = MultiplicationNode(False, True)
        self.input_nodes.append(input2)

        current_dec = input2
        for _ in range(parameter.getNumLayers()):
            # Masked Self-Attention (causal) + Add & Norm
            concatenated_dec = self.concatEdges(
                self.multiHeadAttention(current_dec, parameter, True, random_generator),
                1
            )
            wd1 = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getL(), random_generator),
                    (parameter.getL(), parameter.getL())
                )
            )
            c_dec = self.addEdge(concatenated_dec, wd1)
            input_c_dec = self.addAdditionEdge(current_dec, c_dec, False)
            cd = self.layerNormalization(input_c_dec, parameter, False, ln_size)

            # Cross-Attention: Q from decoder, K and V from encoder + Add & Norm
            cross_nodes = []
            for _ in range(parameter.getN()):
                wk = MultiplicationNode(
                    Tensor(
                        parameter.initializeWeights(parameter.getL(), parameter.getDk(), random_generator),
                        (parameter.getL(), parameter.getDk())
                    )
                )
                k = self.addEdge(encoder, wk)

                wq = MultiplicationNode(
                    Tensor(
                        parameter.initializeWeights(parameter.getL(), parameter.getDk(), random_generator),
                        (parameter.getL(), parameter.getDk())
                    )
                )
                q = self.addEdge(cd, wq)

                wv = MultiplicationNode(
                    Tensor(
                        parameter.initializeWeights(parameter.getL(), parameter.getDk(), random_generator),
                        (parameter.getL(), parameter.getDk())
                    )
                )
                v = self.addEdge(encoder, wv)

                k_transpose = self.addEdge(k, Transpose())
                qk = self.addEdge(q, k_transpose, False, False)
                qk_dk = self.addEdge(qk, MultiplyByConstant(1.0 / math.sqrt(parameter.getDk())))
                s_qk_dk = self.addEdge(qk_dk, Softmax())
                attention = self.addEdge(s_qk_dk, v)
                cross_nodes.append(attention)

            cross_concat = self.concatEdges(cross_nodes, 1)
            wd2 = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getL(), random_generator),
                    (parameter.getL(), parameter.getL())
                )
            )
            cd_cross = self.addEdge(cross_concat, wd2)
            cd_cross_cd = self.addAdditionEdge(cd, cd_cross, False)
            yd = self.layerNormalization(cd_cross_cd, parameter, False, ln_size)

            # Feed Forward + Add & Norm
            ff_dec = self.__feedforwardBlock(yd, parameter.getL(), parameter, random_generator, False)
            od = self.addAdditionEdge(ff_dec, yd, False)
            current_dec = self.layerNormalization(od, parameter, False, ln_size)

        # Final Layer Norm after the full decoder stack
        final_dec = self.layerNormalization(current_dec, parameter, False, ln_size)

        # ── Linear + Softmax output projection ────────────────────────────────
        wdo = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(parameter.getL(), parameter.getV(), random_generator),
                (parameter.getL(), parameter.getV())
            )
        )
        decoder = self.addEdge(final_dec, wdo)
        self.output_node = self.addEdge(decoder, Softmax())

        class_label_node = ComputationalNode()
        self.input_nodes.append(class_label_node)

        loss_inputs = [self.output_node, class_label_node]
        self.addFunctionEdge(loss_inputs, parameter.getLossFunction(), False)

        # ── Training loop ─────────────────────────────────────────────────────
        for _ in range(parameter.getEpoch()):
            for _ in range(len(train_set)):
                i1 = random_generator.randint(0, len(train_set) - 1)
                i2 = random_generator.randint(0, len(train_set) - 1)
                train_set[i1], train_set[i2] = train_set[i2], train_set[i1]

            for instance in train_set:
                class_labels = self.createInputTensors(
                    instance,
                    self.input_nodes[0],
                    self.input_nodes[1],
                    parameter.getL() - 1
                )

                class_label_values = []
                for class_label in class_labels:
                    for j in range(parameter.getV()):
                        if j == class_label:
                            class_label_values.append(1.0)
                        else:
                            class_label_values.append(0.0)

                self.input_nodes[2].setValue(
                    Tensor(class_label_values, (len(class_labels), parameter.getV()))
                )

                self.forwardCalculation()
                self.backpropagation()

            parameter.getOptimizer().setLearningRate()
