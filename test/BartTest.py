import unittest

from Math.Tensor import Tensor

from ComputationalGraph.Function.CrossEntropyLoss import CrossEntropyLoss
from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.Optimizer.AdamW import AdamW

from Dictionary.VectorizedDictionary import VectorizedDictionary

from SequenceProcessing.Classification.Bart import Bart
from SequenceProcessing.Functions.GeLU import GeLU
from SequenceProcessing.Parameters.TransformerParameter import TransformerParameter


class DummyWordComparator:
    """
    Dummy comparator implementation.
    """

    def compare(self, word, word1) -> int:
        return 0


class BartTest(unittest.TestCase):

    def testInitialization(self):
        """
        Tests BART initialization and training.

        gamma/beta counts for num_layers=1:
          input  (encoder): 2 * 1 + 1 = 3  (2 per layer + 1 final LN)
          output (decoder): 3 * 1 + 1 = 4  (3 per layer + 1 final LN)
        """
        tensors = [
            Tensor(
                [
                    0.2, 0.7, 0.1, 0.3, 0.4, 0.8, 0.9,
                    0.35, 0.12, 0.27, 0.17, 0.41,
                    float("inf"),
                    0.27, 0.67, 0.41, 1,
                    0.37, 0.17, 0.41, 6,
                    0.17, 0.65, 0.87, 5,
                    0.97, 0.19, 0.51, 4
                ],
                (29,)
            ),
            Tensor(
                [
                    0.2, 0.7, 0.1, 0.3, 0.4, 0.8, 0.9,
                    0.35, 0.12, 0.27, 0.17, 0.41,
                    float("inf"),
                    0.27, 0.67, 0.41, 1,
                    0.37, 0.17, 0.41, 6,
                    0.77, 0.61, 0.27, 2
                ],
                (25,)
            ),
            Tensor(
                [
                    0.2, 0.7, 0.1, 0.3, 0.4, 0.8, 0.9,
                    0.35, 0.12, 0.27, 0.17, 0.41,
                    float("inf"),
                    1.2, 3.6, 7.1, 3,
                    5.4, 0.17, 9.8, 4,
                    0.77, 0.61, 0.27, 2
                ],
                (25,)
            )
        ]

        # num_layers=1 → encoder needs 3 input LN calls, decoder needs 4 output LN calls
        gamma_input = [1.0, 1.0, 1.0]
        gamma_output = [1.0, 1.0, 1.0, 1.0]
        beta_input = [0.0, 0.0, 0.0]
        beta_output = [0.0, 0.0, 0.0, 0.0]

        parameter = TransformerParameter(
            seed=1,
            epoch=5,
            optimizer=AdamW(0.025, 0.99, 0.99, 0.999, 1e-10, 0.1),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            word_embedding_length=3,
            multi_head_attention_length=2,
            vocabulary_length=7,
            epsilon=1e-9,
            input_hidden_layers=[30, 15],
            output_hidden_layers=[30, 15],
            input_activation_functions=[GeLU(), GeLU()],
            output_activation_functions=[GeLU(), GeLU()],
            gamma_input_values=gamma_input,
            gamma_output_values=gamma_output,
            beta_input_values=beta_input,
            beta_output_values=beta_output,
            num_layers=1
        )

        dictionary = VectorizedDictionary(DummyWordComparator())
        model = Bart(parameter, dictionary)
        model.train(tensors)


if __name__ == "__main__":
    unittest.main()
