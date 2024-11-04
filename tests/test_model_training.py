import unittest
import numpy as np
from src.model_training import build_model

class ModelTrainingTest(unittest.TestCase):
    def test_model_output_shape(self):
        model = build_model()
        input_data = np.random.rand(1, 224, 224, 3)
        output = model(input_data)
        self.assertEqual(output.shape, (1, 2))  # Assuming binary classification

if __name__ == '__main__':
    unittest.main()
