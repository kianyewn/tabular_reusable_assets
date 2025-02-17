import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from example_project.src.features.preprocessing.categorical_encoder import CategoricalEncoder


class TestCategoricalEncoder(unittest.TestCase):
    def setUp(self):
        self.categorical_columns = ["category", "color"]
        self.df = pd.DataFrame(
            {"category": ["A", "B", "C", "A"], "color": ["red", "blue", "red", "green"], "value": [1, 2, 3, 4]}
        )
        self.encoder = CategoricalEncoder(categorical_columns=self.categorical_columns)

    def test_save_load_preserves_state(self):
        # First fit the encoder
        self.encoder.fit(self.df)

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Save the encoder
            self.encoder.save(Path(tmp.name))

            # Load the encoder
            loaded_encoder = CategoricalEncoder.load(Path(tmp.name))

            # Verify all attributes are preserved
            self.assertEqual(self.encoder.categorical_columns, loaded_encoder.categorical_columns)
            self.assertEqual(set(self.encoder.encoders.keys()), set(loaded_encoder.encoders.keys()))

            # Test that both encoders transform data the same way
            original_transform = self.encoder.transform(self.df)
            loaded_transform = loaded_encoder.transform(self.df)
            pd.testing.assert_frame_equal(original_transform, loaded_transform)

            # # Test that both encoders inverse transform the same way
            # original_inverse = self.encoder.inverse_transform(original_transform)
            # loaded_inverse = loaded_encoder.inverse_transform(loaded_transform)
            # pd.testing.assert_frame_equal(original_inverse, loaded_inverse)

        # Clean up the temporary file
        os.unlink(tmp.name)

    def test_save_load_with_empty_encoder(self):
        # Test saving/loading without fitting first
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.encoder.save(Path(tmp.name))
            loaded_encoder = CategoricalEncoder.load(Path(tmp.name))

            self.assertEqual(self.encoder.categorical_columns, loaded_encoder.categorical_columns)
            self.assertEqual(self.encoder.encoders, loaded_encoder.encoders)

        os.unlink(tmp.name)

    def test_encoder_directory_creation(self):
        # Test that save creates directories if they don't exist
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "subdir" / "encoder.pkl"
            self.encoder.fit(self.df)
            self.encoder.save(save_path)

            self.assertTrue(save_path.exists())
            loaded_encoder = CategoricalEncoder.load(save_path)
            self.assertEqual(self.encoder.categorical_columns, loaded_encoder.categorical_columns)


if __name__ == "__main__":
    unittest.main()
