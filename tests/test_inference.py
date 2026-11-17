import unittest

from PIL import Image

from src.inference.common import IMAGE_SIZE, make_transform


class InferenceTests(unittest.TestCase):
    def test_preprocessing_produces_normalized_model_input(self):
        image = Image.new("RGB", (320, 180), color=(80, 120, 160))

        for use_clahe in (False, True):
            with self.subTest(clahe=use_clahe):
                tensor = make_transform(clahe=use_clahe)(image)

                self.assertEqual(tuple(tensor.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
                self.assertTrue(tensor.is_floating_point())
                self.assertTrue(tensor.isfinite().all().item())


if __name__ == "__main__":
    unittest.main()
