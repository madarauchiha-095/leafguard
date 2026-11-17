import unittest

from app.server import ModelService


class CheckpointIntegrityTests(unittest.TestCase):
    def test_packaged_checkpoints_match_reproducibility_manifest(self):
        results = ModelService().integrity()

        self.assertEqual(
            set(results),
            {"mobilenet", "mobilenet_clahe", "deit_tiny", "hybrid"},
        )
        invalid = [key for key, result in results.items() if not result["verified"]]
        self.assertEqual(invalid, [])


if __name__ == "__main__":
    unittest.main()
