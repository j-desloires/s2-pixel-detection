import os
import logging
import unittest


class TestPipeline(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)

        test_rootdir = os.path.dirname(os.path.realpath(__file__))
        cloud_mask_rootdir = f"{test_rootdir}/../"

    def test_load(self):
        pass

    def test_file_experiments(self):
        pass

    def test_ml_methods(self):
        pass

    def test_dl_methods(self):
        pass


if __name__ == "__main__":
    unittest.main()
