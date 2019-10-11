import unittest
import pandas as pd
import process_data

class TestProcessData(unittest.TestCase):
    def setUp(self):

        pass

    def test_clean_data(self):

        # Test dataset
        df = pd.DataFrame({
            'message': ['stuff', 'other stuff'],
            'original': ['le stuff', 'autre stuff'],
            'genre': ['a genre', 'a genre'],
            'categories': ['related-1;request-1;offer-0', 'related-0;request-0;offer-1']
        })
        # Put it through the clean_data function
        result = process_data.clean_data(df)

        # Check that the number of columns is correct
        self.assertEqual(result.shape[1], 6)


if __name__ == '__main__':
    unittest.main()