import unittest
from common.input_tw import tongyong_phonetic_dict, tongyong_to_bytes
from common.input_tw import tongyong_index_dict
from common.input_tw import index_phonetic_dict
from common.input_tw import bytes_to_phonetic

class TestTongyongMethods(unittest.TestCase):

    def test_tongyong_to_phonetic(self):
        phonetic = tongyong_phonetic_dict['lin1']
        self.assertEqual(phonetic, 'ㄌㄧㄣˊ')

    def test_tongyong_to_index(self):
        index = tongyong_index_dict['lin1']
        self.assertEqual(index, 593)
        #self.assertTrue('FOO'.isupper())

    def test_index_to_phonetic(self):
        phonetic = index_phonetic_dict[593]
        self.assertEqual(phonetic, 'ㄌㄧㄣˊ')
        
    def test_tongyong_to_bytes(self):
        bytes = tongyong_to_bytes('lin1sing0')
        self.assertEqual(bytes, [593, 357])
        
    def test_indexies_to_phonetic(self):
        phonetic_list = bytes_to_phonetic([593, 357])
        self.assertEqual(phonetic_list, ['ㄌㄧㄣˊ', 'ㄒㄧㄥ'])

if __name__ == '__main__':
    unittest.main()