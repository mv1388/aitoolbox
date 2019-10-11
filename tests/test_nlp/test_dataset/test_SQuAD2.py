import unittest
from aitoolbox.nlp.dataset.SQuAD2.SQuAD2DataReader import *


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestDataset_SQuAD2ConcatContextDatasetReader(unittest.TestCase):
    def test_dataset_reader(self):
        reader = SQuAD2ConcatContextDatasetReader(file_path=os.path.join(THIS_DIR, 'SQuAD2-dev-v2.0.json'),
                                                  is_train=True)

        self.assertEqual(len(reader.dataset), 35)
        self.assertEqual(reader.vocab.num_words, 4)
        self.assertEqual(reader.vocab.index2word, {0: "PAD", 1: "OOV", 2: "SOS", 3: "EOS"})
        self.assertEqual(reader.vocab.index2word, {0: "PAD", 1: "OOV", 2: "SOS", 3: "EOS"})

        data, vocab = reader.read()

        self.assertEqual(vocab.num_words, 20096)
        self.assertEqual(len(data), 5928)
        self.assertTrue(all([len(span) == 2 for _, _, span, _ in data]))

        self.assertEqual(max([len(paragraph_tokens) for paragraph_tokens, _, _, _ in data]), 706)
        self.assertEqual(max([len(question_tokens) for _, question_tokens, _, _ in data]), 34)

        self.assertTrue(all([len(paragraph_tokens) > span[1] for paragraph_tokens, _, span, _ in data]))

        self.assertEqual([str(el) for el in data[2][0][:20]],
                         ['The', 'Normans', '(', 'Norman', ':', 'Nourmands', ';', 'French', ':', 'Normands', ';',
                          'Latin', ':', 'Normanni', ')', 'were', 'the', 'people', 'who', 'in'])
        self.assertEqual(data[2][2], (55, 59))
