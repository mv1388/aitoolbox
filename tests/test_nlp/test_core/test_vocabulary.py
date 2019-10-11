import unittest
from aitoolbox.nlp.core.vocabulary import *


class TestVocabulary(unittest.TestCase):
    def test_hard_coded_values(self):
        vocab_doc = Vocabulary('testVocabulary', document_level=False)
        self.assertEqual(vocab_doc.PAD_token, 0)
        self.assertEqual(vocab_doc.OOV_token, 1)
        self.assertEqual(vocab_doc.SOS_token, 2)
        self.assertEqual(vocab_doc.EOS_token, 3)

        vocab_doc = Vocabulary('testVocabulary', document_level=True)
        self.assertEqual(vocab_doc.PAD_token, 0)
        self.assertEqual(vocab_doc.OOV_token, 1)
        self.assertEqual(vocab_doc.SOS_token, 2)
        self.assertEqual(vocab_doc.EOS_token, 3)
        self.assertEqual(vocab_doc.SOD_token, 4)
        self.assertEqual(vocab_doc.EOD_token, 5)

    def test_vocabulary_init(self):
        vocab = Vocabulary('testVocabulary', document_level=False)

        self.assertEqual(vocab.name, 'testVocabulary')
        self.assertEqual(vocab.document_level, False)
        self.assertEqual(vocab.num_words, 4)

        self.assertEqual(vocab.index2word, {0: "PAD", 1: "OOV", 2: "SOS", 3: "EOS"})
        self.assertEqual(vocab.index2word, {vocab.PAD_token: "PAD", vocab.OOV_token: "OOV",
                                            vocab.SOS_token: "SOS", vocab.EOS_token: "EOS"})

    def test_vocabulary_init_doc_level(self):
        vocab = Vocabulary('testVocabulary', document_level=True)

        self.assertEqual(vocab.name, 'testVocabulary')
        self.assertEqual(vocab.document_level, True)
        self.assertEqual(vocab.num_words, 6)

        self.assertEqual(vocab.index2word, {0: "PAD", 1: "OOV", 2: "SOS", 3: "EOS", 4: "SOD", 5: "EOD"})
        self.assertEqual(vocab.index2word, {vocab.PAD_token: "PAD", vocab.OOV_token: "OOV",
                                            vocab.SOS_token: "SOS", vocab.EOS_token: "EOS",
                                            vocab.SOD_token: "SOD", vocab.EOD_token: "EOD"})

    def test_add_sentence(self):
        vocab = Vocabulary('testVocabulary', document_level=False)
        initial_vocab_size = vocab.num_words
        sent = ['today', 'the', 'sun', 'shines', '.']
        vocab.add_sentence(sent)

        self.assertEqual(vocab.word2index, {'today': 4, 'the': 5, 'sun': 6, 'shines': 7, '.': 8})
        self.assertEqual(vocab.index2word,
                         {0: 'PAD', 1: 'OOV', 2: 'SOS', 3: 'EOS', 4: 'today', 5: 'the', 6: 'sun', 7: 'shines', 8: '.'})

        self.assertEqual(vocab.num_words, 9)
        self.assertEqual(vocab.num_words, initial_vocab_size + len(sent))

    def test_add_multiple_sentences(self):
        vocab = Vocabulary('testVocabulary', document_level=False)
        initial_vocab_size = vocab.num_words
        sent_1 = ['today', 'the', 'sun', 'shines', '.']
        sent_2 = ['but', 'tomorrow', 'it', 'will', 'be', 'raining', '!']
        vocab.add_sentence(sent_1)
        vocab.add_sentence(sent_2)

        self.assertEqual(vocab.word2index,
                         {'today': 4, 'the': 5, 'sun': 6, 'shines': 7, '.': 8, 'but': 9, 'tomorrow': 10, 'it': 11,
                          'will': 12, 'be': 13, 'raining': 14, '!': 15})
        self.assertEqual(vocab.index2word,
                         {0: 'PAD', 1: 'OOV', 2: 'SOS', 3: 'EOS', 4: 'today', 5: 'the', 6: 'sun', 7: 'shines', 8: '.',
                          9: 'but', 10: 'tomorrow', 11: 'it', 12: 'will', 13: 'be', 14: 'raining', 15: '!'}
                         )

        self.assertEqual(vocab.num_words, 16)
        self.assertEqual(vocab.num_words, initial_vocab_size + len(sent_1) + len(sent_2))

    def test_add_multiple_sentences_repeat_words(self):
        vocab = Vocabulary('testVocabulary', document_level=False)
        initial_vocab_size = vocab.num_words
        sent_1 = ['today', 'the', 'sun', 'shines', '.']
        sent_2 = ['but', 'tomorrow', 'the', 'sun', 'will', 'be', 'gone', 'and', 'it', 'will', 'rain', '.']
        vocab.add_sentence(sent_1)
        vocab.add_sentence(sent_2)

        self.assertEqual(vocab.word2index,
                         {'today': 4, 'the': 5, 'sun': 6, 'shines': 7, '.': 8, 'but': 9, 'tomorrow': 10, 'will': 11,
                          'be': 12, 'gone': 13, 'and': 14, 'it': 15, 'rain': 16}
                         )
        self.assertEqual(vocab.index2word,
                         {0: 'PAD', 1: 'OOV', 2: 'SOS', 3: 'EOS', 4: 'today', 5: 'the', 6: 'sun', 7: 'shines', 8: '.',
                          9: 'but', 10: 'tomorrow', 11: 'will', 12: 'be', 13: 'gone', 14: 'and', 15: 'it', 16: 'rain'}
                         )

        self.assertEqual(vocab.num_words, 17)
        self.assertEqual(vocab.num_words, initial_vocab_size + len(set(sent_1 + sent_2)))

    def test_convert_sent2idx_sent(self):
        vocab = Vocabulary('testVocabulary', document_level=False)
        sent_1 = ['today', 'the', 'sun', 'shines', '.']
        sent_2 = ['but', 'tomorrow', 'the', 'sun', 'will', 'be', 'gone', 'and', 'it', 'will', 'rain', '.']
        vocab.add_sentence(sent_1)
        vocab.add_sentence(sent_2)

        self.assertEqual(
            vocab.convert_sent2idx_sent(sent_1),
            [2, 4, 5, 6, 7, 8, 3]
        )
        self.assertEqual(
            vocab.convert_sent2idx_sent(sent_2),
            [2, 9, 10, 5, 6, 11, 12, 13, 14, 15, 11, 16, 8, 3]
        )

        new_sent = ['today', 'actually', 'the', 'sun', 'is', 'not', 'shine', 'but', 'tomorrow', 'it', 'will', 'rain', '.']
        self.assertEqual(
            vocab.convert_sent2idx_sent(new_sent),
            [2, 4, 1, 5, 6, 1, 1, 1, 9, 10, 15, 11, 16, 8, 3]
        )

        self.assertEqual(len(vocab.convert_sent2idx_sent(sent_1)), len(sent_1) + 2)
        self.assertEqual(len(vocab.convert_sent2idx_sent(sent_2)), len(sent_2) + 2)
        self.assertEqual(len(vocab.convert_sent2idx_sent(new_sent)), len(new_sent) + 2)

    def test_convert_idx_sent2sent(self):
        vocab = Vocabulary('testVocabulary', document_level=False)
        sent_1 = ['today', 'the', 'sun', 'shines', '.']
        sent_2 = ['but', 'tomorrow', 'the', 'sun', 'will', 'be', 'gone', 'and', 'it', 'will', 'rain', '.']
        vocab.add_sentence(sent_1)
        vocab.add_sentence(sent_2)

        self.assertEqual(
            vocab.convert_idx_sent2sent([2, 4, 5, 6, 7, 8, 3]),
            ['SOS'] + sent_1 + ['EOS']
        )
        self.assertEqual(
            vocab.convert_idx_sent2sent([2, 9, 10, 5, 6, 11, 12, 13, 14, 15, 11, 16, 8, 3]),
            ['SOS'] + sent_2 + ['EOS']
        )

        new_sent_back = ['today', 'OOV', 'the', 'sun', 'OOV', 'OOV', 'OOV', 'but', 'tomorrow', 'it', 'will', 'rain', '.']
        self.assertEqual(
            vocab.convert_idx_sent2sent([2, 4, 1, 5, 6, 1, 1, 1, 9, 10, 15, 11, 16, 8, 3]),
            ['SOS'] + new_sent_back + ['EOS']
        )
