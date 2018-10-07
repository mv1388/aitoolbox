import unittest
from AIToolbox.NLP.DataPrep import core


class TestCore_find_sub_list(unittest.TestCase):
    def test_find_sub_list_do_find(self):
        self.assertEqual(
            core.find_sub_list([1, 2, 3], [1, 1, 1, 5, 1, 2, 3, 5, 6, 7]),
            (4, 6)
        )
        self.assertEqual(
            core.find_sub_list([1, 1], [1, 1, 1, 5, 1, 2, 3, 5, 6, 7]),
            (0, 1)
        )

    def test_find_sub_list_no_find(self):
        self.assertEqual(
            core.find_sub_list([10, 10], [1, 1, 1, 5, 1, 2, 3, 5, 6, 7]),
            None
        )
        self.assertEqual(
            core.find_sub_list([7, 7], [1, 1, 1, 5, 1, 2, 3, 5, 6, 7]),
            None
        )

    def test_find_sub_list_sublist_too_long_exception(self):
        self.assertRaises(
            ValueError,
            core.find_sub_list, [1, 1, 1], [1, 1]
        )
        self.assertRaises(
            ValueError,
            core.find_sub_list, range(10), range(5)
        )


class TestCore_prepare_vocab_mapping(unittest.TestCase):
    def test_prepare_vocab_mapping_no_pad_no_special_label(self):
        vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        word2idx, vocab_size = core.prepare_vocab_mapping(vocab, padding=False, special_labels=[])

        self.assertEqual(
            [word2idx[el] for el in sorted(vocab)],
            list(range(len(vocab)))
        )
        self.assertEqual(
            sorted(vocab), sorted(word2idx.keys())
        )
        self.assertEqual(vocab_size, len(vocab))
        self.assertEqual(len(word2idx), len(vocab))

    def test_prepare_vocab_mapping_no_pad_oov_special_label(self):
        vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        # Test with default OOV label
        word2idx, vocab_size = core.prepare_vocab_mapping(vocab, padding=False)
        self.assertTrue('<OOV>' in word2idx)
        self.assertEqual(
            sorted(vocab + ['<OOV>']),
            sorted(word2idx.keys())
        )
        self.assertEqual(
            word2idx['<OOV>'], 0
        )
        self.assertEqual(len(word2idx), vocab_size)
        self.assertEqual(len(word2idx), len(vocab)+1)

        # Test for non default OOV label
        word2idx_2, vocab_size_2 = core.prepare_vocab_mapping(vocab, padding=False, special_labels=['OOOOOOVVVVV'])
        self.assertEqual(
            sorted(vocab + ['OOOOOOVVVVV']),
            sorted(word2idx_2.keys())
        )
        self.assertEqual(
            word2idx_2['OOOOOOVVVVV'], 0
        )
        self.assertTrue('<OOV>' not in word2idx_2.keys())
        self.assertEqual(len(word2idx_2), vocab_size_2)
        self.assertEqual(len(word2idx_2), len(vocab) + 1)

    def test_prepare_vocab_mapping_with_pad_oov_special_label(self):
        vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        # Test with default OOV label
        word2idx, vocab_size = core.prepare_vocab_mapping(vocab)

        # In the actual idx map only special labels are included. Padding 0 label is not
        self.assertEqual(len(word2idx), len(vocab)+1)
        self.assertEqual(vocab_size, len(word2idx)+1)
        # +2 for padding and special OOV label (default)
        self.assertEqual(vocab_size, len(vocab)+2)

        self.assertTrue('<OOV>' in word2idx.keys())
        self.assertEqual(word2idx['<OOV>'], 1)
        self.assertEqual(
            [word2idx[el] for el in sorted(vocab + ['<OOV>'])],
            list(range(1, len(vocab)+2))
        )
        self.assertEqual(
            [word2idx[el] for el in sorted(vocab)],
            list(range(2, len(vocab)+2))
        )

    def test_prepare_vocab_mapping_with_pad_multiple_special_label(self):
        vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        special_labels = ['<OOV>', 'IIIOOOOVVVVIII', 'START_SENT', 'END_SENT', 'STARTdoc', 'EDDoc', 'AAAA333133213321']

        # Test with default OOV label
        word2idx, vocab_size = core.prepare_vocab_mapping(vocab, padding=True, special_labels=special_labels)

        self.assertEqual(len(word2idx), len(vocab) + len(special_labels))
        self.assertEqual(vocab_size, len(vocab) + len(special_labels) + 1)
        self.assertEqual(
            sorted(vocab + special_labels),
            sorted(word2idx.keys())
        )
        self.assertEqual(
            [word2idx[el] for el in special_labels + sorted(vocab)],
            list(range(1, len(vocab)+len(special_labels)+1))
        )

    def test_prepare_vocab_mapping_full_word_with_pad_multiple_special_label(self):
        vocab = ['Aqua', 'Ball', 'cat', 'driver', 'elastic', 'Fish', 'gap', 'hey', '3342_4,', 'aadas dsa', 'asdd_v']
        special_labels = ['<OOV>', 'IIIOOOOVVVVIII', 'START_SENT', 'END_SENT', 'STARTdoc', 'EDDoc', 'AAAA333133213321']

        # Test with default OOV label
        word2idx, vocab_size = core.prepare_vocab_mapping(vocab, padding=True, special_labels=special_labels)

        self.assertEqual(len(word2idx), len(vocab) + len(special_labels))
        self.assertEqual(vocab_size, len(vocab) + len(special_labels) + 1)
        self.assertEqual(
            sorted(vocab + special_labels),
            sorted(word2idx.keys())
        )
        self.assertEqual(
            [word2idx[el] for el in special_labels + sorted(vocab)],
            list(range(1, len(vocab)+len(special_labels)+1))
        )


class TestCore_vectorize_one_text(unittest.TestCase):
    def test_vectorize_one_text_without_oov(self):
        text_tokens = ['driver', 'cat', 'a', 'Fish', 'Ball', 'Ball']
        vocab = ['Aqua', 'Ball', 'cat', 'driver', 'elastic', 'Fish', 'gap', 'hey', 'a', 'b', 'c']
        word_idx = {el: i for i, el in enumerate(['<OOV>'] + vocab)}

        self.assertEqual(
            core.vectorize_one_text(text_tokens, word_idx),
            [word_idx['driver'], word_idx['cat'], word_idx['a'], word_idx['Fish'], word_idx['Ball'], word_idx['Ball']]
        )

    def test_vectorize_one_text_with_oov(self):
        text_tokens = ['driver', 'cat', 'my out of vocab', 'a', 'Fish', 'Ball', 'Ball', 'aaaawwww']
        vocab = ['Aqua', 'Ball', 'cat', 'driver', 'elastic', 'Fish', 'gap', 'hey', 'a', 'b', 'c']
        word_idx = {el: i for i, el in enumerate(['<OOV>'] + vocab)}

        self.assertEqual(
            core.vectorize_one_text(text_tokens, word_idx),
            [word_idx['driver'], word_idx['cat'], 0, word_idx['a'], word_idx['Fish'], word_idx['Ball'], word_idx['Ball'], 0]
        )
        self.assertEqual(
            core.vectorize_one_text(text_tokens, word_idx),
            [word_idx['driver'], word_idx['cat'], word_idx['<OOV>'], word_idx['a'], word_idx['Fish'], word_idx['Ball'],
             word_idx['Ball'], word_idx['<OOV>']]
        )


if __name__ == '__main__':
    unittest.main()
