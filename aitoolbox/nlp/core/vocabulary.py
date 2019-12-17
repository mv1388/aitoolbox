
class Vocabulary:
    def __init__(self, name, document_level=False):
        """Vocabulary used for storing the tokens and converting between the indices and the tokens

        Args:
            name (str): name of the vocabulary / type of vocabulary. Needed just for tracking purposes
            document_level (bool): If the vocabulary is on the sentence level or on the document level. Document
                consists of multiple sentences. This in effect means that we are adding additional tokens for
                start and the end of the doc.
        """
        # Default word tokens
        self.PAD_token = 0  # Used for padding short sentences
        self.OOV_token = 1  # Out of vocab token
        self.SOS_token = 2  # Start-of-sentence token
        self.EOS_token = 3  # End-of-sentence token

        self.SOD_token = 4
        self.EOD_token = 5

        self.name = name
        self.document_level = document_level
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        if not self.document_level:
            self.index2word = {self.PAD_token: "PAD", self.OOV_token: "OOV", self.SOS_token: "SOS", self.EOS_token: "EOS"}
            self.default_tokens = [self.PAD_token, self.OOV_token, self.SOS_token, self.EOS_token]
        else:
            self.index2word = {self.PAD_token: "PAD", self.OOV_token: "OOV", self.SOS_token: "SOS",
                               self.EOS_token: "EOS", self.SOD_token: "SOD", self.EOD_token: "EOD"}
            self.default_tokens = [self.PAD_token, self.OOV_token, self.SOS_token, self.EOS_token,
                                   self.SOD_token, self.EOD_token]

        self.num_words = len(self.default_tokens)

    def add_sentence(self, sentence_tokens):
        """Add tokenized sentence to the vocabulary

        Args:
            sentence_tokens (list): sentence tokens, e.g. list of words representing the sentence

        Returns:
            None
        """
        for word in sentence_tokens:
            self.add_word(str(word))

    def add_word(self, word):
        """Add the single word to the vocabulary

        Args:
            word (str): single word string

        Returns:
            None
        """
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        """Remove words below a certain count threshold

        Args:
            min_count (int):

        Returns:
            None
        """
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        if not self.document_level:
            self.index2word = {self.PAD_token: "PAD", self.OOV_token: "OOV", self.SOS_token: "SOS", self.EOS_token: "EOS"}
            self.num_words = 4  # Count SOS, EOS, PAD
        else:
            self.index2word = {self.PAD_token: "PAD", self.OOV_token: "OOV", self.SOS_token: "SOS", self.EOS_token: "EOS",
                               self.SOD_token: "SOD", self.EOD_token: "EOD"}
            self.num_words = 6

        for word in keep_words:
            self.add_word(word)

    def convert_sent2idx_sent(self, sent_tokens, start_end_token=True):
        """Convert the given tokenized string sentence into the indices

        Args:
            sent_tokens (list):
            start_end_token (bool): string tokens forming a sentence

        Returns:
            list: sentence tokens converted into the corresponding indices
        """
        if len(self.word2index) == 0:
            raise ValueError('word2index is empty')

        return ([self.SOS_token] if start_end_token else []) + \
               [self.word2index[word] if word in self.word2index else self.OOV_token for word in sent_tokens] + \
               ([self.EOS_token] if start_end_token else [])

    def convert_idx_sent2sent(self, idx_sent, rm_default_tokens=False):
        """Convert from token indices back to the human-readable string tokens

        Args:
            idx_sent: index tokens forming the sentence
            rm_default_tokens (bool): should the default tokens such as padding and start/end sentence tokens be
                removed from the result.

        Returns:
            list: sentence represented as a sequence of the string tokens
        """
        if len(self.index2word) == 0:
            raise ValueError('word2index is empty')

        if rm_default_tokens:
            return [self.index2word[idx_word] for idx_word in idx_sent if idx_word not in self.default_tokens]
        else:
            return [self.index2word[idx_word] for idx_word in idx_sent]
