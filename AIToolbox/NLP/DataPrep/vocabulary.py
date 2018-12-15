from nltk.tokenize import word_tokenize

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

SOD_token = 3
EOD_token = 4


class Vocabulary:
    def __init__(self, name, document_level=False):
        """

        Args:
            name (str):
        """
        self.name = name
        self.document_level = document_level
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        if not self.document_level:
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3  # Count SOS, EOS, PAD
        else:
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", SOD_token: "SOD", EOD_token: "EOD"}
            self.num_words = 5

    def add_sentence(self, sentence_tokens):
        """

        Args:
            sentence_tokens (list):

        Returns:

        """
        for word in sentence_tokens:
            self.add_word(word)

    def add_word(self, word):
        """

        Args:
            word (str):

        Returns:

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
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3  # Count SOS, EOS, PAD
        else:
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", SOD_token: "SOD", EOD_token: "EOD"}
            self.num_words = 5

        for word in keep_words:
            self.add_word(word)
