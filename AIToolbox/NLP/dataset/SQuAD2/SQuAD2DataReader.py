import json
from collections import Counter
from tqdm import tqdm

from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers.reading_comprehension import util

from AIToolbox.NLP.core.vocabulary import Vocabulary


"""

    Implementation relying on the allennlp library for the data processing instead of AIToolbox's NLP core package functions 
    
    Faster processing, however, now non alphanum remove, etc.
    
"""


class SQuAD2ConcatContextDatasetReader:
    def __init__(self, file_path, tokenizer=None, is_train=True):
        self.file_path = file_path
        self.is_train = is_train
        self.dataset = None

        self._tokenizer = tokenizer or WordTokenizer()

        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            self.dataset = dataset_json['data'][:3]

        self.vocab = Vocabulary('SQuAD2', document_level=False)

    def read(self):
        data = []

        for article in tqdm(self.dataset, total=len(self.dataset)):
            for paragraph_json in article['paragraphs']:
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)

                if self.is_train:
                    self.vocab.add_sentence(tokenized_paragraph)

                for question_answer in paragraph_json['qas']:
                    if question_answer['is_impossible']:
                        continue

                    question_text = question_answer["question"].strip().replace("\n", "")
                    tokenized_question = self._tokenizer.tokenize(question_text)

                    if self.is_train:
                        self.vocab.add_sentence(tokenized_question)

                    answer_texts = [answer['text'] for answer in question_answer['answers']]

                    span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                    span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]

                    example = self.process_example(tokenized_paragraph, tokenized_question,
                                                   zip(span_starts, span_ends), answer_texts)
                    # yield example
                    data.append(example)

        return data, self.vocab

    def process_example(self, paragraph_tokens, question_tokens, char_spans, answer_texts):
        token_spans = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in paragraph_tokens]

        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start, char_span_end))
            # if error:
            #     print('Error')

            token_spans.append((span_start, span_end))

        candidate_answers = Counter()
        for span_start, span_end in token_spans:
            candidate_answers[(span_start, span_end)] += 1
        span_start, span_end = candidate_answers.most_common(1)[0][0]

        return paragraph_tokens, question_tokens, (span_start, span_end) #, answer_texts
