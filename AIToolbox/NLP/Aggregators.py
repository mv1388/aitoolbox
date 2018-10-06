from abc import ABCMeta, abstractmethod
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize


class TextDataAggregator(metaclass=ABCMeta):
    @abstractmethod
    def compute(self, data_id, data_list):
        pass


class TextDataStatsAggregator(TextDataAggregator):
    def __init__(self, data_saver, verbose=False):
        """

        Args:
            data_saver (MyDataAnalysis.DataAccess.SQLiteDataAccessor, FileDataAccessor):
            verbose:
        """
        self.data_saver = data_saver
        self.verbose = verbose

    def compute(self, usr_id, text_list):
        """

        Args:
            usr_id:
            text_list (list): List of all the posts made by a certain user

        Returns:

        """
        if self.verbose:
            print('Calculating stats for userID: ' + str(usr_id))

        curr_usr_stats = {}

        curr_usr_stats['num_of_posts'] = len(text_list)
        curr_usr_stats['avg_num_char'] = np.mean([len(t) for t in text_list if len(t) > 0])
        curr_usr_stats['avg_num_words'] = np.mean([len(t.split()) for t in text_list if len(t) > 0])
        curr_usr_stats['avg_num_sentences'] = np.mean([len(sent_tokenize(t)) for t in text_list if len(t) > 0])

        # self.user_text_stats[usr_id] = curr_usr_stats
        self.data_saver.persist_data_to_db(curr_usr_stats, usr_id)


class HTMLDataStatsAggregator(TextDataAggregator):
    """
    IMPORTANT!

    This aggregator needs the execution of flush_persist_remaining_data() function of the aggregator

    """

    def __init__(self, data_saver, verbose=False):
        """

        Args:
            data_saver (MyDataAnalysis.DataAccess.SQLiteDataAccessor, FileDataAccessor):
            verbose:
        """
        self.data_saver = data_saver
        self.verbose = verbose

    def compute(self, usr_id, html_text_list):
        """

        Args:
            usr_id:
            html_text_list:

        Returns:

        """
        if self.verbose:
            print('Calculating stats for userID: ' + str(usr_id))

        curr_usr_stats = {}

        curr_usr_stats['num_code_tags'] = []
        curr_usr_stats['num_code_blocks'] = []

        curr_usr_stats['num_code_block_lines_overall'] = []
        curr_usr_stats['num_code_block_lines_per_doc'] = []

        curr_usr_stats['num_images'] = []

        for html_text in html_text_list:
            soup = BeautifulSoup(html_text)

            curr_usr_stats['num_images'].append(len(soup.find_all('img')))

            curr_usr_stats['num_code_tags'].append(len(soup.find_all('code')))
            curr_usr_stats['num_code_blocks'].append(len(soup.find_all('pre')))

            code_block_list = soup.find_all('pre')
            if len(code_block_list) > 0:
                current_doc_num_code_lines = []
                for code_block in code_block_list:
                    num_lines = str(code_block).count('\n')
                    curr_usr_stats['num_code_block_lines_overall'].append(num_lines)
                    current_doc_num_code_lines.append(num_lines)

                curr_usr_stats['num_code_block_lines_per_doc'].append(np.mean(current_doc_num_code_lines))
            else:
                curr_usr_stats['num_code_block_lines_overall'].append(0.)
                curr_usr_stats['num_code_block_lines_per_doc'].append(0.)

        for i in curr_usr_stats:
            curr_usr_stats[i] = np.mean(curr_usr_stats[i])

        # print usr_id
        # print curr_usr_stats

        self.data_saver.persist_data_to_db(curr_usr_stats, usr_id)
