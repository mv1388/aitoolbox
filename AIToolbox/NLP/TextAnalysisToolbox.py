from abc import ABCMeta, abstractmethod

from bs4 import BeautifulSoup
import numpy as np
from nltk.tokenize import sent_tokenize
import re
import gensim


class TextDataPreparation:
    def __init__(self, html_data_aggregator=None, only_html_aggregation=False):
        """
        Examples:
            How to use HTML data aggregator (Here using DataAggregator, so no text cleaning and saving):
                database_path = '/Volumes/Zunanji Disk/MSc_StackOverflow_dump/NO_LOOPS'
                database_name = 'so-dump.db'
                sql_query = '''SELECT OwnerUserId, Body FROM UserText_ONLY_accept_min2_1Column_NOL'''

                # NLP stats aggregator setup
                output_data_path = '/Users/markovidoni/PycharmProjects/UniPostgrad/MSc_project/data/SO_db/W_NO_LOOPS/NO_CODE'
                output_file_name = 'user_HTML_stats_TEST.p'
                stats_file_data_saver = FileDataAccessor(data_folder_path=output_data_path,
                                                         data_file_name=output_file_name,
                                                         data_type='pickle-dict')
                html_stats_aggregator = HTMLDataStatsAggregator(stats_file_data_saver)

                # Get the raw HTML data from the database
                data_accessor = SQLiteDataAccessor(db_path=database_path, db_name=database_name)
                # Use just the data aggregator
                data_aggregator = DataAggregator(data_aggregator=html_stats_aggregator)

                text_prep = TextDataPreparation(html_data_aggregator=data_aggregator, only_html_aggregation=True)
                text_prep.extract_text_body(data_accessor=None, data_saver=None, sql_query=sql_query, verbose=False)


        Args:
            html_data_aggregator (DataAccess.DataAggregator): An additional DataAggregator which works at the initial
                part of the data pipeline. It processes the raw HTML text data. The other DataAggregator which can also
                be used works on the lower level, right before the cleaned text data is saved to database. That
                aggregator is found in the DataAccess accessor objects. So in short, there can be 2 separate
                DataAggregators: one for html data present at TextDataPreparation (this) object level and the other
                present in SQLiteDataAccessor object working on cleaned text data.
            only_html_aggregation (bool): set to True to perform only the HTML stats aggregation with
                the html_data_aggregator.

        """
        self.html_data_aggregator = html_data_aggregator
        self.only_html_aggregation = only_html_aggregation

    def extract_text_body(self, data_accessor, data_saver, sql_query, keep_inline_code=True, rm_math_latex_mode=None,
                          verbose=False):
        """
        Examples:
            For saving to files in the folder on a disk:
                database_path = '/Users/markovidoni/PycharmProjects/UniPostgrad/MSc_project/data/STATS_db'
                database_name = 'stats-dump.db'
                sql_query = '''SELECT OwnerUserId, Body
                FROM Posts
                WHERE OwnerUserId is NOT NULL AND OwnerUserId > -1
                ORDER BY OwnerUserId;'''
                output_database_name = 'stats-text-extract.db'
                db_create_query = 'CREATE TABLE IF NOT EXISTS [Text_Extract](OwnerUserId INTEGER, Body TEXT)'
                db_insert_query = 'INSERT INTO Text_Extract (OwnerUserId, Body) VALUES ({values})'

                data_accessor = SQLiteDataAccessor(db_path=database_path, db_name=database_name)
                db_data_saver = SQLiteDataAccessor(db_path=database_path, db_name=output_database_name,
                                                   db_create_query=db_create_query, db_insert_query=db_insert_query)
                text_prep = TextDataPreparation()
                text_prep.extract_text_body(data_accessor, db_data_saver, sql_query, verbose=False)


            For saving to the database:

                database_path = '/Volumes/Zunanji Disk/MSc_StackOverflow_dump'
                database_name = 'so-dump.db'
                sql_query = '''SELECT OwnerUserId, Body FROM UserText_accept_min2'''
                output_database_name = 'so-text-extract.db'
                db_create_query = 'CREATE TABLE IF NOT EXISTS [Text_Extract](OwnerUserId INTEGER, Body TEXT)'
                db_insert_query = 'INSERT INTO Text_Extract (OwnerUserId, Body) VALUES ({values})'

                data_accessor = SQLiteDataAccessor(db_path=database_path, db_name=database_name)
                db_data_saver = SQLiteDataAccessor(db_path=database_path, db_name=output_database_name,
                                                   db_create_query=db_create_query, db_insert_query=db_insert_query)
                text_prep = TextDataPreparation()
                text_prep.extract_text_body(data_accessor, db_data_saver, sql_query, verbose=False)


        Args:
            data_accessor:
            data_saver:
            sql_query:
            keep_inline_code:
            rm_math_latex_mode:
            verbose:

        Returns:

        """
        for usr_id, text_body in data_accessor.query_db_generator(sql_query):
            if self.html_data_aggregator is not None:
                self.html_data_aggregator.ordered_append_save(usr_id, text_body,
                                                              verbose=(verbose is True or verbose == 3))

            if not self.only_html_aggregation:
                if keep_inline_code:
                    clean_text = self.process_text_rm_code_box(text_body)
                else:
                    clean_text = self.process_text_rm_inline_code_code_box(text_body)

                if rm_math_latex_mode is True:
                    clean_text = self.process_text_rm_math_latex(clean_text)
                elif rm_math_latex_mode == 'placeholder':
                    clean_text = self.process_text_rm_math_latex(clean_text, replacement='_MATH_EQUATION_')

                # text_body should be clean when inserting into file
                if len(clean_text) > 0:
                    data_saver.ordered_append_save(usr_id, clean_text, delimiter='\n\n',
                                                   verbose=(verbose is True or verbose == 1))

            if verbose is True or verbose == 2:
                print('===========================================================\n')
                print(text_body)
                # print clean_text

        # if isinstance(data_saver, SQLiteDataAccessor):
        #     data_saver.db_conn.commit()
        #     data_saver.close_connection()
        #
        # data_accessor.close_connection()

        """
        Needed to add this because there was a problem with ordering of object destruction.
        __del__ function would be called after BeautifulSoup was already destroyed

        Explicitly calling remaining data flushing for last user from buffer to the disk solved this problem.

        TODO: Maybe think of a better and more automatic solution. 
        """
        if self.html_data_aggregator is not None:
            self.html_data_aggregator.flush_persist_remaining_data()

    @staticmethod
    def process_text_rm_code_box(html_text):
        """

        Args:
            html_text:

        Returns: str

        """
        soup = BeautifulSoup(html_text)
        for pre in soup.find_all('pre'):
            pre.decompose()
        clean_text = soup.text
        clean_text = clean_text.replace('\n', ' ').rstrip()
        return clean_text

    @staticmethod
    def process_text_rm_inline_code_code_box(html_text):
        """

        Args:
            html_text:

        Returns: str

        """
        soup = BeautifulSoup(html_text)
        for code in soup.find_all('code'):
            code.decompose()
        for pre in soup.find_all('pre'):
            pre.decompose()
        clean_text = soup.text
        clean_text = clean_text.replace('\n', ' ').rstrip()
        return clean_text

    @staticmethod
    def process_text_rm_math_latex(text, replacement=''):
        """

        Args:
            text (str): Probably should be cleaned text already
            replacement (str):

        Returns: str

        """
        text = re.sub(r'\$\$.+?\$\$', replacement, text)
        clean_text = re.sub(r'\$.+?\$', replacement, text)
        # Make sure that you again remove \n which might appear as I remove the math formulas
        clean_text = clean_text.replace('\n', ' ').rstrip()
        return clean_text

    def extract_comments(self, data_accessor, data_saver, sql_query, rm_math_latex_mode=None, verbose=False):
        """

        Args:
            data_accessor:
            data_saver:
            sql_query:
            rm_math_latex_mode:
            verbose:

        Returns:

        """
        for usr_id, comment_text in data_accessor.query_db_generator(sql_query):
            if rm_math_latex_mode is True:
                clean_text = self.process_text_rm_math_latex(clean_text)
            elif rm_math_latex_mode == 'placeholder':
                clean_text = self.process_text_rm_math_latex(clean_text, replacement='_MATH_EQUATION_')

            # text_body should be clean when inserting into file
            if len(comment_text) > 0:
                data_saver.ordered_append_save(usr_id, comment_text, delimiter='\n\n',
                                               verbose=(verbose is True or verbose == 1))


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


class Doc2vecExtractor:
    def __init__(self, data_accessor, sql_query):
        """
        Expects the data source (db) with cleaned text. Also expects the data to be sorted by the user ID on the
        database side.

        Args:
            data_accessor:
            sql_query:
        """
        self.data_accessor = data_accessor
        self.sql_query = sql_query

        self.data_element_id_order = []
        self.train_corpus = None
        self.model = None

    def construct_train_corpus(self, one_el_per_usr=True, all_user_posts_combined=True, tokens_only=False,
                               verbose=False):
        """

        Args:
            one_el_per_usr:
            all_user_posts_combined:
            tokens_only:
            verbose:

        Returns:

        """
        if verbose:
            print('Starting the train corpus construction.')

        if one_el_per_usr:
            self.train_corpus = list(self.read_corpus_1_el_per_usr(tokens_only=tokens_only))
        else:
            self.train_corpus = list(self.read_corpus_1_el_per_post(all_user_posts_combined=all_user_posts_combined,
                                                                    tokens_only=tokens_only))

        return self

    def read_corpus_1_el_per_usr(self, tokens_only=False):
        """

        Args:
            tokens_only:

        Returns:

        """
        for user_id, clean_text in self.data_accessor.query_db_generator(self.sql_query):
            self.data_element_id_order.append(user_id)

            text = clean_text.replace('\n\n', ' ')

            if tokens_only:
                yield gensim.utils.simple_preprocess(text)
            else:
                # For training data, add tags
                """
                Tags may be one or more unicode string tokens, but typical practice 
                (which will also be most memory-efficient) is for the tags list to include a unique 
                integer id as the only tag.
                """
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(text),
                                                           [int(user_id)])

    def read_corpus_1_el_per_post(self, all_user_posts_combined=True, tokens_only=False):
        """

        Args:
            all_user_posts_combined:
            tokens_only:

        Returns:

        """
        user_post_id_ctr = 0
        curr_usr_id = None

        for user_id, clean_text in self.data_accessor.query_db_generator(self.sql_query):
            if all_user_posts_combined:
                clean_text_post_list = clean_text.split('\n\n')
                for post_ctr, clean_text_post in enumerate(clean_text_post_list):
                    self.data_element_id_order.append(str(user_id) + '_' + str(post_ctr))

                    if tokens_only:
                        yield gensim.utils.simple_preprocess(clean_text_post)
                    else:
                        # For training data, add tags
                        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(clean_text_post),
                                                                   [str(user_id) + '_' + str(post_ctr)])
            else:
                if user_id != curr_usr_id:
                    curr_usr_id = user_id
                    user_post_id_ctr = 0

                self.data_element_id_order.append(str(user_id) + '_' + str(user_post_id_ctr))

                if tokens_only:
                    yield gensim.utils.simple_preprocess(clean_text)
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(clean_text),
                                                               [str(user_id) + '_' + str(user_post_id_ctr)])

                user_post_id_ctr += 1

    def create_train_model(self, size=200, min_count=2, iter=55, verbose=False, **kwargs):
        """

        Args:
            size:
            min_count:
            iter:
            verbose:
            **kwargs:

        Returns: the resulting doc vectors in np array

        """
        if self.train_corpus is None:
            raise ValueError('Train corpus has not been constructed yet')
        if verbose:
            print('Gensim start')

        self.model = gensim.models.doc2vec.Doc2Vec(size=size, min_count=min_count, iter=iter, **kwargs)
        self.model.build_vocab(self.train_corpus)
        self.model.train(self.train_corpus)

        return np.array(self.model.docvecs)
