

class DataAggregator:
    """
    DataAggregator is a wrapper that simulates similar data-flow as in the SQLiteDataAccessor but without text raw data
    saving.

    Similarly as in the case when aggregating with SQLiteDataAccessor, we also need to provide a separate data saver to
    the data_aggregator object. This DataAggregator object will not save data by it self.

    In SQLiteDataAccessor RAW data gets saved, the aggregation part still needs an additional, separate data saver.
    The

    """

    def __init__(self, data_aggregator):
        """
        Examples:
            Example for extracting only aggregate text statistics:

            database_path = '/Volumes/Zunanji Disk/MSc_StackOverflow_dump'
            database_name = 'so-dump.db'
            sql_query = '''SELECT OwnerUserId, Body FROM UserText_ONLY_accept_min2_1Column'''

            # NLP stats aggregator setup
            output_data_path = '/Users/markovidoni/PycharmProjects/UniPostgrad/MSc_project/data/SO_db'
            output_file_name = 'user_text_stats_TEST_TEST.p'
            stats_file_data_saver = FileDataAccessor(data_folder_path=output_data_path, data_file_name=output_file_name,
                                                     data_type='pickle-dict')
            stats_aggregator = TextDataStatsAggregator(stats_file_data_saver)

            # Get the raw HTML data from the database
            data_accessor = SQLiteDataAccessor(db_path=database_path, db_name=database_name)
            # Use just the data aggregator
            data_aggregator = DataAggregator(data_aggregator=stats_aggregator)

            text_prep = TextDataPreparation()
            text_prep.extract_text_body(data_accessor, data_aggregator, sql_query, verbose=False)

        Args:
            data_aggregator: The aggregator should have method compute(data_id, data_list). An example of this is can be
                found in TextAnalysisToolbox: TextDataStatsAggregator class
        """
        self.data_aggregator = data_aggregator

        self.append_current_file_id = None
        self.append_current_file_data_buffer = []

    def __del__(self):
        """
        IMPORTANT!
        Might cause problems when doing HTML aggregation with BeautifulSoup.
        BeautifulSoup object would get destroyed before this __del__ function runs. Rely on explicitly calling
        flush_persist_remaining_data() instead to avoid this problem.

        With other types of aggregators this function worked normally... probably just luck, since object destruction is
        not done in any particular order.
        """
        # Flush and aggregate the last content buffer
        if len(self.append_current_file_data_buffer) > 0:
            self.data_aggregator.compute(self.append_current_file_id, self.append_current_file_data_buffer)

    def ordered_append_save(self, file_id, data, delimiter='\n', verbose=False):
        """
        The function works in the same way as the one in SQLiteDataAccessor with an exception that here always and
        ONLY aggregator is run. In this function there is NO saving of the data (cleaned text) to the database.
        The function only computes the statistics from the data and basically discards the data.

        The signature of the function is again the same as in the other classes to ensure abstraction and compatibility.

        Args:
            file_id:
            data:
            delimiter:
            verbose:

        Returns:

        """
        if file_id != self.append_current_file_id:
            # Check if we aren't at the first user
            if self.append_current_file_id is not None:
                # The idea is to have a text aggregator which produces text statistics for a certain user
                self.data_aggregator.compute(self.append_current_file_id, self.append_current_file_data_buffer)

            if verbose:
                print('=========  New table row added. UserId: ' + str(file_id))

            # Clean data buffer from previous user and prepare to start filling it for the new user
            self.append_current_file_data_buffer = []
            self.append_current_file_id = file_id

        self.append_current_file_data_buffer.append(data)

    def flush_persist_remaining_data(self):
        """
        Explicit flushing of the remaining data and persisting it.
        There was a problem with order in which the objects are destroyed and __del__ code of this object used objects
        that were already destroyed.
        """
        if len(self.append_current_file_data_buffer) > 0:
            self.data_aggregator.compute(self.append_current_file_id, self.append_current_file_data_buffer)
            # Clear the buffer so that problematic __del__ function doesn't execute
            self.append_current_file_data_buffer = []
