import mysql.connector

from AIToolbox.DataManipulation.DataAccess.SQLite import *


class MySQLDataAccessor(SQLiteDataAccessor):
    def __init__(self, host, database, user, password,
                 **kwargs):
        """

        Args:
            host (str):
            database (str):
            user (str):
            password (str):
            **kwargs:
        """
        SQLiteDataAccessor.__init__(self, **kwargs)

        self.db_conn = mysql.connector.connect(user=user, password=password, host=host, database=database)
        self.cursor = None

    def __del__(self):
        if self.cursor is not None:
            self.cursor.close()

        self.db_conn.close()

    def query_db_generator(self, sql_query):
        """

        Args:
            sql_query (str):

        Returns:

        """
        self.cursor = self.db_conn.cursor()
        self.cursor.execute(sql_query)
        for result_row in self.cursor:
            yield result_row
