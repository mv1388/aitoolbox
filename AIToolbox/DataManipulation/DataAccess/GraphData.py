import networkx as nx

# try:
#     import snap
# except ImportError:
#     print 'SNAP not installed'


class GraphDataAccessor:
    def __init__(self, directed=False, asker_responder_direction=True):
        """

        Args:
            directed (bool): If the constructed graph is directed or undirected
            asker_responder_direction (bool): If set to True and graph is directed, the edge goes from asker user
                to responder user. If set to False, the directed edge goes into opposite direction - from responder
                user to asker user.
        """
        self.asker_responder_direction = asker_responder_direction
        if not directed:
            self.graph = nx.Graph()
        else:
            self.graph = nx.DiGraph()

    def build_graph_from_db(self, db_accessor, query,
                            verbose=False, granularity=100000, max_num_connections=10000000):
        """
        max_num_connections only works if verbose is True


        Examples:
            When using SQLLiteDataAccessor to access data from the database:

            my_query_full_graph = '''select QuestionUserId, AnswerUserId from QAFullGraph'''
            graph_builder = GraphDataAccessor(directed=True)
            graph_directed = graph_builder.build_graph_from_db(data_accessor, my_query_full_graph,
                                                               verbose=True, max_num_connections=-1)


            When using FileDataAccessor to access data from the disk file:

            column_query_idx = [1, 3]
            graph_builder = GraphDataAccessor(directed=True)
            graph_directed = graph_builder.build_graph_from_db(data_accessor, column_query_idx,
                                                               verbose=True, max_num_connections=-1)

        Args:
            db_accessor (SQLiteDataAccessor, FileDataAccessor):
            query (str):
            verbose (bool):
            granularity (int):
            max_num_connections (int):

        Returns: networkx.Graph

        """
        if verbose:
            print('Start building graph')
            for iteration, (asker_id, responder_id) in enumerate(db_accessor.query_db_generator(query)):
                if self.asker_responder_direction:
                    self.graph.add_edge(asker_id, responder_id)
                else:
                    self.graph.add_edge(responder_id, asker_id)

                if iteration % granularity == 0:
                    print(iteration)
                if iteration > max_num_connections > 0:
                    break
            print('Finished building graph')
        else:
            for asker_id, responder_id in db_accessor.query_db_generator(query):
                if self.asker_responder_direction:
                    self.graph.add_edge(asker_id, responder_id)
                else:
                    self.graph.add_edge(responder_id, asker_id)

        return self.graph

    def build_weighted_graph_from_db(self, db_accessor, query,
                                     verbose=False, granularity=100000, max_num_connections=10000000):
        """
        max_num_connections only works if verbose is True

        Args:
            db_accessor:
            query:
            verbose:
            granularity:
            max_num_connections:

        Returns:

        """
        if verbose:
            print('Start building graph')
            for iteration, (asker_id, responder_id, weight) in enumerate(db_accessor.query_db_generator(query)):
                self.graph.add_edge(asker_id, responder_id, weight=weight)
                if iteration % granularity == 0:
                    print(iteration)
                if iteration > max_num_connections > 0:
                    break
            print('Finished building graph')
        else:
            for asker_id, responder_id, weight in db_accessor.query_db_generator(query):
                self.graph.add_edge(asker_id, responder_id, weight=weight)
        return self.graph

    def save_graph(self, graph, db_accessor, save_query):
        """

        Args:
            graph:
            db_accessor:
            save_query:

        Returns:

        """
        raise NotImplementedError


# class SNAPGraphBuilder:
#     """
#     Make sure that SNAP library is installed in the interpreter you are using.
#     Also make sure that SNAP library import is uncommented at the top of this code file and that import doesn't throw
#     an ImportError exception.
#     """
#
#     def __init__(self, directed=False, asker_responder_direction=True):
#         """
#
#         Args:
#             directed (bool):
#             asker_responder_direction (bool):
#         """
#         self.asker_responder_direction = asker_responder_direction
#
#         if not directed:
#             self.graph = snap.TUNGraph.New()
#         else:
#             self.graph = snap.TNGraph.New()
#
#     def build_graph_from_db(self, db_accessor, query_nodes, query_edges,
#                             verbose=False, granularity=100000, max_num_connections=10000000):
#         """
#
#         Args:
#             db_accessor (SQLiteDataAccessor or FileDataAccessor):
#             query_nodes (str):
#             query_edges (str):
#             verbose (bool): not implemented yet
#             granularity (int): not implemented yet, possibly deprecated
#             max_num_connections (int): not implemented yet, possibly deprecated
#
#         Returns:
#
#         """
#         for node_id in db_accessor.query_db_generator(query_nodes):
#             self.graph.AddNode(int(node_id[0]))
#
#         for asker_id, responder_id in db_accessor.query_db_generator(query_edges):
#             if self.asker_responder_direction:
#                 self.graph.AddEdge(int(asker_id), int(responder_id))
#             else:
#                 self.graph.AddEdge(int(responder_id), int(asker_id))
#
#         return self.graph


class PajekDataAccessor:
    """
    Still need to decide what the functionality is going to be... basically some code is needed to prepare the dataset
    that can be used with Pajek.

    Probably need:
        load from SQLite DB and create Pajek input file
        load from file on disk and create Pajek input
    """

    def __init__(self):
        raise NotImplementedError
