from joblib import Parallel, delayed
import multiprocessing

# Find a best parallelisation library option

import networkx as nx

"""     
        TODO:

        In the end remove this file... parallel stuff should go to the Parallel module and network stuff to SocialNet
        module

"""



def calculate_path_len(u_id, node_id_list, graph):
    """

    Args:
        u_id (int): user node id from which the distances are calculated
        node_id_list (list): list of user node ids
        graph (NetworkX.Graph): NetworkX Graph object

    Returns:

    """
    top_usr_dists = {}
    for u_id2 in node_id_list:
        if u_id != u_id2:
            try:
                top_usr_dists[u_id2] = nx.shortest_path_length(graph, u_id, u_id2)
            except nx.exception.NetworkXNoPath:
                # Set distance to -1 if there is no path between 2 nodes
                top_usr_dists[u_id2] = -1

    return u_id, top_usr_dists



class TaskParallelizator:
    """
    Parallelise a given function and data across the multiple CPUs

    TODO: Implement...
    """

    def __init__(self, number_of_cores=3, num_free_cores=-1):
        """

        Args:
            number_of_cores (int):
            num_free_cores (int):
        """
        self.number_of_cores = number_of_cores
        if num_free_cores > 0:
            self.number_of_cores = multiprocessing.cpu_count() - num_free_cores

    def closeness_centrality_parallel(self, graph, distance=None, normalized=True):
        """

        Args:
            graph (nx.Graph):
            distance:
            normalized (bool):

        Returns:

        """
        results = Parallel(n_jobs=self.number_of_cores)(
            delayed(nx.closeness_centrality)(graph, node, distance, normalized) for node in list(graph))
        return results

    def shortest_path_length_between_nodes_parallel(self, graph):
        """

        Args:
            graph:

        Returns:
            dict:

        """
        node_list = list(set(graph.nodes()))
        results = Parallel(n_jobs=self.number_of_cores)(
            delayed(calculate_path_len)(node, node_list, graph) for node in list(node_list))

        results_dict = {}
        for u_id, top_usr_dists in results:
            results_dict[u_id] = top_usr_dists

        return results_dict


# # *args testing
# # Delete when development is finished
# def my_test(arr, multi):
#     return [el * multi for el in arr]
# tp = TaskParallelizator()
# print tp.graph_function_execute_parallel(sum, a=[1,2,3])
# print tp.graph_function_execute_parallel(my_test, a=[1,2,3], b=3)
