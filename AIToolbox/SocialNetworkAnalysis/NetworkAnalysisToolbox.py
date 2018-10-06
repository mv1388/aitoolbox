from multiprocessing import Pool
import itertools
import numpy

from AIToolbox.SocialNetworkAnalysis.ParallelExecution import *


"""     
        TODO:
        
        Rewrite code from ParallelExecution
            This should be in the Parallel module... now just so that it works there is a separate file for 
            ParallelExecution 
        
"""


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def _betmap(G_normalized_weight_sources_tuple):
    """Pool for multiprocess only accepts functions with one argument.
    This function uses a tuple as its only argument. We use a named tuple for
    python 3 compatibility, and then unpack it when we send it to
    `betweenness_centrality_source`
    """
    # print 'NEWNEW'
    return nx.betweenness_centrality_source(*G_normalized_weight_sources_tuple)


def betweenness_centrality_parallel(G, num_cpu_cores=None):
    """
    Parallel betweenness centrality  function

    Args:
        G (NetworkX.graph):
        num_cpu_cores:

    Returns:

    """
    p = Pool(processes=num_cpu_cores)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), int(G.order() / node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.map(_betmap,
                  zip([G] * num_chunks,
                      [True] * num_chunks,
                      [None] * num_chunks,
                      node_chunks))

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c


def shortest_path_length_between_nodes_parallel(graph, number_of_cores=3, num_free_cores=-1):
    """

    Args:
        graph (NetworkX.Graph):
        number_of_cores (int):
        num_free_cores (int):

    Returns:
        dict: Dict of dicts, where first dict represents node A and second dict represents node B of the A-B connection.

    """
    tp = TaskParallelizator(number_of_cores=number_of_cores, num_free_cores=num_free_cores)
    return tp.shortest_path_length_between_nodes_parallel(graph)


def closeness_centrality_parallel(graph, number_of_cores=3, num_free_cores=-1):
    """

    Args:
        graph:
        number_of_cores:
        num_free_cores:

    Returns:

    """
    tp = TaskParallelizator(number_of_cores=number_of_cores, num_free_cores=num_free_cores)
    return tp.closeness_centrality_parallel(graph)


"""

    The following code structural holes is from:
    https://github.com/isnotinvain/nodens/blob/master/structural_holes.py

"""


def _calcProportionaTieStrengthWRT(A, i):
    """
    Calculates P from Burt's equation
    using only the intersection of each node's ego network and that of node i

    A is the adjacencey matrix
    """
    num = A.copy()
    num = num + num.transpose()

    P = A.copy()
    mask = numpy.where(A, 1, 0)

    mask = mask[i]
    mask[0, i] = 1.0
    P = numpy.multiply(mask, P)
    P = numpy.multiply(mask.transpose(), P)
    P = P + P.transpose()

    denom = P.sum(1)
    denom = numpy.repeat(denom, len(P), axis=1)

    mask = numpy.where(denom, 1, float('nan'))
    denom = numpy.multiply(denom, mask)

    return numpy.nan_to_num(numpy.divide(num, denom))


def _calcProportionalTieStrengths(A):
    """
    Calculates P from Burt's equation,
    using each node's entire ego network

    A is the adjacencey matrix
    """
    num = A.copy()
    num = num + num.transpose()

    denom = num.sum(1)
    denom = numpy.repeat(denom, len(num), axis=1)
    mask = numpy.where(denom, 1, float('nan'))
    denom = numpy.multiply(denom, mask)
    return numpy.nan_to_num(numpy.divide(num, denom))


def _neighborsIndexes(graph, node, includeOutLinks, includeInLinks):
    """
    returns the neighbors of node in graph
    as a list of their INDEXes within graph.node()
    """
    neighbors = set()

    if includeOutLinks:
        neighbors |= set(graph.neighbors(node))

    if includeInLinks:
        neighbors |= set(graph.predecessors(node))

    return map(lambda x: graph.nodes().index(x), neighbors)


def structural_holes(graph, includeOutLinks=True, includeInLinks=False, wholeNetwork=True):
    """
    Calculate each node's contraint / structural holes value, as described by Ronal Burt

    Parameters
    ----------
    G : graph
        a networkx Graph or DiGraph

    includeInLinks : whether each ego network should include nodes which point to the ego - this should be False for undirected graphs

    includeOutLinks : whether each ego network should include nodes which the ego points to - this should be True for undirected graphs

    wholeNetwork : whether to use the whole ego network for each node, or only the overlap between the current ego's network and the other's ego network

    Returns
    -------
    constraints : dictionary
                  dictionary with nodes as keys and dictionaries as values in the form {"C-Index": v,"C-Size": v,"C-Density": v,"C-Hierarchy": v}
                  where v is each value

    References
    ----------
    .. [1] Burt, R.S. (2004). Structural holes and good ideas. American Journal of Sociology 110, 349-399
    """

    if not hasattr(graph, "predecessors"):
        print("graph is undirected... setting includeOutLinks to True and includeInLinks to False")
        includeOutLinks = True
        includeInLinks = False

    # get the adjacency matrix view of the graph
    # which is a numpy matrix
    A = nx.to_numpy_matrix(graph)

    # calculate P_i_j from Burt's equation
    p = _calcProportionalTieStrengths(A)

    # this is the return value
    constraints = {}

    for node in graph.nodes():
        # each element of constraints will be a dictionary of this form
        # unless the node in question is an isolate in which case it
        # will be None
        constraint = {"C-Index": 0.0, "C-Size": 0.0, "C-Density": 0.0, "C-Hierarchy": 0.0}

        # Vi is the set of i's neighbors
        Vi = _neighborsIndexes(graph, node, includeOutLinks, includeInLinks)
        if len(Vi) == 0:
            # isolates have no defined constraint
            constraints[node] = None
            continue

        # i is the node we are calculating constraint for
        # and is thus the ego of the ego net
        i = graph.nodes().index(node)

        if not wholeNetwork:
            # need to recalculate p w/r/t this node
            pq = _calcProportionaTieStrengthWRT(A, i)
        else:
            # don't need to calculate p w/r/t any node
            pq = p

        for j in Vi:
            Pij = p[i, j]
            constraint["C-Size"] += Pij ** 2
            innerSum = 0.0
            for q in Vi:
                if q == j or q == i: continue
                innerSum += p[i, q] * pq[q, j]

            constraint["C-Hierarchy"] += innerSum ** 2
            constraint["C-Density"] += 2 * Pij * innerSum

        constraint["C-Index"] = constraint["C-Size"] + constraint["C-Density"] + constraint["C-Hierarchy"]
        constraints[node] = constraint
    return constraints


"""

    The following code brokerage is from:
    https://github.com/isnotinvain/nodens/blob/master/brokerage.py

"""


class _RoleClassifier(object):
    roleTypes = {
        "coordinator": lambda pred, broker, succ: pred == broker == succ,
        "gatekeeper" 	 : lambda pred , broker ,succ: pred != broker == succ,"representative"	: lambda pred , broker , succ: pred == broker != succ,
        "consultant"		:lambda pred , broker, succ: pred == succ != broker,
        "liaison"		: lambda pred , broker ,succ: pred != succ and pred != broker and broker != succ,
    }

    @classmethod
    def classify( cls , predecessor_group , broker_group ,successor_group):
        for role ,predicate in cls.roleTypes.iteritems():
            if predicate( predecessor_group , broker_group ,successor_group):
                return role
        raise Exception("Could not classify... this should never happen")


def getBrokerageRoles (graph ,partition):
    """
    Counts how many times each node in graph acts as one of the five brokerage roles described by Steven Borgatti in
    http://www.analytictech.com/essex/Lectures/Brokerage.pdf

    graph: a networx DiGraph
    partition: a dictionary mapping node -> group, must map every node. If a node has no group associate then put it by itself in a new group

    returns: {node -> {"cooridnator": n, "gatekeeper": n, "representative": n, "consultant": n, "liaison": n}} where n is the number of times
    node acted as that role
    """

    roleClassifier = _RoleClassifier()

    roles = dict((node,dict ((role ,0) for role in roleClassifier.roleTypes)) for node in graph)
    for node in graph:
        for successor in graph.successors(node):
            for predecessor in graph.predecessors(node):
                if successor == predecessor or successor == node or predecessor == node: continue
                if not (graph.has_edge(predecessor, successor)):
                    # found a broker!
                    # now which kind depends on who is in which group
                    roles[node][roleClassifier.classify(partition[predecessor],
                                                        partition[node], partition[successor])] += 1
    return roles
