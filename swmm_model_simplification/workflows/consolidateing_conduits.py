import networkx as nx

from swmm_api import SwmmInput
from swmm_api.input_file.macros import (
    combine_conduits,
    delete_node,
    subcatchments_connected,
)
from swmm_api.input_file.sections import Conduit

from .aggregating_subcatchments import aggregate_subcatchments


def _dissolve_conduits(inp: SwmmInput, list_conduits: list[Conduit], graph: nx.DiGraph = None):
    """
    Consolidate all conduits into one.

    Combine pairwise conduits starting from the last in the list.

    Args:
        inp (swmm_api.SwmmInput): SWMM input-file data.
        list_conduits (list[Conduit]): list of conduits to consolidate.
        graph (nx.DiGraph): optional - a graph of the model network to accelerate code - otherwise the graph will be created based on the input data.
    """
    for i in range(len(list_conduits) - 1, 0, -1):  # from n to 1
        combine_conduits(
            inp, list_conduits[i - 1], list_conduits[i], graph=graph, reroute_flows_to="to_node"
        )


def consolidate_conduits(
    inp: SwmmInput,
    c_list: list[Conduit],
    length_max: int | float,
    length_min=50,
    graph_with_sc: nx.DiGraph = None,
    logging_func=None,
    kwargs_optimize: dict = None,
    write_sc_transformation_as_tag=False,
):
    """
    Consolidate conduits in a simple path into

    Works inplace.

    Args:
        inp (swmm_api.SwmmInput): SWMM input-file data.
        c_list (list[Conduit]): list of conduits to consolidate.
        length_max (int | float): longest path to combine into one conduit. If the path is longer, make equal long conduits with a maximum length of length_max.
        length_min (int | float): shortest source-path to keep in the model.
        graph_with_sc (nx.DiGraph): optional - a graph of the model network to accelerate code - otherwise the graph will be created based on the input data.
        logging_func (function): function used for logging some messages. Default: no messages logged.
        kwargs_optimize (dict): keyword arguments passed to :func:`~calibrate_subcatchments`.
        write_sc_transformation_as_tag (bool): if the new tag of the SC should be the name of the SC that were aggregated.
    """
    # TODO parallel Links -> with same start end end nodes

    if not c_list:  # list is empty
        return

    # full length of the simple path to be consolidated.
    l_sum = sum(l.length for l in c_list)

    # set prefix to logging function.
    def logging_func_(msg):
        return logging_func(
            f"L={l_sum:0.1f} m -> {msg} (n={len(c_list)}) {[i.name for i in c_list]}"
        )

    if l_sum < length_max:
        # if full length of pass is m´smaller than the upper limit - consolidated into one single conduit
        logging_func_("<combine all conduits>")
        _dissolve_conduits(inp, c_list, graph_with_sc)

    elif (l_sum < length_min) and (graph_with_sc.in_degree[c_list[0].from_node] == 0):
        # if the full length of the path is smaller than the lower limit and there is no sewer upstream of the path -
        #   delete full path.
        for i in range(len(c_list) - 1):
            delete_node(
                inp,
                c_list[i].from_node,
                graph=graph_with_sc,
                alt_node=c_list[i].to_node,
            )
        logging_func_("<remove all conduits>")

    else:
        n_splits = int(l_sum / length_max) + 1
        logging_func_(f"<split conduit in parts> s={n_splits} ")

        n_conduits = len(c_list)
        k, m = divmod(n_conduits, n_splits)
        for start, end in (
            (i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(n_splits)
        ):
            _dissolve_conduits(inp, c_list[start:end], graph_with_sc)
            if end != n_conduits:
                logging_func(f"keep middle node {c_list[end - 1].to_node}")

                # don't aggregate the SC on the last node (yet)
                pre_sc = subcatchments_connected(
                    inp, node=c_list[end - 1].to_node, graph=graph_with_sc
                )
                aggregate_subcatchments(
                    inp,
                    pre_sc,
                    graph_with_sc,
                    logging_func=logging_func,
                    kwargs_optimize=kwargs_optimize,
                    write_sc_transformation_as_tag=write_sc_transformation_as_tag,
                )
