from itertools import permutations

import networkx as nx

from swmm_api import SwmmInput
from swmm_api.input_file import SEC
from swmm_api.input_file.macros import (
    inp_to_graph,
    nodes_dict,
    junction_to_outfall,
    storage_to_outfall,
    create_sub_inp,
    subcatchments_connected,
    delete_subcatchment, delete_link,
)
from swmm_api.input_file.sections import (
    Outfall,
    Weir,
    CrossSection,
    Coordinate,
    TimeseriesData,
    Inflow, DryWeatherFlow,
)


def node_to_outfall(
        inp: SwmmInput, node, outfall_kind=Outfall.TYPES.NORMAL, graph=None
):
    """
    Convert any node in the model to a system outfall node.

    Used to cut off downstream part of a model.

    Adds a weir downstream if specified node has multiple incoming links and created an outfall below the weir.

    Works inplace.

    Args:
        inp (swmm_api.SwmmInput): SWMM input-file data.
        node (str): label of the node.
        outfall_kind (str): outfall kind. See :class:`~swmm_api.input_file.sections.node.Outfall.TYPES`.
        graph (nx.DiGraph): optional - a graph of the model network to accelerate code - otherwise the graph will be created based on the input data.
    """
    if graph is None:
        graph = inp_to_graph(inp)

    if (graph.in_degree[node] > 1) or (
            (graph.in_degree[node] == 1)
            and isinstance(graph.edges[next(graph.predecessors(node)), node]["label"], list)
    ):
        # ..., if there are multiple incoming links upstream.
        # Create a weir with a huge cross-section.
        # Create outfall after weir.
        inp.add_multiple(
            Outfall(f"DUMMY_OUT_{node}", nodes_dict(inp)[node].elevation, outfall_kind),
            Weir(
                f"DUMMY_{node}",
                from_node=node,
                to_node=f"DUMMY_OUT_{node}",
                form=Weir.FORMS.TRANSVERSE,
                height_crest=0,
                discharge_coefficient=1,
            ),
            CrossSection(
                f"DUMMY_{node}", CrossSection.SHAPES.RECT_OPEN, height=5, parameter_2=10
            ),
        )
        if inp.COORDINATES:
            inp.add_obj(
                Coordinate(
                    f"DUMMY_OUT_{node}",
                    inp.COORDINATES[node].x + 2,
                    inp.COORDINATES[node].y,
                )
            )
    elif inp.JUNCTIONS and node in inp.JUNCTIONS:
        junction_to_outfall(inp, node, kind=outfall_kind)
    elif inp.OUTFALLS and node in inp.OUTFALLS:
        ...  # everything is OK
    elif inp.STORAGE and node in inp.STORAGE:
        storage_to_outfall(inp, node, kind=outfall_kind)


def add_pre_calculated_timeseries(
        inp_cut_out, dict_pre_calculated_ts, logging_func=None
):
    """
    Setting a previously calculated timeseries of flow as direct inflow for boundary condition.
    at nodes where model was cut off before.

    Used for HD and Agg model.

    DWF, INFLOWS and SUBCATCHMENTS directly connected to the boundary nodes will be removed.

    Args:
        inp_cut_out (swmm_api.SwmmInput): SWMM input-file data of already cut-out model.
        dict_pre_calculated_ts (dict[str, pandas.Series]): for each node (that was previously calculated) a series of the total_inflow into the node.
        logging_func (function): function used for logging some messages. Default: no messages logged.

    Returns:
        swmm_api.SwmmInput: model with cut-off upstream part.
    """
    # nodes in the cut-out model, that have a previously calculated time-series.
    current_ts_calculated_in_model = set(nodes_dict(inp_cut_out)) & set(dict_pre_calculated_ts)

    # go through every pre-calculated node in the cut-out model
    for n in current_ts_calculated_in_model:
        if (SEC.OUTFALLS in inp_cut_out) and (n in inp_cut_out.OUTFALLS):
            # if node is an outfall - skip (set no direct inflow) - should not happen
            # for example CSOs.
            continue

        # remove dwf and inflow for these nodes
        index = (n, DryWeatherFlow.TYPES.FLOW)
        if inp_cut_out.DWF and index in inp_cut_out.DWF:
            del inp_cut_out.DWF[index]
        if inp_cut_out.INFLOWS and index in inp_cut_out.INFLOWS:
            del inp_cut_out.INFLOWS[index]

        # remove connected SC
        for sc in subcatchments_connected(inp_cut_out, n):
            delete_subcatchment(inp_cut_out, sc.name)

        # add timeseries as inflow to node
        inp_cut_out.add_multiple(
            TimeseriesData.from_pandas(dict_pre_calculated_ts[n], f"inflow_{n}"),
            Inflow(n, time_series=f"inflow_{n}"),
        )

    logging_func(
        f"pre-calculated timeseries in sub-model (n={len(current_ts_calculated_in_model)}) {current_ts_calculated_in_model}"
    )

    return inp_cut_out


def _get_upstream_nodes(downstream_node, graph, dict_pre_calculated_ts_):
    """
    Get nodes upstream of the current node of interest.

    Stop search at nodes that are pre-calculated


    Args:
        downstream_node (str): node of interest.
        graph (nx.DiGraph): a graph of the current state of the model network.
        dict_pre_calculated_ts_ (dict[str, pandas.Series]): for each node (that was previously calculated) a series of the total_inflow into the node.

    Returns:
        (list[str], list[str]): nodes that should be included in the cut-out model and  nodes as outfall of the cut-out model
    """
    nodes_list = []  # nodes that should be included in the cut-out model
    outfall_list = [downstream_node]  # nodes as outfall of the cut-out model

    # incoming (upstream) nodes of outfall node
    _queue = list(graph.predecessors(downstream_node))

    while _queue:
        # next node in queue
        node = _queue.pop()
        # if (node != downstream_node) and (node in outfall_list):
        if node in outfall_list:
            # if node is not the node of interest and in the outfall-list - remove the node from that list.
            # logging_func_(f"{node} x outfall_list")
            # print(f"{node} x outfall_list")
            outfall_list.remove(node)

        # logging_func_(f"{node} > nodes_list")
        # print(f"{node} > nodes_list")
        nodes_list.append(node)

        # add successors to outfall_list
        # nodes that are parallel of the downstream_node
        for _node_downstream in graph.successors(node):
            # except if the node is already in node_list or is downstream_node
            if (_node_downstream not in nodes_list) and (
                    _node_downstream not in outfall_list
            ):
                # logging_func_(f"{_node_downstream} > outfall_list")
                # print(f"{_node_downstream} > outfall_list")
                outfall_list.append(_node_downstream)

        # add all upstream nodes of the current node to the queue of nodes that should be included in the cut-out model.
        if node in dict_pre_calculated_ts_:
            # print(f"{node} is already in dict_pre_calculated_ts_")
            ...
        else:
            _queue += list(graph.predecessors(node))

    return outfall_list, nodes_list


def cut_network(
        inp, downstream_node, dict_pre_calculated_ts=None, logging_func=None, graph=None
) -> SwmmInput:
    """
    Cut the model based on the downstream node and upstream nodes as boundary nodes.

    Args:
        inp (swmm_api.SwmmInput): SWMM input-file data.
        downstream_node (str): label of the downstream node as downstream boundary of the model.
        dict_pre_calculated_ts (dict[str, pandas.Series]): for each node (that was previously calculated) a series of the total_inflow into the node.
        logging_func (function): function used for logging some messages. Default: no messages logged.
        graph (nx.DiGraph): optional - a graph of the model network (in the same state of the `inp` data) to accelerate code - otherwise the graph will be created based on the input data.

    Returns:
        swmm_api.SwmmInput: Cut-out SWMM input-file data.
    """
    # ---
    if graph is None:
        graph_uncut = inp_to_graph(inp)
    else:
        graph_uncut = graph.copy()

    # remove downstream node from the pre-calculated nodes
    dict_pre_calculated_ts_ = dict_pre_calculated_ts.copy()
    if downstream_node in dict_pre_calculated_ts_:
        del dict_pre_calculated_ts_[downstream_node]

    # ---
    # add label for logging function
    def logging_func_(msg):
        return logging_func("cut_network | " + msg)

    logging_func_(f"nodes in graph = {len(graph_uncut)}")

    # ---
    logging_func_(f"{downstream_node} > outfall_list")
    outfall_list, nodes_list = _get_upstream_nodes(downstream_node, graph_uncut, dict_pre_calculated_ts_)

    # ---
    # create new cut-out mode based on determined nodes
    logging_func_("create inp")
    inp_cut_out = create_sub_inp(inp.copy(), nodes_list + outfall_list)
    logging_func_("create new network")
    graph_cut_out = inp_to_graph(inp_cut_out)

    logging_func_(f"nodes in new graph = {graph_cut_out.number_of_nodes()}")

    # delete links between outfall_list nodes
    for possible_link in permutations(outfall_list, 2):
        if possible_link in graph_cut_out.edges:
            # break
            illicit_link_label = graph_cut_out.edges[possible_link]['label']
            delete_link(inp_cut_out, illicit_link_label)
            graph_cut_out.remove_edge(*possible_link)

    nodes_cut_out = nodes_dict(inp_cut_out)
    # ---
    # set most downstream nodes as outfall
    for n in outfall_list:
        _suc = [i in nodes_cut_out for i in graph_uncut.successors(n)]
        if graph_cut_out.out_degree[n] == 0:
            node_to_outfall(
                inp_cut_out, n, outfall_kind=Outfall.TYPES.NORMAL, graph=graph_cut_out
            )
        elif all(_suc):
            # for looped networks
            logging_func_(
                f'node "{n}" not converted to outfall -> all downstream node are in the model'
            )
        elif any(_suc):
            # should not happen ...
            # node will not be converted
            logging_func_(
                f'ERROR: downstream nodes of node "{n}" are in the network, but not all'
            )
    # ---
    # nodes in the cut-out model, that have a previously calculated time-series.
    dict_pre_calculated_ts_cut_out = {k: v for k, v in dict_pre_calculated_ts_.items() if k in nodes_cut_out}

    # reroute every link, that is connected upstream of a node with a pre-calculated time-series, to a dummy outfall.
    # for every node with a pre-calculated time-series
    for node_pre_calc in dict_pre_calculated_ts_cut_out:
        # does the node have incoming links, if so iterate through them
        for i, node_incoming in enumerate(graph_cut_out.predecessors(node_pre_calc)):
            # create a dummy outfall for every incoming link
            outfall = Outfall(f'dummy_{node_pre_calc}_out{i}',
                              nodes_cut_out[node_pre_calc].elevation,
                              Outfall.TYPES.NORMAL)
            inp_cut_out.add_obj(outfall)

            # reroute every incoming link to dummy outfall
            conduit = graph_cut_out.edges[(node_incoming, node_pre_calc)]['obj']
            conduit.to_node = outfall.name

        # ---
    add_pre_calculated_timeseries(
        inp_cut_out, dict_pre_calculated_ts_cut_out, logging_func=logging_func_
    )

    return inp_cut_out
