from itertools import permutations

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
    inp, dict_pre_calculated_ts, graph_full, logging_func=None
):
    """
    Cut off model part upstream of nodes with of previously calculated timeseries of flow and set the timeseries as direct inflow for boundary condition.

    DWF, INFLOWS and SUBCATCHMENTS directly connected to the boundary nodes will be removed.

    Args:
        inp (swmm_api.SwmmInput): SWMM input-file data of already cut-out model.
        dict_pre_calculated_ts (dict[str, pandas.Series]): for each node (that was previously calculated) a series of the total_inflow into the node.
        graph_full (nx.DiGraph): a graph of the full (uncut) model network.
        logging_func (function): function used for logging some messages. Default: no messages logged.

    Returns:
        swmm_api.SwmmInput: model with cut-off upstream part.
    """
    graph = inp_to_graph(inp)
    final_nodes = nodes_dict(inp)
    current_ts_calculated_in_model = set(final_nodes) & set(dict_pre_calculated_ts)

    # go through every pre-calculated node in the cut-out model
    for n in current_ts_calculated_in_model:
        if (SEC.OUTFALLS in inp) and (n in inp.OUTFALLS):
            # if node is an outfall - skip (set no direct inflow) - should not happen
            continue

        # check if the nodes upstream (in the already simplified model) are also pre-calculated
        _previous_nodes_are_calculated = [
            i in current_ts_calculated_in_model for i in graph.predecessors(n)
        ]  # type: list[bool]

        # If all incoming nodes are pre-calculated and also in the cut-out model - skip (set no direct inflow)
        if all(_previous_nodes_are_calculated) and all(
            i in final_nodes for i in graph_full.predecessors(n)
        ):
            continue

        # Error: if only some node upstream are pre-calculated...
        elif any(_previous_nodes_are_calculated):
            # TODO why is this happening - can maybe happen in some models (?)
            logging_func(
                f'ERROR: double counted pre-calculated timeseries in node "{n}"'
            )

        # ---
        # remove dwf and inflow for these nodes
        index = (n, DryWeatherFlow.TYPES.FLOW)
        if inp.DWF and index in inp.DWF:
            del inp.DWF[index]
        if inp.INFLOWS and index in inp.INFLOWS:
            del inp.INFLOWS[index]

        # remove connected SC
        for sc in subcatchments_connected(inp, n):
            delete_subcatchment(inp, sc.name)

        # add timeseries as inflow to node
        if isinstance(dict_pre_calculated_ts, dict):
            inp.add_multiple(
                TimeseriesData.from_pandas(dict_pre_calculated_ts[n], f"inflow_{n}"),
                Inflow(n, time_series=f"inflow_{n}"),
            )
        else:
            inp.add_multiple(
                TimeseriesData(f"inflow_{n}_dummy", [("01/01/1990 00:00:00", 0)]),
                Inflow(n, time_series=f"inflow_{n}_dummy"),
            )

    logging_func(
        f"pre-calculated timeseries in sub-model (n={len(current_ts_calculated_in_model)}) {current_ts_calculated_in_model}"
    )

    return inp


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
        graph = inp_to_graph(inp)
    else:
        graph = graph.copy()

    # remove downstream node from the pre-calculated nodes
    dict_pre_calculated_ts_ = dict_pre_calculated_ts.copy()
    if downstream_node in dict_pre_calculated_ts_:
        del dict_pre_calculated_ts_[downstream_node]

    # ---
    # add label for logging function
    def logging_func_(msg): return logging_func("cut_network | " + msg)
    logging_func_(f"nodes in graph = {len(graph)}")

    # ---
    nodes_list = []  # nodes should be included in the cut-out model
    outfall_list = [downstream_node]  # nodes as outfall of the cut-out model
    logging_func_(f"{downstream_node} > outfall_list")

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

    # ---
    # create new cut-out mode based on determined nodes
    logging_func_("create inp")
    inp_part = create_sub_inp(inp.copy(), nodes_list + outfall_list)
    logging_func_("create new network")
    graph_part = inp_to_graph(inp_part)

    # delete links between outfall_list nodes
    for possible_link in permutations(outfall_list, 2):
        if possible_link in graph_part.edges:
            # break
            illicit_link_label = graph_part.edges[possible_link]['label']
            delete_link(inp_part, illicit_link_label)
            graph_part.remove_edge(*possible_link)

    nodes = nodes_dict(inp_part)
    # ---
    # set most downstream nodes as outfall
    for n in outfall_list:
        _suc = [i in nodes for i in graph.successors(n)]
        if graph_part.out_degree[n] == 0:
            node_to_outfall(
                inp_part, n, outfall_kind=Outfall.TYPES.NORMAL, graph=graph_part
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

    logging_func_(f"nodes in new graph = {graph_part.number_of_nodes()}")
    # ---
    add_pre_calculated_timeseries(
        inp_part, dict_pre_calculated_ts_, graph, logging_func=logging_func_
    )

    return inp_part
