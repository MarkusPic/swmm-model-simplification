import datetime
import json
import re
import sys

from tqdm.auto import tqdm

from swmm_api import SwmmInput
from swmm_api.input_file import SEC
from swmm_api.input_file.macros import (
    delete_node,
    inp_to_graph,
    links_connected,
    subcatchments_connected,
    next_links,
)
from swmm_api.input_file.sections import (
    Conduit,
    TitleSection,
)
from .workflows import aggregate_subcatchments, consolidate_conduits

from .helpers.cumulative_catchment_area import get_cumulative_catchment_area
from .helpers.search_algorithm import iter_depth_first_search
from .helpers.swmm_input_summary import (
    get_link_full_flow,
    get_link_cross_section_area,
)


def aggregate_model(
        inp_hd,
        area_min,
        length_max,
        skip_optimisation=False,
        optimize_volume=True,
        optimize_flow_full=True,
        optimize_flow_full_ratio=False,
        write_sc_transformation_as_tag=False,
        logging_func=None,
):
    """
    Simplifies SWMM model using the preset rainfall. Auto-calibrated based on flow rate.

    Tested on small rainfall event with no flooding.

    Preparation:
        - set rain for raingauges TIMESERIES, RAINGAUGES, SUBCATCHMENTS
        - set times for rain in OPTIONS
        - set report start for start time of calibration time range for the NSE in OPTIONS
        - set simulation options - routing step, ...

    Args:
        inp_hd (SwmmInput): SWMM input data.
        area_min (int | float): minimum aggregated area for one SC which should stay in the model at the boundary of the network.
        length_max (int | float): longest path to combine into one conduit. If the path is longer, make equal long conduits with a maximum length of length_max.
        skip_optimisation (bool): if `True` -> SC width and conduits are not calibrated on the HD model after aggregation and cosolidation. default `False`.
        write_sc_transformation_as_tag (bool): if the new tag of the SC should be the name of the SC that were aggregated.
        logging_func (optional | function): function used for logging some messages. Default: no messages logged.
        optimize_volume (bool): when true - change the cross-section height to fit the volume of the consolidated conduit to the volume of the original path of conduit in the HD model.
        optimize_flow_full (bool): when false - use the full flow rate of the first (most upstream) conduit of the HD path that was consolidated. when true - use the smallest (most limiting) full flow rate of the HD path.
        optimize_flow_full_ratio (bool): when false - use the smallest full flow rate of the HD path. when true - use the smallest ratio of full flow rate and the connected catchment area to limit flow capacity.
    """
    inp = inp_hd.copy()

    graph_sewer = inp_to_graph(inp_hd, add_subcatchments=False)  # HD network
    graph_with_sc = inp_to_graph(inp, add_subcatchments=True)  # updated -> aggregated

    time_start = datetime.datetime.now()  # for documentation

    # -------
    counter = {}

    tag_regex = re.compile(r"\<([^\>]+)\>")

    if logging_func is None:
        def logging_func(*_): ...

    def logging_func_agg(msg):
        tags = tag_regex.findall(msg)
        if tags:
            tag = tags[0]
            if tag not in counter:
                counter[tag] = 1
            else:
                counter[tag] += 1
        # ---
        return logging_func(f'aggregate_model | {msg}')

    # -------
    logging_func("get_cross_section_summary_hd | start")
    di_link_full_flow_hd = get_link_full_flow(inp_hd)
    di_link_cross_section_area_hd = get_link_cross_section_area(inp_hd)
    di_link_cumulative_catchment_area_hd = get_cumulative_catchment_area(inp_hd, full_flow=di_link_full_flow_hd, logging_func=logging_func)
    logging_func("get_cross_section_summary_hd | stop")

    # -------
    nodes_done = set()

    if skip_optimisation:
        kwargs_optimize = None
    else:
        # key-word arguments for function :func:`swmm_model_simplification.workflows.aggregating_subcatchments.calibrate_subcatchments`
        kwargs_optimize = dict(
            inp_agg=inp,
            inp_hd=inp_hd,
            graph_hd=graph_sewer,
            logging_func=lambda msg: logging_func(f'autocalibrate model | {msg}'),
            current_ts_calculated_hd={},
            current_ts_calculated_agg={},
            # following arguments are only for calibrate_conduits_on_hd
            link_full_flow_hd=di_link_full_flow_hd,
            link_cross_section_area_hd=di_link_cross_section_area_hd,
            link_cumulative_catchment_area_hd=di_link_cumulative_catchment_area_hd,
            optimize_volume=optimize_volume,
            optimize_flow_full=optimize_flow_full,
            optimize_flow_full_ratio=optimize_flow_full_ratio,
        )

    # conduits in a simple path that can be consolidated (combined to one)
    combine_conduits_list = []

    def log_sub(msg):
        return logging_func_agg(" " * 8 + " | " + msg)

    # Iterate through the network from upstream to downstream.
    for node_current in tqdm(
            iter_depth_first_search(graph_sewer, logging_func=logging_func_agg),
            total=len(graph_sewer.nodes),
            desc="simplify model - iter over all nodes",
    ):
        if combine_conduits_list:
            # if we have conduits that can be consolidated,
            #   but we changed the path, and now we consolidate the conduits
            #   before we start we the new path.

            last_node = combine_conduits_list[-1].to_node

            if last_node != node_current:
                # OR: node_current not in set(graph_with_sc.successors(last_node))

                # we changed the simple path that can be consolidated
                # we consolidate that last path

                log_sub("<combine_conduits_list - new path>")
                consolidate_conduits(
                    inp,
                    combine_conduits_list,
                    length_max,
                    graph_with_sc=graph_with_sc,
                    logging_func=log_sub,
                    kwargs_optimize=kwargs_optimize,
                    write_sc_transformation_as_tag=write_sc_transformation_as_tag,
                )
                combine_conduits_list = []

                # if all nodes before the last node in the consolidated path were processed
                #   meaning no unprocessed parallel path exists
                #   then we aggregate the SCs to that last node and calibrate the consolidated conduit and the SC
                if all(
                        n in nodes_done
                        for n in graph_with_sc.predecessors(last_node)
                        if n not in inp.SUBCATCHMENTS
                ):
                    log_sub("combine_conduits_list - dissolve_subcatchments")
                    sc_connected_list = subcatchments_connected(
                        inp, node=last_node, graph=graph_with_sc
                    )
                    aggregate_subcatchments(
                        inp,
                        sc_connected_list,
                        graph_with_sc,
                        logging_func=log_sub,
                        kwargs_optimize=kwargs_optimize,
                        write_sc_transformation_as_tag=write_sc_transformation_as_tag,
                    )

                logging_func_agg("_" * 50)

        # --- ---
        def log_node(msg):
            return logging_func_agg(f"{node_current:8s} | {msg}")

        # get a list of connected upstream (incoming) nodes and SC to the current node
        # based on the current state of the simplified model
        pre_nodes = []
        sc_connected_list = []
        for k in graph_with_sc.predecessors(node_current):
            if k in inp.SUBCATCHMENTS:
                sc_connected_list.append(inp.SUBCATCHMENTS[k])
            else:
                pre_nodes.append(k)

        # number of paths connected to the node upstream
        n_inflows = len(pre_nodes)  # graph_with_sc.in_degree[node_current]

        # if the conduits should be consolidated
        #   will be True when we are in a simple path of conduits
        skip_combine_conduits = False

        # -----------------
        if node_current not in inp.JUNCTIONS:  # or fixed/problem node
            # if the node is not a junction but some special structure
            log_node("<keep node - not a junction>")
        # -----------------
        elif n_inflows == 0:
            # start node => no inflow links (current simplification state)
            area = sum(
                [sc.area for sc in sc_connected_list]
            )  # sum of all connected SC area to node

            if graph_with_sc.out_degree[node_current] > 1:
                # start node that is a branching node - multiple outflows
                log_node(
                    f"<keep node - start node with branching downstream> {skip_combine_conduits=}"
                )

            elif area < area_min:
                # node is a start node + Junction + one or none outgoing links.
                # if the directly connected area is small (negligible) or no SC is connected (separate wastewater sewer)
                # node will be deleted and all inflows (direct, dry weather, SC) will be connected to the node downstream

                if any(
                        [
                            not isinstance(l, Conduit)
                            for l in next_links(inp, node_current, g=graph_sewer)
                        ]
                ):
                    # if any of the next (downstream) links is not a conduit, then the node will be not deleted.
                    log_node(
                        f"<keep node - next link not a conduit> {skip_combine_conduits=}"
                    )

                # elif next_sc_has_different_soil(sc_connected_list, graph_with_sc, node_current, inp):
                #     log_node('<keep node - new soil type>')

                else:
                    # start node and one single outflow link that is a conduit and small connected area.
                    delete_node(
                        inp,
                        node_current,
                        graph=graph_with_sc,
                        alt_node=next(graph_with_sc.successors(node_current)),
                    )
                    log_node("<delete_node>")
                    nodes_done.add(node_current)
                    continue  # go to next node in network

            elif len(sc_connected_list) > 1:
                # start node + connected area big enough + only one or none outgoing links.
                # and more than one SC is connected
                # continue to aggregate and calibrate SCs.
                log_node("<keep node - area big enough>")

            else:
                # start node + connected area big enough + only one or none outgoing links + only one or node SC connected.
                log_node("<no SC connected>")

        # -----------------
        elif n_inflows == 1:
            # node is a junction + exactly one inflow link

            if graph_with_sc.out_degree[node_current] == 1:
                # node is a junction + exactly one inflow link + exactly one outflow (outgoing) link|node
                links_before, links_after = links_connected(
                    inp, node_current, g=graph_with_sc
                )

                # based on the previous conditions, there are only one incoming and outgoing links
                # so we select there in the list (only type conversion from list of links to link)
                link0 = links_before[0]
                link1 = links_after[0]

                if isinstance(link0, Conduit) and isinstance(link1, Conduit):
                    # if both links are conduits -> condition to get consolidated
                    # check if links are already in list (use list for the right order)
                    # MAYBE: if conduits_are_equal(inp, link0, link1, diff_height=0.15, diff_slope=0.3):
                    if link0 not in combine_conduits_list:
                        combine_conduits_list.append(link0)
                        log_node(f'<add to combine_conduits> "{link0.name}"')
                    if link1 not in combine_conduits_list:
                        combine_conduits_list.append(link1)
                        log_node(f'<add to combine_conduits> "{link1.name}"')
                    # log_node(f'<add to combine_conduits> ... {[c.name for c in combine_conduits_list]}')
                    skip_combine_conduits = True  # don't combine conduits yet (wait for end of simple path)

                else:
                    # if one of both links was not a conduit
                    log_node("<keep node - not a conduit>")

            elif graph_with_sc.out_degree[node_current] > 1:
                # node is a junction + exactly one inflow link + more than one outflow (outgoing) links|nodes
                # branching node
                log_node(f"<keep node - branching downstream> {skip_combine_conduits=}")

            else:
                # node is a junction + exactly one inflow link + no outflow (outgoing) links|nodes
                # last node in the network
                log_node("<last node>")

        # -----------------
        elif n_inflows > 1:
            # node is a junction + more than one inflow link
            # merging node (multiple inflow links)
            log_node("<keep node - merging node - branch upstream>")

        else:
            # node is a junction + neither none, one, nor more than one inflow.
            log_node("<Error> unreachable point")

        # -----------------
        if not skip_combine_conduits:
            # while adding conduits to combine_conduits_list -> skip this part
            # skipping is true, while on simple path of conduits

            # consolidate that last path

            log_sub("<combine_conduits_list inline>")
            consolidate_conduits(
                inp,
                combine_conduits_list,
                length_max,
                graph_with_sc=graph_with_sc,
                logging_func=log_sub,
                kwargs_optimize=kwargs_optimize,
                write_sc_transformation_as_tag=write_sc_transformation_as_tag,
            )
            combine_conduits_list = []

            # aggregate the SCs to that last node and calibrate the consolidated conduit and the SC
            sc_connected_list = subcatchments_connected(
                inp, node=node_current, graph=graph_with_sc
            )
            aggregate_subcatchments(
                inp,
                sc_connected_list,
                graph_with_sc,
                logging_func=log_sub,
                kwargs_optimize=kwargs_optimize,
                write_sc_transformation_as_tag=write_sc_transformation_as_tag,
            )

        nodes_done.add(node_current)

    logging_func_agg("\n" + "_" * 50 + "\n" + str(json.dumps(counter, indent=4)))

    time_end = datetime.datetime.now()

    # add meta info in TITLE
    inp[SEC.TITLE] = TitleSection(
        inp[SEC.TITLE].to_inp_lines() + f"\nAggregated  model\n"
                                        f"Area for simplification: {area_min}\n"
                                        f"Length for simplification: {length_max}\n"
                                        f"Auto-calibration: {'skipped' if skip_optimisation else ('conduit volume + ' if optimize_volume else '') + ('full flow rate' if optimize_flow_full else '') + ('full flow rate ratio' if optimize_flow_full_ratio else '')}\n"
                                        f"Operating system: {sys.platform}\n"
                                        f"Timestamp: {time_end}\n"
                                        f"Simplification duration: {time_end - time_start}"
    )

    return inp
