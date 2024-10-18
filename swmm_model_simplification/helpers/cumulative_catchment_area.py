from swmm_api import SwmmInput
from swmm_api.input_file.macros import links_dict
from swmm_api.input_file.macros.graph import _previous_links_labels, inp_to_graph, _next_links_labels

from .search_algorithm import iter_depth_first_search
from .swmm_input_summary import get_link_full_flow


def get_cumulative_catchment_area(inp: SwmmInput, full_flow=None, logging_func=None):
    """
    Calculate the cumulative catchment area for a given input SWMM model.

    Args:
        inp (swmm_api.SwmmInput): SWMM input-file data.
        full_flow (dict[str, float]): series with key = label of the link and value is the full flow capacity.
        logging_func (function): function used for logging some messages. Default: no messages logged.

    Returns:
        dict[str, float]: key = label of the link and value = cumulative catchment area.
    """
    if full_flow is None:
        # Calculate full if not given. Runs SWMM and get the info from the report file section 'Cross Section Summary'.
        full_flow = get_link_full_flow(inp)

    # The original sewer network as graph
    graph_sewer = inp_to_graph(inp, add_subcatchments=False)  # HD network
    # The current state of the network as graph inclusive subcatchments
    graph_with_sc = inp_to_graph(inp, add_subcatchments=True)  # updated -> simplified

    # dict for the results
    res_links = {}

    # dict of all link in the model
    di_links = links_dict(inp)

    # process every node in the network from upstream to downstream
    for node_current in iter_depth_first_search(graph_sewer):

        # sum the catchment area of all connected links upstream
        # 0 if no link upstream = most ubstream node
        area_links_upstream = sum(
            res_links[l] for l in _previous_links_labels(graph_sewer, node_current)
        )

        # calculate the sum of catchment area of all directly connected subcatchments to the current node.
        area_sc_connected = sum(
            inp.SUBCATCHMENTS[k].area * inp.SUBCATCHMENTS[k].imperviousness / 100
            for k in graph_with_sc.predecessors(node_current)
            if k in inp.SUBCATCHMENTS
        )

        area_upstream = float(area_sc_connected + area_links_upstream)

        # list of links downstream of the current node (directly connected to the node)
        links_downstream = [l for l in _next_links_labels(graph_sewer, node_current)]

        if len(links_downstream) == 1:
            # if the link is not a branching node - set the connected area to the link downstream
            res_links[links_downstream[0]] = area_upstream

        elif all(l in full_flow for l in links_downstream):
            # if the node is a branching node and we have a full flow value for all downstream links.
            # split the catchment area based on the full flow ratio of the links.
            # (acc. to DWA: 1/15 for Throttle/Weir)
            flow_capacity_downstream = sum(full_flow[l] for l in links_downstream)

            for l in links_downstream:
                res_links[l] = float(
                    area_upstream * full_flow[l] / flow_capacity_downstream
                )

        else:
            # if we don't know the full flow capacity. e.g. for Pumps and Outlets

            if logging_func is not None:
                logging_func(f'cumulative_catchment_area | unknown divider for catchment area for object types {tuple(sorted([di_links[l]._section_label for l in links_downstream]))}.')

            # ---
            # another possibility ...
            # weights = {SEC.CONDUITS: 9,
            #            SEC.WEIRS: 5,  # q_voll ausrechnen oder von folge-conduit
            #            SEC.ORIFICES: 1,# q_voll ausrechnen
            #            SEC.PUMPS: 9 # von folge-conduit
            #            }

            for l in links_downstream:
                res_links[l] = area_upstream / len(links_downstream)

    return res_links
