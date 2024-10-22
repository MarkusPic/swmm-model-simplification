import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from swmm_api import SwmmInput
from swmm_api.input_file import SEC
from swmm_api.input_file.macros import inp_to_graph
from swmm_api.input_file.section_lists import SUBCATCHMENT_SECTIONS
from ..helpers.swmm_input_summary import (
    get_cross_section_summary,
)


def calibrate_conduits_on_hd(
        inp_agg: SwmmInput,
        inp_agg_clip: SwmmInput,
        link_full_flow_hd: dict[str, float],
        link_cross_section_area_hd: dict[str, float],
        link_cumulative_catchment_area_hd: dict[str, float],
        node: str,
        graph_hd: nx.DiGraph,
        optimize_volume=False,
        optimize_flow_full=True,
        optimize_flow_full_ratio=True,
        logging_func=None,
):
    """
    Calibrate the consolidated conduit directly connected upstream of the given node.

    Compare to the path of links in the HD model between the upstream and downstream node of the consolidated conduit.

    Set the storage volume equal to the volume of the HD model path by adjusting the cross-section height.
    Set the flow capacity to the most unfavourable capacity of the HD model path by adjusting the roughness of the link.

    Args:
        inp_agg (SwmmInput): SWMM input-file data of current state in simplification.
        inp_agg_clip (SwmmInput): SWMM input-file data of already cut-out model.
        link_full_flow_hd (dict[str, float]): The full flow rate for all conduits in the model.
        link_cross_section_area_hd (dict[str, float]): The cross-section area for all conduits in the model.
        link_cumulative_catchment_area_hd (dict[str, float]): The cumulative catchment area for each link in the network.
        node (str): label of the current node of interest. Calibrate the link directly connected upstream of this node.
        graph_hd (nx.DiGraph): A graph of the full (uncut) model network.
        optimize_volume (bool): when true - change the cross-section height to fit the volume of the consolidated conduit to the volume of the original path of conduit in the HD model.
        optimize_flow_full (bool): when false - use the full flow rate of the first (most upstream) conduit of the HD path that was consolidated. when true - use the smallest (most limiting) full flow rate of the HD path.
        optimize_flow_full_ratio (bool): when false - use the smallest full flow rate of the HD path. when true - use the smallest ratio of full flow rate and the connected catchment area to limit flow capacity.
        logging_func (function): function used for logging some messages. Default: no messages logged.
    """
    if not optimize_volume and not optimize_flow_full:
        return

    # slim model without unnecessary sections for this function.
    inp_agg_clip_copy = inp_agg_clip.copy()

    g = inp_to_graph(inp_agg_clip_copy)

    # if there are no conduits in the clipped part of the simplified model - skip.
    if SEC.CONDUITS not in inp_agg_clip_copy:
        return

    # remove all sections which are not needed to calculate the volume and full flow of the conduits.
    inp_agg_clip_copy.delete_sections(
        [SEC.PATTERNS, SEC.DWF, SEC.INFLOWS, SEC.RAINGAGES, SEC.TIMESERIES]
        + SUBCATCHMENT_SECTIONS,
    )

    for node_pre in g.predecessors(node):
        # For every upstream node in the simplified model.
        # if node is a start node, nothing will happen in this function.

        # Incoming link (upstream) in the simplified model.
        link_label = g.edges[(node_pre, node)]["label"]

        # Are there any parallel links in the simplified model? Skip if True.
        if isinstance(link_label, list):
            continue

        # only optimize conduits - skip other link types like Weir, Orifice, ...
        if link_label not in inp_agg_clip_copy.CONDUITS:
            continue

        # simple paths in the HD model between the adjacent nodes in simplified model
        # from the current node of interest upstream
        paths_hd = list(nx.all_simple_edge_paths(graph_hd, node_pre, node))

        # no path is not possible ...

        # if there are multiple paths in the HD model.
        # parallel path with intermediate nodes.
        # don't optimize because it should not happen in the algorithm
        # here for debugging reasons
        if len(paths_hd) != 1:
            logging_func(
                f"NOT IMPLEMENTED: calibrate_conduits_on_hd() -> parallel path with inter-nodes"
            )
            continue

        # list of links in simple path in HD model
        links_hd = [graph_hd.edges[n]["obj"] for n in paths_hd[0]]

        if any(isinstance(e, list) for e in links_hd):
            # parallel path WITHOUT intermediate nodes.
            # edge in graph is a list of links then the links are parallel and have the same start AND end nodes.
            # don't optimize because there is no reasonable solution.
            logging_func(
                f"NOT IMPLEMENTED: calibrate_conduits_on_hd() -> parallel path without inter-nodes"
            )
            continue

        # -------------------------------------------
        # path in HD model is a simple path without any parallel paths
        logging_func(f'_________________ Link "{link_label}" _________________')

        # === optimize storage volume of AGG link to fit storage volume of HD links ===
        full_height_old = inp_agg_clip_copy.XSECTIONS[link_label].height

        volume_hd = sum([l.length * link_cross_section_area_hd[l.name] for l in links_hd])

        volume_agg = (
                link_cross_section_area_hd[link_label]
                * inp_agg_clip_copy.CONDUITS[link_label].length
        )

        logging_func(
            f"OLD:height = {full_height_old:0.4f} m | HD:V_full = {volume_hd:6.2f} m³ | AGG:V_full = {volume_agg:0.2f} m³"
        )

        if abs(volume_hd - volume_agg) <= 0.01:
            logging_func("skip optimize volume - volumes are already equal")

        elif optimize_volume:

            def minimize_func(height):
                if height <= 0:
                    return -1e9 * height

                inp_agg_clip_copy.XSECTIONS[link_label].height = height

                xs = get_cross_section_summary(inp_agg_clip_copy)

                area = xs.at[link_label, "Full_Area"]
                volume_new = (
                        area * inp_agg_clip_copy.CONDUITS[link_label].length
                )
                logging_func(
                    f"  > {height = :0.4f} m |    V_full = {volume_new:6.2f} m³"
                )
                return abs(volume_new - volume_hd)

            results = minimize_scalar(
                minimize_func,
                bracket=(full_height_old * 0.9, full_height_old * 1.1),
                tol=1e-2,
            )

            logging_func(
                f"{results.nfev} iterations needed for optimal height."
            )

            height_new = results.x

            # set new cross-section height to all simplified models
            inp_agg.XSECTIONS[link_label].height = height_new
            inp_agg_clip.XSECTIONS[link_label].height = height_new
            inp_agg_clip_copy.XSECTIONS[link_label].height = height_new

            logging_func(f"{height_new = :0.4f} m")

        # === optimize roughness of link to fit full flow rate ===
        if optimize_flow_full:
            # calibrate the flow capacity on the HD model
            # look at the full flow rate as indicator of the flow capacity
            # change the roughness of the links to modify full flow rate.

            if len(links_hd) > 1:
                # only of HD conduits are consolidated (multiple HD links between nodes in simplified model)
                df = pd.DataFrame(
                    {
                        l.name: (
                            link_full_flow_hd[l.name],
                            link_cumulative_catchment_area_hd[l.name],
                        )
                        for l in links_hd
                    },
                    index=["Q_full", "area_cumulative"],
                ).T

                if (not optimize_flow_full_ratio) or df["area_cumulative"].lt(0.00001).any():
                    # if connected cumulative catchment area is very small (close to zero) then use the smallest full flow rate.
                    # of if the flow rate should NOT be calibrated on the ratio of full flow rate and connected area.
                    full_flow_hd_red = df["Q_full"].min()

                else:
                    # use the smallest ratio of full flow rate and connected area of all links in the HD model path
                    # multiply with actually connected area in simplified model
                    # use result as new full flow rate
                    df["Q/A"] = df["Q_full"] / df["area_cumulative"]
                    full_flow_hd_red = df["Q/A"].min() * link_cumulative_catchment_area_hd[link_label]

                if np.isnan(full_flow_hd_red):
                    logging_func(" ! " * 20)
                    logging_func("some cumulative catchment area or full flow rate is invalid in the HD model - can't optimize full flow rate.")

                # full flow rate of the first link in the HD model path - link that is extended with the conduit consolidation
                full_flow_hd = link_full_flow_hd[link_label]

                if full_flow_hd_red != full_flow_hd:
                    # compare new reduced (limited) full flow rate with the full flow rate of the first (most upstream) link in the path.
                    logging_func(f"\n{df.to_string()}")
                    logging_func(
                        f"<><><><><><> HD:Q_full = {full_flow_hd:7.2f} L/s -> {full_flow_hd_red:7.2f} L/s <><><><><><>"
                    )

                    logging_func(
                        f"HD:Q_full_reduced = {full_flow_hd_red:7.2f} L/s"
                    )

                full_flow_hd = full_flow_hd_red
            else:
                # get hd full flow rate of link left in simplified model.
                # link_label is the label in the simplified model which corresponds to the first (most upstream) link in the HD path.
                full_flow_hd = link_full_flow_hd[link_label]
                # consolidated Link has a different full flow rate because:
                #   the slope is changed due to the consolidation
                #   the cross-section is changed

            roughness_old = inp_agg_clip_copy.CONDUITS[link_label].roughness

            logging_func(
                f"OLD:roughness = {roughness_old:0.4f} | HD:Q_full = {full_flow_hd:7.2f} L/s"
            )

            def minimize_func(roughness):
                if roughness <= 0:
                    return -1e9 * roughness

                inp_agg_clip_copy.CONDUITS[link_label].roughness = roughness

                full_flow_new = get_cross_section_summary(inp_agg_clip_copy).at[
                    link_label, "Full_Flow"
                ]
                logging_func(
                    f"  > {roughness = :0.4f} |    Q_full = {full_flow_new:7.2f} L/s"
                )
                return abs(full_flow_new - full_flow_hd)

            results = minimize_scalar(
                minimize_func,
                bracket=(roughness_old / 2, roughness_old * 2),
                tol=1e-2,
            )

            logging_func(
                f"{results.nfev} iterations needed for optimal roughness."
            )

            roughness_new = results.x

            inp_agg.CONDUITS[link_label].roughness = roughness_new
            inp_agg_clip.CONDUITS[link_label].roughness = roughness_new
            inp_agg_clip_copy.CONDUITS[link_label].roughness = roughness_new

            logging_func(f"{roughness_new = :0.4f}")
