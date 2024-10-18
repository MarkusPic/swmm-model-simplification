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
    Calibrate the conduits

    Args:
        inp_agg (SwmmInput): SWMM input-file data of current state in simplification.
        inp_agg_clip (SwmmInput): SWMM input-file data of already cut-out model.
        link_full_flow_hd (dict[str, float]): The full flow rate for all conduits in the model.
        link_cross_section_area_hd (dict[str, float]): The cross-section area for all conduits in the model.
        link_cumulative_catchment_area_hd (dict[str, float]): The cumulative catchment area for each link in the network.
        node (str):
        graph_hd (nx.DiGraph): A graph of the full (uncut) model network.
        optimize_volume (bool): when true - change the cross-section height to fit the volume of the consolidated conduit to the volume of the original path of conduit in the HD model.
        optimize_flow_full (bool):
        optimize_flow_full_ratio (bool):
        logging_func (function): function used for logging some messages. Default: no messages logged.

    Returns:

    """
    inp_agg_clip_copy = inp_agg_clip.copy()

    g = inp_to_graph(inp_agg_clip_copy)

    # if there are not conduits in the clipped part of the simplified model - skip.
    if SEC.CONDUITS not in inp_agg_clip_copy:
        return

    # remove all sections which are not needed to calculate the volume and full flow of the conduits.
    inp_agg_clip_copy.delete_sections(
        [SEC.PATTERNS, SEC.DWF, SEC.INFLOWS, SEC.RAINGAGES, SEC.TIMESERIES]
        + SUBCATCHMENT_SECTIONS,
    )

    # Für jeden davor liegenden Knoten im aggregierten Modell
    # For every upstream node in the
    for node_pre in g.predecessors(node):
        # is previous node a start-node?

        # Link im aggregierten Modell
        link_label = g.edges[(node_pre, node)]["label"]

        # Gibt es parallele Links im aggregierten Modell
        if isinstance(link_label, list):
            continue

        # only optimize conduits
        if link_label not in inp_agg_clip_copy.CONDUITS:
            continue

        # HD-links between nodes in aggregated model
        paths_hd = list(nx.all_simple_edge_paths(graph_hd, node_pre, node))
        if len(paths_hd) == 1:
            path_hd_only_one = paths_hd[
                0
            ]  # liste and links die auf einfachen pfad liegen
            if any(
                isinstance(graph_hd.edges[e]["obj"], list) for e in path_hd_only_one
            ):
                # parallel pfad ohne zwischenknoten
                logging_func(
                    f"NOT IMPLEMENTED: optimize_full_flow_for_links -> parallel path without inter-nodes"
                )
                # TODO parallel Links -> gleichen Anfangs- und Endknoten
                # dann ist im 'obj' eine liste von objekten
            else:
                # einfacher pfad
                links_hd = [graph_hd.edges[n]["obj"] for n in path_hd_only_one]

                logging_func(f'_________________ Link "{link_label}" _________________')

                # -------------------------------------------
                # optimize volume of link to fit full volume in HD
                full_height_old = inp_agg_clip_copy.XSECTIONS[link_label].height

                volume_hd = sum([l.length * link_cross_section_area_hd[l] for l in links_hd])

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
                        if xs is None:
                            inp_agg_clip_copy.write_file("error_file.inp")
                            exit()
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

                    inp_agg.XSECTIONS[link_label].height = height_new
                    inp_agg_clip.XSECTIONS[link_label].height = height_new
                    inp_agg_clip_copy.XSECTIONS[link_label].height = height_new

                    logging_func(f"{height_new = :0.4f} m")

                # -------------------------------------------
                # optimize roughness of link to fit full flow
                if optimize_flow_full:
                    # get hd full flow of link
                    # use the minimum
                    full_flow_hd = link_full_flow_hd[link_label]
                    # full_flow_hd = full_flow_hd_list.loc[link_lengths_hd.index].min()

                    roughness_old = inp_agg_clip_copy.CONDUITS[link_label].roughness

                    logging_func(
                        f"OLD:roughness = {roughness_old:0.4f} | HD:Q_full = {full_flow_hd:7.2f} L/s"
                    )

                    # Vorher:
                    # obersten Link aus HD genommen und das Q_voll von diesem Link auf den aggregierten Link übertragen
                    # aggregierter Link hat anders Q_voll weil:
                    #   die Steigung verändert wurde durch das Zusammenführen
                    #   der Querschnitt (siehe oberhalb) verändert wurde.
                    # Nachher:
                    # geringstes Q_voll/A_ezg von allen Links in HD wählen und auf Q_voll vom Anfangsstrang ansetzen
                    # neues Q_voll als Basis - dann gleiches Vorgehen wie vorher.

                    # cumulative_catchment_area_hd = link_cumulative_catchment_area_hd[link_label]

                    if len(links_hd) > 1:
                        df = pd.DataFrame(
                            {
                                l.name: (
                                    link_full_flow_hd[l.name],
                                    link_cumulative_catchment_area_hd[l.name],
                                )
                                for l in links_hd
                            },
                            index=["Q_voll", "A_EZG"],
                        ).T

                        if (not optimize_flow_full_ratio) or df["A_EZG"].lt(0.00001).any():
                            # A kann nur überall oder am Anfang 0 sein.
                            # in beiden Fällen das kleinste Q_voll nehmen
                            full_flow_hd_red = df["Q_voll"].min()

                        else:
                            df["Q/A"] = df["Q_voll"] / df["A_EZG"]

                            full_flow_hd_red = (
                                df["Q_voll"] / df["A_EZG"]
                            ).min() * link_cumulative_catchment_area_hd[link_label]

                        if np.isnan(full_flow_hd_red):
                            logging_func(" ! " * 20)
                            # INF wenn A_EZG = 0

                        if full_flow_hd_red != full_flow_hd:
                            # print(df)
                            logging_func(f"\n{df.to_string()}")
                            logging_func(
                                f"<><><><><><> HD:Q_full = {full_flow_hd:7.2f} L/s -> {full_flow_hd_red:7.2f} L/s <><><><><><>"
                            )

                            logging_func(
                                f"HD:Q_full_reduced = {full_flow_hd_red:7.2f} L/s"
                            )

                        full_flow_hd = full_flow_hd_red

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
        else:
            # parallel pfad mit zwischenknoten
            logging_func(
                f"NOT IMPLEMENTED: optimize_full_flow_for_links -> parallel path with inter-nodes"
            )
