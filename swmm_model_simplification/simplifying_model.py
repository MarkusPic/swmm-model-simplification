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
    area_min=5,
    length_max=400,
    skip_optimisation=False,
    optimize_volume=True,
    optimize_flow_full=True,
    optimize_flow_full_ratio=True,
    write_sc_transformation_as_tag=False,
    logging_func=None,
):
    """
    Simplifies SWMM model using the set rainfall. Auto-calibrated based on flow.

    Tested on small rainfall event with no flooding.

    Preparation:
        - set rain for raingauges TIMESERIES, RAINGAUGES, SUBCATCHMENTS
        - set times for rain in OPTIONS
        - set report start for start time of calibration time range for the NSE in OPTIONS
        - set simulation options - routing step, ...

    Args:
        inp_hd (SwmmInput): SWMM input data.
        area_min (int | float):
        length_max (int | float):
        skip_optimisation (bool):
        write_sc_transformation_as_tag (bool): if the new tag of the SC should be the name of the SC that were aggregated.
        logging_func (function): function used for logging some messages. Default: no messages logged.
        optimize_volume (bool):
        optimize_flow_full (bool):
        optimize_flow_full_ratio (bool):
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

    combine_conduits_list = []

    def log_sub(msg):
        return logging_func_agg(" " * 8 + " | " + msg)

    for node_current in tqdm(
        iter_depth_first_search(graph_sewer, logging_func=logging_func_agg),
        total=len(graph_sewer.nodes),
        desc="simplify model - iter over all nodes",
    ):

        # progressbar.update(-1)
        if combine_conduits_list:
            # log_sub('combine_conduits_list available')
            # while adding conduits to combine_conduits list -> skip this part
            last_node = combine_conduits_list[-1].to_node

            if (
                last_node != node_current
            ):  # not in set(graph_with_sc.successors(last_node)):
                log_sub("<combine_conduits_list - new path>")
                combine_conduits_list = consolidate_conduits(
                    inp,
                    combine_conduits_list,
                    length_max,
                    graph_with_sc=graph_with_sc,
                    logging_func=log_sub,
                    kwargs_optimize=kwargs_optimize,
                    write_sc_transformation_as_tag=write_sc_transformation_as_tag,
                )

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

        def log_node(msg):
            return logging_func_agg(f"{node_current:8s} | {msg}")

        pre_nodes = []
        sc_connected_list = []
        for k in graph_with_sc.predecessors(node_current):
            if k in inp.SUBCATCHMENTS:
                sc_connected_list.append(inp.SUBCATCHMENTS[k])
            else:
                pre_nodes.append(k)

        n_inflows = len(pre_nodes)  # graph_with_sc.in_degree[node_current]

        skip_combine_conduits = False

        # -----------------
        # Vorhergehender Knoten,
        # Muss im aktuellen Pfad sein und in den oberhalb liegenden Knoten
        # node_previous = None
        # Möglichkeiten
        # Anfangsknoten im Agg-Modell aber nicht im HD-Modell
        # Es gibt einen vorherigen Knoten
        # Es gibt zwei vorherige Knoten

        # -----------------
        if node_current not in inp.JUNCTIONS:  # or fixed/problem node
            # Wenn der Knoten kein Junction ist
            log_node("<keep node - not a junction>")
        # -----------------
        elif n_inflows == 0:  # Startknoten => keine Zuläufe
            area = sum(
                [sc.area for sc in sc_connected_list]
            )  # sum of all connected SC area to node

            if graph_with_sc.out_degree[node_current] > 1:  # Verzweigung nach unten
                log_node(
                    f"<keep node - start node with branching downstream> {skip_combine_conduits=}"
                )

            elif area < area_min:
                # Wenn die Fläche zu klein (unbedeutend) ist (oder keine SC angeschlossen sind),
                # dann wird der Knoten gelöscht und alle Zuflüsse und angeschlossenen SC zum nächsten Knoten geleitet.

                # Wenn eine der nächsten Haltungen kein Conduit ist, bleibt der Knoten erhalten.
                if any(
                    [
                        not isinstance(l, Conduit)
                        for l in next_links(inp, node_current, g=graph_sewer)
                    ]
                ):
                    log_node(
                        f"<keep node - next link not a conduit> {skip_combine_conduits=}"
                    )

                # elif next_sc_has_different_soil(sc_connected_list, graph_with_sc, node_current, inp):
                #     log_node('<keep node - new soil type>')

                else:
                    delete_node(
                        inp,
                        node_current,
                        graph=graph_with_sc,
                        alt_node=next(graph_with_sc.successors(node_current)),
                    )
                    log_node("<delete_node>")
                    nodes_done.add(node_current)
                    # progress.update()
                    continue  # nichts mehr zu tun

            elif len(sc_connected_list) > 1:
                # current node has a big enough catchment area or is a special node.
                # and has subcatchments connected
                # then combine connected SC
                log_node("<keep node - area big enough>")

            else:
                log_node("<no SC connected>")

        # -----------------
        elif n_inflows == 1:
            # Es gibt einen Zulauf (einen Knoten oberhalb)
            # node_before = pre_nodes[0]  # path_downstream[i - 1]

            # Wenn der Knoten vorher noch nicht bearbeitet wurde,
            # handelt es sich um eine Verzweigung oberhalb,
            # wobei der Knoten oberhalb dieses Pfades vorher gelöscht wurde.
            # if node_before not in nodes_done:
            #     log_node('<break - branch with unknown (in HD)>\n'+'-'*20)  # branch ...
            #     continue  # Pfad abbrechen

            # Wenn es genau einen Knoten unterhalb gibt
            if graph_with_sc.out_degree[node_current] == 1:
                links_before, links_after = links_connected(
                    inp, node_current, g=graph_with_sc
                )

                # Durch die vorherigen Abfragen gibt es jeweils nur einen Link vorher und nachher.
                link0 = links_before[0]
                link1 = links_after[0]

                # Wenn beide Links conduits sind -> zusammenführen
                if isinstance(link0, Conduit) and isinstance(link1, Conduit):
                    # if conduits_are_equal(inp, link0, link1, diff_height=0.15, diff_slope=0.3):
                    if link0 not in combine_conduits_list:
                        combine_conduits_list.append(link0)
                        log_node(f'<add to combine_conduits> "{link0.name}"')
                    if link1 not in combine_conduits_list:
                        combine_conduits_list.append(link1)
                        log_node(f'<add to combine_conduits> "{link1.name}"')
                    # log_node(f'<add to combine_conduits> ... {[c.name for c in combine_conduits_list]}')
                    skip_combine_conduits = True  # don't combine conduits

                # zumindest ein angeschlossener Link ist kein Conduit -> nicht zusammenführen
                # elif combine_conduits_list:  # Wenn es von vorher noch Conduits gibt, die zusammengefasst werden sollen
                #     # do_combine_conduits = True
                #     log(_log_label + f'<combine_conduits - special link>')
                # dissolve_conduits(inp, combine_conduits_list, length_max, g=graph_with_sc, log_func=log)
                # combine_conduits_list = []

                else:  # Wenn vorher bereits ein spezieller Link war
                    log_node("<keep node - not a conduit>")  # | conduit too long

            elif graph_with_sc.out_degree[node_current] > 1:  # Verzweigung nach unten
                log_node(f"<keep node - branching downstream> {skip_combine_conduits=}")

            else:  # Letzter Knoten im Netz
                log_node("<last node>")

        # -----------------
        elif n_inflows > 1:  # Knoten mit Verzweigung oberhalb
            log_node("<keep node - branch upstream>")

        else:
            log_node("<last node>")

        # -----------------
        if (
            not skip_combine_conduits
        ):  # while adding conduits to combine_conduits list -> skip this part
            log_sub("<combine_conduits_list inline>")
            combine_conduits_list = consolidate_conduits(
                inp,
                combine_conduits_list,
                length_max,
                graph_with_sc=graph_with_sc,
                logging_func=log_sub,
                kwargs_optimize=kwargs_optimize,
                write_sc_transformation_as_tag=write_sc_transformation_as_tag,
            )

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
        f"Auto-calibration: {'skipped' if skip_optimisation else ('conduit volume + ' if optimize_volume else '')+('full flow rate' if optimize_flow_full else '')+('full flow rate ratio' if optimize_flow_full_ratio else '')}\n"
        f"Operating system: {sys.platform}\n"
        f"Timestamp: {time_end}\n"
        f"Simplification duration: {time_end - time_start}"
    )

    return inp
