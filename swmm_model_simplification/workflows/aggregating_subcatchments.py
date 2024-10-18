import string
from itertools import groupby

import networkx as nx
import pandas as pd
from scipy.optimize import minimize_scalar
import shapely.geometry as shp

from swmm_api import SwmmInput
from swmm_api.input_file import SEC
from swmm_api.input_file.macros import subcatchments_connected
from swmm_api.input_file.sections import Tag, SubCatchment, SubArea, Polygon
from swmm_api.output_file import OBJECTS, VARIABLES
from swmm_api.run_swmm.run_temporary import swmm5_run_temporary

from .calibrating_conduits import calibrate_conduits_on_hd
from ..helpers.calculate_metrics import calculate_nse
from ..helpers.config import CONFIG
from ..helpers.create_sub_model import cut_network
from ..helpers.polygons_for_swmm import _to_coords_list, _include_inner_rings, remodel_poly


def _int_to_letter(col: int) -> str:
    """Convert given column number to an Excel-style column name."""
    # 0 = a
    n_letters = len(string.ascii_lowercase)

    result = ""
    col += 1
    while col:
        col, rem = divmod(col - 1, n_letters)
        result = string.ascii_lowercase[rem] + result
    return result


def aggregate_subcatchments(
    inp: SwmmInput,
    sc_list: list,
    g: nx.DiGraph = None,
    logging_func=None,
    kwargs_optimize: dict = None,
    suffix="",
    write_sc_transformation_as_tag=False,
):
    """
    First aggregate subcatchments to one, then calibrate conduits and finally calibrate subcatchment width.

    Calls following function:

      - :func:`~calibrate_subcatchments`.
        - :func:`~calibrate_conduits_on_hd`.

    Works inplace.

    Args:
        inp (swmm_api.SwmmInput): SWMM input-file data.
        sc_list (list[str]): list of SC to be aggregated to one SC.
        g (nx.DiGraph): optional - a graph of the model network to accelerate code - otherwise the graph will be created based on the input data.
        logging_func (function): function used for logging some messages. Default: no messages logged.
        kwargs_optimize (dict): keyword arguments passed to :func:`~calibrate_subcatchments`.
        suffix (str): suffix if multiple SC will exist on one node (multiple soil types).
        write_sc_transformation_as_tag (bool): if the new tag of the SC should be the name of the SC that were aggregated.
    """
    if len(sc_list) <= 1:
        # TODO implement: if the SC was moved downstream, the flow length was shortened and the width must be calibrated.
        # current assumption is that this time lag is small enough to be neglected.
        if (len(sc_list) == 1) and write_sc_transformation_as_tag:
            inp.add_obj(Tag(Tag.TYPES.Subcatch, sc_list[0].name, sc_list[0].name))
        return

    soil_groups = [
        list(g)
        for k, g in groupby(
            sorted(sc_list, key=lambda i: inp.INFILTRATION[i.name].values[1:]),
            key=lambda i: inp.INFILTRATION[i.name].values[1:],
        )
    ]

    # split SC when two different soil types are present
    i = 0
    if len(soil_groups) != 1:
        for sc_list_sub in soil_groups:
            # SC parameters will be first aggregated and only finally once calibrated.
            aggregate_subcatchments(
                inp,
                sc_list_sub,
                g=g,
                logging_func=logging_func,
                kwargs_optimize=None,
                suffix=_int_to_letter(i),
                write_sc_transformation_as_tag=write_sc_transformation_as_tag,
            )
            i += 1
        outlet = sc_list[0].outlet
    else:
        logging_func(f"<combine_subcatchments> (n={len(sc_list)})")

        sc_labels = [i.name for i in sc_list]
        logging_func(f"{sc_labels}")

        # _________________________
        # SUBCATCHMENTS
        df_sc = inp.SUBCATCHMENTS.slice_section(sc_labels).frame
        area_imperv = df_sc.area * df_sc.imperviousness / 100
        area_imperv_sum = area_imperv.sum()

        area_perv = df_sc.area * (100 - df_sc.imperviousness) / 100
        area_perv_sum = area_perv.sum()

        outlet = df_sc.outlet.unique()[0]
        logging_func(f"{outlet = }")
        label = f"SC_{outlet}{suffix}"
        logging_func(f"<new sc-name> {label}")

        area_sum = df_sc.area.sum()
        sc = SubCatchment(
            name=label,
            rain_gage=df_sc.rain_gage.unique()[0],  # most often used!
            outlet=outlet,
            area=area_sum,
            imperviousness=area_imperv_sum / area_sum * 100,
            width=(df_sc.area * df_sc.width).sum() / area_sum,
            slope=(df_sc.area * df_sc.slope).sum() / area_sum,
        )
        for sc_label in sc_labels:
            del inp.SUBCATCHMENTS[sc_label]
        inp.add_obj(sc)

        # _________________________
        # SUBAREAS
        df_sa = inp.SUBAREAS.slice_section(sc_labels).frame

        area_imperv_store = area_imperv * (100 - df_sa.pct_zero) / 100
        area_imperv_store_sum = area_imperv_store.sum()

        if area_imperv_sum == 0:
            area_imperv_sum = area_perv_sum
            area_imperv_store_sum = area_perv_sum

        # ---
        # how much impervious area is rerouted to the pervious subarea
        # search for pervious route_to
        # calculate pct_routed * area_imperv_i

        # if _:=0:
        #     area_imperv_rerouted_sum = 0
        #     for sc_label in sc_labels:
        #         sa_i = inp.SUBAREAS[sc_label]
        #         if sa_i.route_to == SubArea.RoutToOption.PERVIOUS:
        #             area_imperv_rerouted_sum += sa_i.pct_routed/100 * area_imperv[sc_label]
        #
        #     pct_routed = area_imperv_rerouted_sum / area_imperv_sum * 100
        #     # FUNTIONIERT NICHT => IMMER 100%
        # else:
        #     pct_routed = 90

        # if 0.01 < pct_routed < 99.9:
        #     print()

        route_to = SubArea.RoutToOption.OUTLET
        pct_routed = 100

        # if pct_routed < 0.01:
        #     route_to = SubArea.RoutToOption.OUTLET
        #     pct_routed = 100

        sa = SubArea(
            subcatchment=label,
            n_imperv=(area_imperv * df_sa.n_imperv).sum() / area_imperv_sum,
            n_perv=(area_perv * df_sa.n_perv).sum() / area_perv_sum,
            storage_imperv=(area_imperv_store * df_sa.storage_imperv).sum()
            / area_imperv_store_sum,
            storage_perv=(area_perv * df_sa.storage_perv).sum() / area_perv_sum,
            pct_zero=(area_imperv * df_sa.pct_zero).sum() / area_imperv_sum,
            route_to=route_to,
            pct_routed=pct_routed,
        )
        for sc_label in sc_labels:
            del inp.SUBAREAS[sc_label]
        inp.add_obj(sa)

        # _________________________
        # INFILTRATION
        df_i = inp.INFILTRATION.slice_section(sc_labels).frame
        c = df_i.value_counts(dropna=False).index
        i = inp.INFILTRATION._section_object(
            subcatchment=label, **dict(zip(list(c.names), list(c.values[0])))
        )
        for sc_label in sc_labels:
            del inp.INFILTRATION[sc_label]
        inp.add_obj(i)

        # _________________________
        # POLYGONS
        if SEC.POLYGONS in inp:
            df_p = inp.POLYGONS.slice_section(sc_labels).geo_series
            from shapely import GEOSException

            try:
                # poly = df_p.buffer(0, cap_style="square", join_style="bevel").union_all()
                poly = df_p.buffer(0).union_all()
            except GEOSException:
                logging_func(f"<ERROR for Polygon> {label}")
                poly = df_p[0]

                # df_p.buffer(.1, cap_style="square", join_style="bevel").union_all()
                # ax = df_p.plot()
                # ax.get_figure().show()

            if isinstance(poly, shp.MultiPolygon):
                logging_func(f"<MultiPolygon> {label}")

                # -------------
                polypoints = [_to_coords_list(p.exterior) for p in poly.geoms]
                new_points = _include_inner_rings(polypoints[0], polypoints[1:])
                poly = shp.Polygon(new_points)

                # -------------
                # points = []
                # for polygon in poly.geoms:
                #     points.extend(polygon.exterior.coords[:-1])
                # -------------

                # from scipy.spatial import ConvexHull
                # ch = ConvexHull(points, incremental=True)
                # poly = shp.Polygon(ch.points)
                # -------------
                # from shapely import concave_hull, convex_hull
                # poly2 = concave_hull(sh.MultiPoint(points))
                # -------------
                # import matplotlib.pyplot as plt
                # from shapely.plotting import plot_polygon
                # fig, ax = plt.subplots()
                # plot_polygon(poly, ax=ax)
                # fig.show()

                # -------------
                if poly.interiors:
                    logging_func(f"<Interior> {label}")
                    poly = remodel_poly(poly)

                # poly = poly.convex_hull

            # poly = poly.simplify(0.1)

            p = Polygon.from_shapely(subcatchment=label, polygon=poly)
            for sc_label in sc_labels:
                del inp.POLYGONS[sc_label]
            inp.add_obj(p)

        # _________________________
        # delete tag without replacement
        if SEC.TAGS in inp:
            for sc_label in sc_labels:
                if (Tag.TYPES.Subcatch, sc_label) in inp.TAGS:
                    del inp.TAGS[(Tag.TYPES.Subcatch, sc_label)]

        if write_sc_transformation_as_tag:
            inp.add_obj(Tag(Tag.TYPES.Subcatch, label, "+".join(sc_labels)))

        # --------------------
        if g is not None:
            for i in sc_list:
                g.remove_node(i.name)

            g.add_node(sc.name, obj=sc)
            g.add_edge(sc.name, sc.outlet, label=f"Outlet({sc.name})")

    # -----------------
    if kwargs_optimize is not None:
        calibrate_subcatchments(node=outlet, **kwargs_optimize)


def calibrate_subcatchments(
    inp_agg: SwmmInput,
    inp_hd: SwmmInput,
    node: str,
    graph_hd: nx.DiGraph,
    logging_func,
    current_ts_calculated_hd,
    current_ts_calculated_agg,
    # following arguments are only for calibrate_conduits_on_hd
    link_full_flow_hd: dict[str, float],
    link_cross_section_area_hd: dict[str, float],
    link_cumulative_catchment_area_hd: dict[str, float],
    optimize_volume=True,
    optimize_flow_full=True,
    optimize_flow_full_ratio=True,
):
    """
    Calibrate subcatchments, that are connected to the given node, on the HD simulation results.

    Works inplace.

    This function is currently fixed on metric units.

    Args:
        inp_agg (SwmmInput): SWMM input-file data of current state in simplification.
        inp_hd (SwmmInput): SWMM input-file data of the original HD model.
        node (str): label of the node
        graph_hd (nx.DiGraph):
        logging_func (function): function used for logging some messages. Default: no messages logged.
        current_ts_calculated_hd (dict[str: pd.Series]): Flow timeseries in nodes if HD model calculated previously; with key=node_label and value=pd.Series of total_node_inflow.
        current_ts_calculated_agg (dict[str: pd.Series]): Flow timeseries in nodes if AGG model calculated previously; with key=node_label and value=pd.Series of total_node_inflow.

        # following arguments are only for calibrate_conduits_on_hd

        link_full_flow_hd (dict[str, float]): The full flow rate for all conduits in the model.
        link_cross_section_area_hd (dict[str, float]): The cross-section area for all conduits in the model.
        link_cumulative_catchment_area_hd (dict[str, float]): The cumulative catchment area for each link in the network.
        optimize_volume (bool):
        optimize_flow_full (bool):
        optimize_flow_full_ratio (bool):
    """
    logging_func(f'Start optimize for node "{node}"')

    # ------------------------------------
    # run inp hd
    # only use the upstream part of the network
    # cut the network at previously calculated nodes
    # get the timeseries of the node inflow
    if node not in current_ts_calculated_hd:
        # inp_hd_clip = split_network2(inp_hd.copy(), node, graph=graph_hd.copy(), logger_func=lambda x: logger_func_opt('HD: ' + x), current_ts_calculated=CURRENT_TS_CALCULATED_HD)
        inp_hd_clip = cut_network(
            inp_hd.copy(),
            node,
            logging_func=lambda x: logging_func("HD: " + x),
            dict_pre_calculated_ts=current_ts_calculated_hd,
            graph=graph_hd.copy(),
        )

        # ------------------------------------
        inp_hd_clip.REPORT["NODES"] = node

        # set init_defizit 0
        for i in inp_hd_clip.INFILTRATION.values():
            i.moisture_deficit_init = 0

        with swmm5_run_temporary(
                inp_hd_clip, run=CONFIG["SWMM_runner"], label="HD"
        ) as res:
            ts_hd = res.out.get_part(OBJECTS.NODE, node, VARIABLES.NODE.TOTAL_INFLOW)

        current_ts_calculated_hd[node] = ts_hd

    ts_hd = current_ts_calculated_hd[node]
    # ------------------------------------
    # only use the upstream part of the network
    # cut the network at previously calculated nodes
    # inp_agg_clip = split_network2(inp_agg.copy(), node, graph=None, logger_func=lambda x: logger_func_opt('AGG: ' + x), current_ts_calculated=CURRENT_TS_CALCULATED_AGG)
    # upstream_nodes_list darf nicht <node> enthalten!!!
    inp_agg_clip = cut_network(
        inp_agg.copy(),
        node,
        logging_func=lambda x: logging_func("AGG: " + x),
        dict_pre_calculated_ts=current_ts_calculated_hd,
    )  # CURRENT_TS_CALCULATED_AGG

    # if node == '17012':
    #     inp_agg_clip.write_file('interim_17012_agg.inp')
    #     write_geo_package(inp_agg_clip, PTH_CHIANTI_local_GIS / f'debugging_17012_agg.gpkg')
    #     exit()
    # ------------------------------------
    # optimize full flow
    # optimize volume of links
    calibrate_conduits_on_hd(
        inp_agg,
        inp_agg_clip,
        link_full_flow_hd,
        link_cross_section_area_hd,
        link_cumulative_catchment_area_hd,
        node,
        graph_hd,
        optimize_volume=optimize_volume,
        optimize_flow_full=optimize_flow_full,
        optimize_flow_full_ratio=optimize_flow_full_ratio,
        logging_func=logging_func,
    )

    # ------------------------------------
    # run inp agg

    # displayed units:
    UNIT_FLOW = inp_agg.OPTIONS['FLOW_UNITS'].upper()
    if inp_agg.OPTIONS.is_imperial():  # # CFS*/GPM/MGD
        UNIT_AREA = 'acres'  # area of SC
        UNIT_LENGTH = 'ft'  # width of SC
    else:
        if UNIT_FLOW == 'LPS':
            UNIT_FLOW = 'L/s'
        elif UNIT_FLOW == 'CMS':
            UNIT_FLOW = 'm³/s'
        # MLD: million liters per day
        UNIT_AREA = 'ha'  # area of SC
        UNIT_LENGTH = 'm'  # width of SC

    # ------------------------------------
    sc_connected = subcatchments_connected(inp_agg_clip, node)

    inp_agg_clip.REPORT["NODES"] = node

    # set init_defizit 0 => doesn't change much
    for i in inp_agg_clip.INFILTRATION.values():
        i.moisture_deficit_init = 0

    multiple_sc = len(sc_connected) > 1

    if multiple_sc:
        # this node has multiple SC (with differenz soil types) connected.

        sc_biggest = max(sc_connected, key=lambda s: s.area)
        width_old = sc_biggest.width

        # the original ratio of flow length to flow width should be preserved
        ratio_biggest = sc_biggest.area / sc_biggest.width**2

        ratios = {
            sc.name: (sc.area / sc.width**2) / ratio_biggest for sc in sc_connected
        }

        areas = {sc.name: round(sc.area, 1) for sc in sc_connected}
        logging_func(f"areas = {areas} {UNIT_AREA}")

        ratio_l_w = {
            sc.name: round(sc.area / sc.width**2 * 1e4, 1) for sc in sc_connected
        }
        logging_func(f"ratios FL/FW = {ratio_l_w}")

        widths = {sc.name: round(sc.width, 3) for sc in sc_connected}
        logging_func(
            f"OLD:width = {widths} {UNIT_LENGTH} | HD:max = {ts_hd.max():0.2f} {UNIT_FLOW} | HD:sum = {ts_hd.sum():0.2f}"
        )

        # TODO this node hase multiple SC connected
        # calculated FL/FW ratio and fix ratio between SC
        # manipulate smaller SC's width with this ratio
        def prep_inp(inp, width):
            ratio_biggest_new = sc_biggest.area / width**2
            for sc in sc_connected:
                ratio_length_by_width = ratios[sc.name] * ratio_biggest_new
                # print((sc.area / ratio_length_by_width)**(1/2))
                inp.SUBCATCHMENTS[sc.name].width = (sc.area / ratio_length_by_width) ** (1 / 2)
            return inp

    else:
        try:
            sc = sc_connected[0]
        except:
            print()
        width_old = sc.width
        area = sc.area
        ratio_l_w = area / width_old**2 * 1e4

        logging_func(f"area = {area:6.2f} {UNIT_AREA} | ratio FL/FW = {ratio_l_w:0.1f}")
        logging_func(
            f"OLD:width = {width_old:6.2f} {UNIT_LENGTH} | HD:max = {ts_hd.max():0.2f} {UNIT_FLOW} | HD:sum = {ts_hd.sum():0.2f}"
        )

        def prep_inp(inp, width):
            inp.SUBCATCHMENTS[f"SC_{node}"].width = width
            return inp

    def minimize_func(width):
        # force width between user set limits
        if width <= CONFIG['min_flow_width']:
            logging_func(f"  > {width = :6.2f} m")
            return 1e6 * (11 - width)
        elif width > CONFIG['max_flow_width']:
            logging_func(f"  > {width = :6.2f} m")
            return 1e3 * width

        if multiple_sc:
            # TODO limit ratio for multiple SC
            pass
        else:
            ratio_l_w_new = area / width ** 2 * 1e4

            if ratio_l_w_new < 0.9:
                logging_func(
                    f"  > {width = :6.2f} {UNIT_LENGTH} | ratio FL/FW = {ratio_l_w_new:0.1f}"
                )
                return 1e6 * (1 - ratio_l_w_new)
            elif ratio_l_w_new > 200:
                logging_func(
                    f"  > {width = :6.2f} {UNIT_LENGTH} | ratio FL/FW = {ratio_l_w_new:0.1f}"
                )
                return 1e6 * ratio_l_w_new
            elif 0.9 < ratio_l_w_new < 200:
                pass

        with swmm5_run_temporary(
                prep_inp(inp_agg_clip, width),
                run=CONFIG["SWMM_runner"],
                label="opt-agg-width",
        ) as res:
            ts_agg = res.out.get_part(OBJECTS.NODE, node, VARIABLES.NODE.TOTAL_INFLOW)

        nse = calculate_nse(ts_hd, ts_agg)
        logging_func(
            f"  > width = {width:6.2f} {UNIT_LENGTH} |    max = {ts_agg.max():0.2f} {UNIT_FLOW} |    sum = {ts_agg.sum():0.2f} | {nse = :0.3f}"
        )
        return round(1 - nse, 3)

    # results = minimize_scalar(minimize_func, tol=1e-1, bounds=(width_old/2, width_old*2))
    results = minimize_scalar(
        minimize_func, bracket=(width_old * 3 / 4, width_old), tol=1
    )
    # "bracket" ist der jeweils erste und zweite iterationsschritt...
    # tol = xtol => kleinster schritt für neue width

    logging_func(f"{results.nfev} iterations needed for optimal width.")

    width_new = results.x
    inp_agg = prep_inp(inp_agg, width=width_new)  # neue Modellbasis
    inp_agg_clip = prep_inp(inp_agg_clip, width=width_new)  #

    # ----------
    logging_func(
        f"{width_new = :0.2f} {UNIT_LENGTH} | ==> final NSE = {1 - results.fun:0.3f} <==\n"
        + "_" * 50
    )

    # ----------
    # zwischenspeichern der Zeitreihe in den Arbeitsspeicher
    # wird für das Splitten vom Modell verwendet (schnellere Laufzeiten)
    with swmm5_run_temporary(inp_agg_clip, run=CONFIG["SWMM_runner"]) as res:
        ts_agg = res.out.get_part(OBJECTS.NODE, node, VARIABLES.NODE.TOTAL_INFLOW)
    current_ts_calculated_agg[node] = ts_agg
