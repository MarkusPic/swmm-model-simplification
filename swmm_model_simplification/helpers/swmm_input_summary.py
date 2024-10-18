import pandas as pd

from swmm_api import SwmmInput
from swmm_api.input_file import SEC
from swmm_api.input_file.sections import ReportSection, Conduit, Weir, Orifice
from swmm_api.run_swmm.run_temporary import swmm5_run_temporary
from .config import CONFIG


def prep_run_input_summary(inp: SwmmInput):
    """Set SWMM options and reporting settings for only getting input summary tables in report file."""
    inp_cpy = inp.copy()

    # deactivate all solvers
    inp_cpy.OPTIONS.set_ignore_routing(True)
    inp_cpy.OPTIONS.set_ignore_rainfall(True)
    # set shortest simulation duration
    inp_cpy.OPTIONS.set_simulation_duration(pd.Timedelta(seconds=2))
    inp_cpy.OPTIONS.set_report_step(1)
    inp_cpy.OPTIONS.set_routing_step(1)

    # only activate input summary tables for the report files.
    inp_cpy[SEC.REPORT] = ReportSection()
    inp_cpy.REPORT.set_input(True)
    inp_cpy.REPORT.set_continuity(False)
    inp_cpy.REPORT.set_flowstats(False)
    return inp_cpy


def prep_run_links_to_conduit(inp: SwmmInput, roughness=0.0125, length=50):
    """Delete Controls and convert orifices and weirs to conduits."""
    # length has no effect but must be set
    # roughness has a big effect and must be set accordingly
    inp_cpy = inp.copy()
    if SEC.CONTROLS in inp_cpy:
        del inp_cpy[SEC.CONTROLS]

    if SEC.ORIFICES in inp_cpy:
        for label in list(inp_cpy.ORIFICES):
            link = inp_cpy[SEC.ORIFICES].pop(label)  # type: Orifice

            inp_cpy.add_obj(
                Conduit(
                    name=label,
                    from_node=link.from_node,
                    to_node=link.to_node,
                    offset_upstream=link.offset,
                    roughness=roughness,
                    length=length,
                )
            )

    if SEC.WEIRS in inp_cpy:
        for label in list(inp_cpy.WEIRS):
            link = inp_cpy[SEC.WEIRS].pop(label)  # type: Weir

            inp_cpy.add_obj(
                Conduit(
                    name=label,
                    from_node=link.from_node,
                    to_node=link.to_node,
                    offset_upstream=link.height_crest,
                    roughness=roughness,
                    length=length,
                )
            )
    return inp_cpy


def get_cross_section_summary(inp: SwmmInput) -> pd.DataFrame:
    """Run SWMM in a temporary manner and only return the cross-section summary table."""
    with swmm5_run_temporary(
        prep_run_input_summary(prep_run_links_to_conduit(inp)),
        run=CONFIG["SWMM_runner"],
        label="cross_section_summary",
    ) as res:
        xs = res.rpt.cross_section_summary
        # slope = res.rpt.link_summary['%Slope']
        # print(res.rpt.analyse_duration)
    return xs


def get_link_full_flow(inp: SwmmInput) -> dict[str, float]:
    """Get the full flow rate for all conduits in the model."""
    return get_cross_section_summary(inp)["Full_Flow"].to_dict()


def get_link_cross_section_area(inp: SwmmInput) -> dict[str, float]:
    """Get the cross-section area for all conduits in the model."""
    return get_cross_section_summary(inp)["Full_Area"].to_dict()


def get_full_flow_velocity(inp: SwmmInput) -> pd.Series:
    """Get the full flow velocity for all conduits in the model."""
    df = get_cross_section_summary(inp)
    return df["Full_Flow"] / df["Full_Area"] / 1000
