import logging
import sys
import warnings

import pandas as pd

from swmm_api import SwmmInput, CONFIG
from swmm_api.input_file.macros import set_crs

from swmm_model_simplification.simplifying_model import aggregate_model


def main():
    CONFIG['exe_path'] = "/Users/markus/.bin/runswmm"

    #%%
    inp = SwmmInput('model_bellinge.inp')
    set_crs(inp, 'EPSG:25832')

    # %%
    logger = logging.getLogger('simplify bellinge')
    logger.setLevel(logging.DEBUG)

    # logging.basicConfig(filename=os.path.join(PATH_CHANGES_LOG, 'flows.log'))

    fmt = logging.Formatter('[%(asctime)s] %(message)s')

    # log to the console/terminal
    # console_handler = logging.StreamHandler(sys.stderr)
    # console_handler.setFormatter(fmt)
    # logger.addHandler(console_handler)

    # log to a file
    file_handler = logging.FileHandler('log_simplify_bellinge.txt', mode='w', encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: logger.warning(str(message))

    # %%
    # set rain for raingauges TIMESERIES, RAINGAUGES, SUBCATCHMENTS
    # set times for rain in OPTIONS
    # set simulation options - routing step, ...
    inp.OPTIONS.set_routing_step(5)
    inp.OPTIONS.set_variable_step(0.75)
    inp.OPTIONS.set_lengthening_step(1)
    inp.OPTIONS.set_minimum_step(0.1)
    inp.OPTIONS.set_max_trials(24)
    inp.OPTIONS.set_threads(1)

    from swmm_api.input_file.sections import RainGage, TimeseriesData
    ts = TimeseriesData('block_rain', [(0, 0), (1, 20)])
    inp.add_obj(ts)

    rg = RainGage(name='RG', form=RainGage.FORMATS.VOLUME, interval='1:00',
                  SCF=1, source=RainGage.SOURCES.TIMESERIES, timeseries=ts.name)
    inp.add_obj(rg)

    for sc in inp.SUBCATCHMENTS:
        inp.SUBCATCHMENTS[sc].rain_gage = rg.name

    # set times so the
    inp.OPTIONS.set_start(pd.Timestamp.today().replace(minute=0, second=0, hour=0, microsecond=0, nanosecond=0))
    inp.OPTIONS.set_report_start(inp.OPTIONS.get_start() + pd.Timedelta(hours=1))
    inp.OPTIONS.set_simulation_duration(pd.Timedelta(hours=2))

    # %%
    inp_simple = aggregate_model(inp, area_min=5, length_max=400, logging_func=logger.debug, write_sc_transformation_as_tag=True)
    inp_simple.write_file('model_bellinge_simple.inp')
    # %%


if __name__ == "__main__":
    main()
