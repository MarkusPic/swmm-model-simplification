from swmm_api.run_swmm import swmm5_run_epa

CONFIG = {
    'SWMM_runner': lambda fn: swmm5_run_epa(fn, init_print=False),
    # 'SWMM_runner': swmm5_run_owa,
    # 'SWMM_runner': lambda fn: swmm5_run_progress(fn, n_total=100),
    'min_flow_width': 7,  # meters
    'max_flow_width': 10_000,  # meters
    'objective_function': None,  # default Nash-Sutcliffe Efficiency
}
