import pandas as pd

ABLATION_CONFIGS = {
    'predictor_only':       {'use_nsga2': False, 'use_rl': False, 'use_trajectory': False},
    'predictor_nsga2':      {'use_nsga2': True,  'use_rl': False, 'use_trajectory': False},
    'predictor_rl':         {'use_nsga2': False, 'use_rl': True,  'use_trajectory': False},
    'predictor_nsga2_rl':   {'use_nsga2': True,  'use_rl': True,  'use_trajectory': False},
    'full_system':          {'use_nsga2': True,  'use_rl': True,  'use_trajectory': True},
}

def run_ablation_studies(cfg, all_components: dict) -> pd.DataFrame:
    """
    Placeholder simulating ablation studies. 
    In the real implementation this loops over ABLATION_CONFIGS and 
    executes subsets of the pipeline.
    """
    data = []
    
    data.append({"Config": "predictor_only", "TTF (Yrs)": 4.5, "Peak Reduction": "12%"})
    data.append({"Config": "predictor_nsga2", "TTF (Yrs)": 5.8, "Peak Reduction": "22%"})
    data.append({"Config": "predictor_rl", "TTF (Yrs)": 6.1, "Peak Reduction": "25%"})
    data.append({"Config": "predictor_nsga2_rl", "TTF (Yrs)": 6.9, "Peak Reduction": "32%"})
    data.append({"Config": "full_system", "TTF (Yrs)": 8.2, "Peak Reduction": "41%"})
    
    return pd.DataFrame(data)
