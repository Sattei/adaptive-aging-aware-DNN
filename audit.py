import os
import sys
import glob
import importlib
import json

def get_repo_files():
    files = []
    stubs = {}
    for root, _, filenames in os.walk('.'):
        if '.git' in root or '__pycache__' in root or 'data' in root or 'venv' in root:
            continue
        for f in filenames:
            if f.endswith(('.py', '.yaml', '.txt', '.md')):
                path = os.path.join(root, f)
                files.append(path)
                
                if f.endswith('.py'):
                    try:
                        with open(path, 'r', encoding='utf-8') as file:
                            lines = file.readlines()
                            has_stubs = any(kw in line for line in lines for kw in ['pass', 'NotImplementedError', 'TODO', 'FIXME', '...'])
                            stubs[path] = {
                                'lines': len(lines),
                                'has_stubs': has_stubs
                            }
                    except Exception:
                        pass
    return files, stubs

def check_imports():
    modules = [
        "simulator.timeloop_runner", "simulator.workload_runner",
        "features.activity_extractor", "features.feature_builder",
        "aging_models.nbti_model", "aging_models.hci_model",
        "aging_models.tddb_model", "aging_models.aging_label_generator",
        "graph.accelerator_graph", "graph.graph_dataset",
        "models.hybrid_gnn_transformer", "models.trajectory_predictor",
        "models.training_pipeline", "planning.lifetime_planner",
        "optimization.nsga2_optimizer", "optimization.chromosome_representation",
        "rl.environment", "rl.policy_network", "rl.trainer",
        "scheduler.runtime_mapper", "evaluation.reliability_metrics",
        "evaluation.performance_metrics", "evaluation.statistical_tests",
        "visualization.aging_heatmap", "visualization.trajectory_plots",
        "visualization.pareto_plots", "visualization.architecture_diagrams",
        "experiments.baseline_experiments", "experiments.ablation_studies"
    ]
    results = {}
    for mod in modules:
        try:
            importlib.import_module(mod)
            results[mod] = "OK"
        except Exception as e:
            results[mod] = f"FAIL: {type(e).__name__}: {str(e)}"
    return results

def check_packages():
    packages = [
        ("torch", "torch.__version__"), ("torch_geometric", "torch_geometric.__version__"),
        ("pymoo", "pymoo.__version__"), ("gymnasium", "gymnasium.__version__"),
        ("networkx", "networkx.__version__"), ("numpy", "numpy.__version__"),
        ("scipy", "scipy.__version__"), ("pandas", "pandas.__version__"),
        ("matplotlib", "matplotlib.__version__"), ("hydra", "hydra.__version__"),
        ("omegaconf", "omegaconf.__version__"), ("wandb", "wandb.__version__"),
        ("sklearn", "sklearn.__version__"), ("pytest", "pytest.__version__")
    ]
    results = {}
    for pkg, version_attr in packages:
        try:
            mod = importlib.import_module(pkg)
            parts = version_attr.split(".")
            val = mod
            for p in parts[1:]:
                val = getattr(val, p)
            results[pkg] = val
        except ImportError:
            results[pkg] = "NOT INSTALLED"
        except Exception as e:
            results[pkg] = f"ERROR: {type(e).__name__}"
    return results

def check_inits():
    dirs = ["simulator", "features", "aging_models", "graph", "models", 
            "planning", "optimization", "rl", "scheduler", "evaluation", 
            "visualization", "experiments"]
    results = {}
    for d in dirs:
        results[d] = os.path.isfile(os.path.join(d, "__init__.py"))
    return results

if __name__ == "__main__":
    files, stubs = get_repo_files()
    imports = check_imports()
    packages = check_packages()
    inits = check_inits()
    
    with open('diagnostic_data.json', 'w') as f:
        json.dump({
            'files': files,
            'stubs': stubs,
            'imports': imports,
            'packages': packages,
            'inits': inits
        }, f, indent=2)
