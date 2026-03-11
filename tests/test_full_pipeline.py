import pytest
import os
import subprocess
from pathlib import Path

def test_full_pipeline_smoke():
    # Execute the monolithic pipeline script
    # It incorporates very small batch sizes automatically for fast sanity checks
    cwd = Path(__file__).parent.parent.absolute()
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    
    result = subprocess.run(
        ["python", "scripts/run_full_pipeline.py"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True
    )
    
    # Should not crash
    assert result.returncode == 0, f"Pipeline execution failed:\nStdout: {result.stdout}\nStderr: {result.stderr}"
    
    # Verify mandatory artifacts spawned
    assert (cwd / "checkpoints/trajectory_best.pt").exists()
    assert (cwd / "checkpoints/rl_policy_final.pt").exists()
    assert (cwd / "paper/tables/prediction_accuracy.tex").exists()
    assert (cwd / "paper/plots/pareto_frontier_3d.pdf").exists()
