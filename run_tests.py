import subprocess
import sys
try:
    result = subprocess.run(["pytest", "tests/", "-v"], capture_output=True, text=True)
    with open("pytest_clean.log", "w") as f:
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
    print("Pytest executed. Log written to pytest_clean.log.")
except Exception as e:
    print(e)
