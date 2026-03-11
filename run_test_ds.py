import traceback
import sys

try:
    with open("test_ds.py") as f:
        exec(f.read())
except Exception as e:
    with open("trace.txt", "w") as out:
        traceback.print_exc(file=out)
