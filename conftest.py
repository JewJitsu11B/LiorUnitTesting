import sys, os

# Make the local cal/ package (copied into testpy/cal/) importable when the suite is
# run from this folder, so the consolidated testpy/ is self-contained.
sys.path.insert(0, os.path.dirname(__file__))
