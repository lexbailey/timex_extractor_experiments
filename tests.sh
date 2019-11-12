#!/usr/bin/env bash
set -e
./timex_finder_bayes.py
#./timex_finder_deep_nn.py
./timex_finder_regex.py
./run_experiments.py
./build_web
