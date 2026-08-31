# app/state.py
#
# Why this file exists:
# ml_models used to live directly in main.py as a global dict. That worked
# fine when main.py owned all the routes. Now that /predict and /health live
# in app/routers/v1.py, that router needs access to the same dict — and if
# v1.py imported it from main.py while main.py imports the router from
# v1.py, we'd get a circular import (main -> v1 -> main -> ...).
#
# Pulling shared app state into its own module with no dependencies on
# main.py or the routers breaks that cycle. Both main.py (which populates
# it at startup via lifespan) and v1.py (which reads it per-request) import
# from here independently.

ml_models = {}