# Why this file exists:
# ml_models used to live directly in main.py as a global dict. That worked
# fine when main.py owned all the routes. Now that prediction endpoints live
# in app/routers/v1.py and app/routers/v2.py, those routers need access to
# the same dict — and if a router imported it from main.py while main.py
# imports the router from it, we'd get a circular import (main -> v1 -> main -> ...).
#
# Pulling shared app state into its own module with no dependencies on
# main.py or the routers breaks that cycle. main.py (which populates it at
# startup via lifespan) and both v1.py and v2.py (which read it per-request)
# all import from here independently.

ml_models = {}