# Archived C++ App

`apps/` is no longer an active implementation root.
The historical `orchard_cli.cpp` source file has been removed from the main branch.

The historical `orchard_cli.cpp` application has been replaced by the Orchard FEM Python CLI:

```bash
python -m orchard_fem run examples/demo_orchard.json
python -m orchard_fem modal examples/demo_orchard.json
python -m orchard_fem visualize examples/demo_orchard.json build/demo_frequency_response.csv
```

Only inspect this directory when you explicitly need archived project history.
