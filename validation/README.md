Validation support directory for the pKaNET 72.65% benchmark figure.

This directory also includes staged copies of the two larger validation/support
datasets requested for GitHub upload:

- `support/reproduction_pkanet_vs_obabel_285`
- `support/box_18a_support`

Included files

- `run_benchmark_gate.py`
  Computes benchmark accuracy from `benchmark_27k_relabeled_v81.csv` using `heuristic_net_charge()`.
- `update_main_pka_validation.py`
  Inserts the manuscript sentence that hard-codes `72.65% (19,685/27,096)`.
- `pKaNET.py`
  Source file containing `heuristic_net_charge()`, which is called by `run_benchmark_gate.py`.
- `support/reproduction_pkanet_vs_obabel_285`
  Staged copy of the pKaNET vs Open Babel rerun materials.
- `support/box_18a_support`
  Staged copy of the 18 x 18 x 18 box-size validation materials.

Not included

- `benchmark_27k_relabeled_v81.csv`
  This benchmark input file is referenced by the scripts but is not present in this repository snapshot.

Relevant relationships

- `run_benchmark_gate.py` imports `heuristic_net_charge` from local `pKaNET.py`.
- `update_main_pka_validation.py` does not compute the value; it only writes the already determined `72.65%` figure into manuscript text.
