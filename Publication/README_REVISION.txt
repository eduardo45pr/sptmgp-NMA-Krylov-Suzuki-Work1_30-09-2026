PAPER 1 - REVISED WORKING PACKAGE

Primary paper:
A Reproducible Software Workflow for Compact Pauli Analysis of Molecular Vibrational Hamiltonians

This package keeps the uploaded main manuscript as the primary paper and adds the quantum-resource / protein-mode viewer material to that paper rather than creating a separate protein manuscript.

Files:
- Pauli_Workflow_Paper1_REVISED.pdf
- Pauli_Workflow_Paper1_Supplement_REVISED.pdf
- main_addition.tex
- supplement_addition.tex
- main_revised_wrapper.tex
- supplement_revised_wrapper.tex
- fig_quantum_viewer_architecture.pdf/png
- original_main.pdf
- original_supplement.pdf

Important:
Only PDF sources were supplied for the original manuscript. Therefore the compiled revised PDFs preserve the original PDF pages and append insert-ready LaTeX additions. The new LaTeX is fully editable. When the native original .tex source is available, main_addition.tex should be merged directly into that source (ideally near the interactive viewer/software discussion) and the supplementary addition should be merged into the native supplementary .tex.

Scientific guardrails added:
- compact qubit count is an index-register statement;
- a 64-dimensional reduced/modal block maps to six qubits, not an entire protein;
- Pauli interaction connectivity is called a qubit-coupling graph, not entanglement;
- true entanglement requires state-derived reduced-density-matrix metrics;
- runtime panels report classical stage timings and do not claim quantum speedup;
- the existing 1CRN reduced-space result remains a scaling stress test, not a replacement for full protein NMA.
