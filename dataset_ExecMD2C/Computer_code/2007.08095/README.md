# Synthesize, Execute and Debug: Learning to Repair for Neural Program Synthesis (2007.08095)

## Project source

- **Project ID:** `2007.08095`
- **Paper:** Synthesize, Execute and Debug: Learning to Repair for Neural Program Synthesis
- **PDF:** https://arxiv.org/pdf/2007.08095v2.pdf
- **GitHub:** https://github.com/sunblaze-ucb/SED

## Reproduction task

The system repairs synthesized programs by predicting and applying code edits after execution feedback. In `template.py`, complete `compute_edit_ops`, `DiversitySearch.diverse_decision`, and the other TODO-marked functions to convert a source token sequence into an edit script for the target token sequence and break ties by choosing the candidate with the rarest semantic pass/fail pattern.
