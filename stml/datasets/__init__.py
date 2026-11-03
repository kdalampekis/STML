"""Converters from public human-learning datasets and our own game into one format.

common     the stml-traj/1 schema, validation, and the domain-free learning-event view
harc       H-ARC (ARC puzzles, 1,729 people, attempts with right/wrong feedback)
stml_game  Wugs and Daxes logs

Planned: ARCTraj (Hugging Face) and Sudoku-Bench CTC solves, once the files are
in hand: their exact field layout has to be read from the data, not guessed.
"""
