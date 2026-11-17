# Open-system run record

These two files are the unedited log and result of one of the paper's direct open-system runs:
the 40 ns CZ on the full 4Q4C ring (`D = 6561`, 256 dressed computational matrix units), computed
with the fp32 build and the custom RK4 (12000 steps), sharded over 7 GPUs with checkpointing.

| File | Contents |
|---|---|
| `open_cz_4q4c_T40_G50_RESULTS.txt` | full run log: GPUs, exact command line, per-rank binding, timing, merge |
| `open_cz_4q4c_T40_G50.txt` | result: F_closed = 0.99903, F_open = 0.99551, loss 3.519e-3 |

**Label note.** In these logs, `T2` means **Tφ**, the pure-dephasing time in `L_dep = √(2/Tφ)·n`.
This run is therefore T1 = Tφ = 50 µs, as in the paper's Appendix F (total T2 = 33.3 µs). The
result matches the paper: F_closed is Table I's 40 ns CZ, and F_open and the loss are Appendix F's
50 µs values.

The `/nfs_scratch/...` paths and hostnames are from the cluster where the run was made; the files
are kept exactly as written.
