"""
Merge per-rank MPI shard dumps (propcheck/<tag>_rank<r>.txt) into one dump file, reconstructing
the single-process global state order. Each rank wrote its contiguous shard with STATE numbered
LOCALLY 0..cnt-1; since the shard is contiguous and rank-ordered (rank 0 = global positions
[0,cnt0), rank 1 = [cnt0,...), ...), concatenating the rank files IN RANK ORDER and renumbering
STATE to 0..N-1 reproduces the single-process ordering -> a bit-identical comparison target.
The merged PROPAGATOR_DUMP_BEGIN has batch_size and num_states rewritten to the total so
golden_regression's (backend, batch_size) block key matches the single-process golden.

Usage:  python merge_ranks.py <tag_prefix> <merged_out.txt>
  e.g.  python merge_ranks.py propcheck/tc4q4c_shard merged.txt
"""
import sys
import re
import glob

tag = sys.argv[1]
out_path = sys.argv[2]

files = glob.glob(tag + "_rank*.txt")
if not files:
    sys.exit("merge_ranks: no files match %s_rank*.txt" % tag)
files.sort(key=lambda p: int(re.search(r"_rank(\d+)\.txt$", p).group(1)))

begin_line = None
end_line = None
blocks = []  # list of [entry lines], in global (rank-then-local) order

for fp in files:
    cur = None
    for raw in open(fp):
        line = raw.rstrip("\n")
        if line.startswith("PROPAGATOR_DUMP_BEGIN"):
            if begin_line is None:
                begin_line = line
        elif line.startswith("PROPAGATOR_DUMP_END"):
            if end_line is None:
                end_line = line
            if cur is not None:
                blocks.append(cur)
                cur = None
        elif line.startswith("STATE"):
            if cur is not None:
                blocks.append(cur)
            cur = []
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        blocks.append(cur)

if begin_line is None or end_line is None:
    sys.exit("merge_ranks: malformed rank files (missing BEGIN/END)")

total = len(blocks)
def setkv(s, k, v):
    return re.sub(k + r"=\S+", "%s=%d" % (k, v), s)
merged_begin = setkv(setkv(begin_line, "batch_size", total), "num_states", total)

with open(out_path, "w") as f:
    f.write(merged_begin + "\n")
    for i, entries in enumerate(blocks):
        f.write("STATE %d\n" % i)
        f.write("\n".join(entries) + "\n")
    f.write(end_line + "\n")

print("merge_ranks: %d rank files -> %d states -> %s" % (len(files), total, out_path))
