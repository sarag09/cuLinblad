"""
Merge per-rank MPI shard dumps (propcheck/<tag>_rank<r>.txt) into one dump file, reconstructing
the single-process global state order. Each rank wrote its contiguous shard with STATE numbered
LOCALLY 0..cnt-1; since the shard is contiguous and rank-ordered (rank 0 = global positions
[0,cnt0), rank 1 = [cnt0,...), ...), concatenating the rank files IN RANK ORDER and renumbering
STATE to 0..N-1 reproduces the single-process ordering -> a bit-identical comparison target.

CORRECTNESS GATES (so a missing/stale/truncated shard cannot silently yield a wrong-but-plausible
merged anchor -- critical for the production 'run' mode that has no golden compare):
  * rank files must form the contiguous set {0..k-1} (a gap = a dropped middle shard)
  * if expected_nranks is given, exactly that many files must be present (catches a dropped LAST shard)
  * every rank file must contain its own BEGIN and END (a truncated/crashed shard is rejected)
  * all rank BEGIN headers must agree on everything except the per-rank batch_size/num_states
  * all STATE blocks must have the same entry count, == the header density_dim/state_len
  * if expected_N is given, the merged state count must equal it
The merged BEGIN/END have batch_size + num_states rewritten to the total so golden_regression's
(backend,batch_size) block key matches the single-process golden.

Usage:  python merge_ranks.py <tag_prefix> <merged_out.txt> [expected_nranks] [expected_N]
  e.g.  python merge_ranks.py propcheck/tc4q4c_shard merged.txt 2 256
"""
import sys
import re
import glob

tag = sys.argv[1]
out_path = sys.argv[2]
expected_nranks = int(sys.argv[3]) if len(sys.argv) > 3 else None
expected_N = int(sys.argv[4]) if len(sys.argv) > 4 else None


def die(msg):
    sys.exit("merge_ranks: " + msg)


def rankof(p):
    return int(re.search(r"_rank(\d+)\.txt$", p).group(1))


def parse_kv(line):
    d = {}
    for tok in line.split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


files = glob.glob(tag + "_rank*.txt")
if not files:
    die("no files match %s_rank*.txt" % tag)
files.sort(key=rankof)
ranks = [rankof(p) for p in files]
if ranks != list(range(len(ranks))):
    die("non-contiguous rank files (have %s; need 0..%d) -- a shard is missing" % (ranks, len(ranks) - 1))
if expected_nranks is not None and len(ranks) != expected_nranks:
    die("expected %d rank files, found %d %s -- a shard is missing/stale" % (expected_nranks, len(ranks), ranks))

begin_line = None
end_line = None
ref_hdr = None
blocks = []  # [entry lines], in global (rank-then-local) order

for fp in files:
    saw_begin = saw_end = False
    cur = None
    for raw in open(fp):
        line = raw.rstrip("\n")
        if line.startswith("PROPAGATOR_DUMP_BEGIN"):
            saw_begin = True
            hdr = parse_kv(line)
            cmp_hdr = {k: v for k, v in hdr.items() if k not in ("batch_size", "num_states")}
            if ref_hdr is None:
                ref_hdr = cmp_hdr
                begin_line = line
            elif cmp_hdr != ref_hdr:
                die("rank %d BEGIN header mismatch vs rank 0: %s != %s" % (rankof(fp), cmp_hdr, ref_hdr))
        elif line.startswith("PROPAGATOR_DUMP_END"):
            saw_end = True
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
    if not (saw_begin and saw_end):
        die("rank %d file truncated (missing BEGIN/END) -- shard crashed mid-dump" % rankof(fp))

if begin_line is None or end_line is None:
    die("malformed rank files (missing BEGIN/END)")

counts = set(len(b) for b in blocks)
if len(counts) != 1:
    die("inconsistent state lengths across blocks: %s -- a shard is truncated" % sorted(counts))
blen = counts.pop()
hdr_len = ref_hdr.get("density_dim") or ref_hdr.get("state_len")
if hdr_len is not None and int(hdr_len) != blen:
    die("block entry count %d != header density_dim/state_len %s" % (blen, hdr_len))

total = len(blocks)
if expected_N is not None and total != expected_N:
    die("merged %d states but expected %d -- shard count wrong" % (total, expected_N))


def setkv(s, k, v):
    return re.sub(k + r"=\S+", "%s=%d" % (k, v), s)


merged_begin = setkv(setkv(begin_line, "batch_size", total), "num_states", total)
merged_end = setkv(end_line, "batch_size", total)

with open(out_path, "w") as f:
    f.write(merged_begin + "\n")
    for i, entries in enumerate(blocks):
        f.write("STATE %d\n" % i)
        f.write("\n".join(entries) + "\n")
    f.write(merged_end + "\n")

print("merge_ranks: %d rank files -> %d states (len %d) -> %s" % (len(files), total, blen, out_path))
