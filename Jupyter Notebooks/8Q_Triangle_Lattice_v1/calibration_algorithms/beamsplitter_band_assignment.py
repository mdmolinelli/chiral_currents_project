# qubit_band_assignment.py
# Author: ChatGPT (GPT-5)
# Purpose: Assign low/mid/high frequency bands to qubits or qubit pairs
#          in a superconducting-qubit chain, respecting coupling constraints.

from itertools import product
from collections import defaultdict

# ------------------------------------------------------------
# Helper definitions
# ------------------------------------------------------------

BANDS = ["Low", "Mid", "High"]
BAND_DIST = {
    ("Low", "Low"): 0, ("Mid", "Mid"): 0, ("High", "High"): 0,
    ("Low", "Mid"): 1, ("Mid", "Low"): 1,
    ("Mid", "High"): 1, ("High", "Mid"): 1,
    ("Low", "High"): 2, ("High", "Low"): 2
}

# ------------------------------------------------------------
# Build block structure
# ------------------------------------------------------------

def build_blocks(pairs, singles):
    """
    Convert pairs and singles into block dict.
    Returns:
        blocks: dict {block_label: [qubit_indices]}
        qubit_to_block: reverse map
    """
    blocks = {}
    qubit_to_block = {}
    label_id = 0

    for p in pairs:
        label = f"P{label_id}"
        blocks[label] = list(p)
        for q in p:
            qubit_to_block[q] = label
        label_id += 1

    for s in singles:
        label = f"S{label_id}"
        blocks[label] = [s]
        qubit_to_block[s] = label
        label_id += 1

    return blocks, qubit_to_block

# ------------------------------------------------------------
# Coupling graph builder
# ------------------------------------------------------------

def build_edges(blocks, qubit_to_block, n_qubits=8):
    """
    Construct weighted edges between blocks using:
      - nearest neighbor (i, i+1) -> weight 1
      - next-nearest neighbor (i, i+2) -> weight 2
    Returns: dict {(blockA, blockB): weight}
    """
    edges = defaultdict(int)
    def add_edge(q1, q2, weight):
        b1, b2 = qubit_to_block[q1], qubit_to_block[q2]
        if b1 == b2:
            return
        key = tuple(sorted([b1, b2]))
        edges[key] += weight

    for i in range(1, n_qubits):
        add_edge(i, i + 1, 1)  # nearest
    for i in range(1, n_qubits - 1):
        add_edge(i, i + 2, 2)  # next-nearest

    return edges

# ------------------------------------------------------------
# Scoring function
# ------------------------------------------------------------

def score_assignment(assign, edges):
    total = 0
    for (a, b), w in edges.items():
        total += w * BAND_DIST[(assign[a], assign[b])]
    return total

# ------------------------------------------------------------
# Enumerate all valid assignments
# ------------------------------------------------------------

def optimize_single(blocks, edges, constraints=None, prefer_high=True):
    """
    Brute-force search for best assignment.
    constraints: dict {block: {"no_high": bool}}
    prefer_high: if True, tie-breaker prefers pairs in High band
    """
    if constraints is None:
        constraints = {}

    best_score = -1e9
    best_assign = None

    labels = list(blocks.keys())

    for combo in product(BANDS, repeat=len(labels)):
        assign = dict(zip(labels, combo))
        # Apply constraints
        violated = False
        for b in labels:
            if constraints.get(b, {}).get("no_high") and assign[b] == "High":
                violated = True
                break
        if violated:
            continue

        s = score_assignment(assign, edges)
        if s > best_score:
            best_score, best_assign = s, assign
        elif s == best_score and prefer_high:
            # tie-break: count how many blocks are High
            high_count_new = sum(1 for v in assign.values() if v == "High")
            high_count_old = sum(1 for v in best_assign.values() if v == "High")
            if high_count_new > high_count_old:
                best_assign = assign

    return best_assign, best_score

# ------------------------------------------------------------
# Conflict detection and splitting heuristic
# ------------------------------------------------------------

def find_conflicts(assign, edges, weight_threshold=2):
    conflicts = []
    for (a, b), w in edges.items():
        if w >= weight_threshold:
            dist = BAND_DIST[(assign[a], assign[b])]
            if dist < 2:
                conflicts.append((a, b, w))
    return conflicts

def greedy_split(conflicts, blocks):
    """
    Simple greedy 2-run split of blocks based on conflict edges.
    Returns two sets of block labels (runA, runB).
    """
    runA, runB = set(), set()
    for a, b, w in sorted(conflicts, key=lambda x: -x[2]):
        if a in runA:
            runB.add(b)
        elif a in runB:
            runA.add(b)
        elif b in runA:
            runB.add(a)
        elif b in runB:
            runA.add(a)
        else:
            runA.add(a)
            runB.add(b)
    # assign unassigned blocks arbitrarily to runA
    all_blocks = set(blocks.keys())
    unassigned = all_blocks - runA - runB
    for u in unassigned:
        runA.add(u)
    return runA, runB

# ------------------------------------------------------------
# Main algorithm
# ------------------------------------------------------------

def assign_bands(pairs, singles, n_qubits=8):
    blocks, q2b = build_blocks(pairs, singles)
    edges = build_edges(blocks, q2b, n_qubits=n_qubits)

    # Constraint: any block containing qubit 5 cannot be High
    constraints = {}
    for b, qs in blocks.items():
        if 5 in qs:
            constraints[b] = {"no_high": True}

    # Optimize single config
    best_assign, best_score = optimize_single(blocks, edges, constraints)

    # Check for heavy-edge conflicts
    conflicts = find_conflicts(best_assign, edges)
    if not conflicts:
        print("✅ Single configuration sufficient")
        return [best_assign]

    # Otherwise split into 2 runs
    print("⚠️  Conflicts detected, splitting into two configurations...")
    runA, runB = greedy_split(conflicts, blocks)

    # Optimize each run separately
    assignA, _ = optimize_single({b: blocks[b] for b in runA}, edges, constraints)
    assignB, _ = optimize_single({b: blocks[b] for b in runB}, edges, constraints)

    # Fill missing blocks as Low (off)
    for b in blocks:
        assignA.setdefault(b, "Low")
        assignB.setdefault(b, "Low")

    return [assignA, assignB]

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

if __name__ == "__main__":
    # Example 1
    pairs = [(1,2),(3,4),(5,6),(7,8)]
    singles = []
    configs = assign_bands(pairs, singles)
    for i, cfg in enumerate(configs):
        print(f"\nConfiguration {i+1}:")
        for b, band in cfg.items():
            print(f"  {b}: {band}")

    # Example 2
    print("\n" + "="*40)
    pairs = [(1,2),(4,5),(6,7)]
    singles = [3,8]
    configs = assign_bands(pairs, singles)
    for i, cfg in enumerate(configs):
        print(f"\nConfiguration {i+1}:")
        for b, band in cfg.items():
            print(f"  {b}: {band}")
