# MibS Input Format Notes (for this build)

These notes match your local MibS build at `/home/ole/mibs_build/dist/bin/mibs` (Build Date Feb 27, 2026).

## 1) Required files

For deterministic bilevel instances, MibS needs:

1. An **MPS** model file (all vars/constraints/objective)
2. An **aux/txt** file that marks which variables/constraints are lower-level

Run pattern:

```bash
mibs -Alps_instance problem.mps -MibS_auxiliaryInfoFile problem.txt
```

## 2) MPS requirements

- Standard MPS sections: `NAME`, `ROWS`, `COLUMNS`, `RHS`, `BOUNDS`, `ENDATA`
- Put integer vars between `INTORG` / `INTEND` markers
- Keep row and column names stable (aux references them by index or name)
- In this build, avoid `E` rows when possible (use paired `L` and `G` constraints)

## 3) AUX/TXT formats accepted by your build

Your `MibSModel.cpp` parser supports the classic format with keywords:

- `N` = number of lower-level variables
- `M` = number of lower-level constraints
- `LC` = lower-level variable index (or name if `MibS_inputFormat` is `nameBased`)
- `LR` = lower-level row index (or name if `MibS_inputFormat` is `nameBased`)
- `LO` = lower-level objective coefficient for each LL variable (same order as `LC`)
- `OS` = LL objective sense (`1` min, `-1` max)

Default is `MibS_inputFormat = indexBased`.

### Important indexing rule

Indices are **0-based** in the order variables/rows appear in the MPS after parsing.

## 4) Minimal robust template (index-based)

```txt
N <num_ll_vars>
M <num_ll_rows>
LC <col_idx_1>
LC <col_idx_2>
...
LR <row_idx_1>
LR <row_idx_2>
...
LO <coef_for_lc_1>
LO <coef_for_lc_2>
...
OS 1
```

Counts must match exactly:
- number of `LC` lines = `N`
- number of `LR` lines = `M`
- number of `LO` lines = `N`

## 5) Name-section variant in your examples

Your shipped example `moore90WithNameSection.txt` uses:

```txt
N 1
M 4
OS 1
@VARSBEGIN
LV 1
@CONSTSBEGIN
R1
R2
R3
R4
```

So this build also recognizes `@VARSBEGIN` and `@CONSTSBEGIN`, but still depends on `N` and `M`.

## 6) Common reasons generated instances fail

1. Wrong LL partition (e.g., budget row accidentally included as LL row)
2. `N/M` not consistent with actual listed vars/rows
3. Wrong index order (not matching parsed MPS order)
4. Equality rows (`E`) in formulations where this build rejects them
5. Path issues under WSL when Windows path has spaces

## 7) Working references in your workspace

- Official readme pointer: `mibs_build/MibS/README.md` (links to `https://coin-or.github.io/MibS/input.html`)
- Parser implementation: `mibs_build/MibS/src/MibSModel.cpp` (`readAuxiliaryData`)
- Working sample pair:
  - `mibs_build/MibS/data/moore90WithNameSection.mps`
  - `mibs_build/MibS/data/moore90WithNameSection.txt`

