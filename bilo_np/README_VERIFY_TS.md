# Verifying TypeScript BILO gradients against PyTorch

## One command (recommended)

From **playground** or repo root:

```bash
./bilo_np/verify_ts_gradients.sh
```

This (1) builds the TS snapshot bundle and writes `bilo_np/ts_snapshot.json`, then (2) runs the Python verifier (NumPy + PyTorch). Uses `conda activate math10` for PyTorch if conda is available.

## Step by step

**1. Write snapshot from TypeScript** (from `playground`):

```bash
npm run write-bilo-snapshot
```

**2. Run the Python verifier** (from `playground` or `bilo_np`):

```bash
cd bilo_np
conda activate math10   # if you use it for PyTorch
python verify_ts_gradients.py [path/to/ts_snapshot.json]
```

Default path is `ts_snapshot.json` in the current directory. The verifier compares TS snapshot vs NumPy and (if PyTorch is available) vs PyTorch.

## Snapshot from NumPy only (no TypeScript)

To test the verifier or compare NumPy vs PyTorch only:

```bash
cd bilo_np
python generate_snapshot.py
python verify_ts_gradients.py
```
