/**
 * Unit tests for BILO and PINN TS implementation.
 * - Forward and loss tests (boundary, loss decrease, finite gradients, evalUArray).
 * - Snapshot export: run forward + backward, save weights and gradients to JSON
 *   for verification against PyTorch (run: python playground/bilo_np/verify_ts_gradients.py).
 */

import { BILOModel, PINNModel } from "./bilo_nn";

const DEFAULT_DEPTH = 2;

function applyGradients(model: BILOModel, grads: Record<string, number[] | number[][] | number>, lr: number): void {
  for (let k = 0; k < model.depth; k++) {
    const Wk = model._W[k];
    const gWk = grads[`W${k + 1}`];
    if (Array.isArray(Wk[0])) {
      const W = Wk as number[][];
      const g = gWk as number[][];
      for (let i = 0; i < W.length; i++)
        for (let j = 0; j < W[i].length; j++) W[i][j] -= lr * g[i][j];
    } else {
      const W = Wk as number[];
      const g = gWk as number[];
      for (let i = 0; i < W.length; i++) W[i] -= lr * g[i];
    }
    const bk = model._b[k];
    const gbk = grads[`b${k + 1}`];
    if (typeof bk === "number") {
      (model._b[k] as number) = bk - lr * (gbk as number);
    } else {
      const b = bk as number[];
      const g = gbk as number[];
      for (let i = 0; i < b.length; i++) b[i] -= lr * g[i];
    }
  }
}

export function runBiloTests(): { pass: boolean; messages: string[] } {
  const messages: string[] = [];
  let pass = true;

  // Test 1: Forward pass u(0,a) = 1 (boundary)
  {
    const model = new BILOModel(4, DEFAULT_DEPTH, 42);
    for (const a of [0.5, 1, 2]) {
      const u = model.evalU(0, a);
      if (Math.abs(u - 1) > 1e-10) {
        pass = false;
        messages.push(`FAIL: u(0,${a}) = ${u}, expected 1`);
      }
    }
    if (pass) messages.push("PASS: u(0,a) = 1 (boundary)");
  }

  // Test 2: Loss decreases over training (small lr, enough steps)
  {
    const model = new BILOModel(4, DEFAULT_DEPTH, 42);
    const t = [0.25, 0.5, 0.75];
    const a = [1, 1, 1];
    const { losses: l0 } = model.computeLossesAndGradients(t, a, { w_res: 1, w_grad: 0.1 });
    for (let i = 0; i < 150; i++) {
      const { losses, grads } = model.computeLossesAndGradients(t, a, { w_res: 1, w_grad: 0.1 });
      applyGradients(model, grads, 0.01);
    }
    const { losses: l1 } = model.computeLossesAndGradients(t, a, { w_res: 1, w_grad: 0.1 });
    if (l1.L_res + l1.L_grad >= l0.L_res + l0.L_grad) {
      pass = false;
      messages.push(`FAIL: loss did not decrease: ${l0.L_res + l0.L_grad} -> ${l1.L_res + l1.L_grad}`);
    } else {
      messages.push("PASS: loss decreases over training");
    }
  }

  // Test 3: Gradients are finite
  {
    const model = new BILOModel(4, DEFAULT_DEPTH, 42);
    const { grads } = model.computeLossesAndGradients([0.5], [1], { w_res: 1, w_grad: 0.1 });
    const allFinite = (arr: number[] | number[][] | number): boolean => {
      if (typeof arr === "number") return isFinite(arr);
      if (Array.isArray(arr[0])) return (arr as number[][]).every(row => row.every(x => isFinite(x)));
      return (arr as number[]).every(x => isFinite(x));
    };
    const ok = Object.keys(grads).filter(k => k !== "a").every(k => allFinite(grads[k] as number[] | number[][] | number));
    if (!ok) {
      pass = false;
      messages.push("FAIL: gradients contain NaN/Inf");
    } else {
      messages.push("PASS: gradients are finite");
    }
  }

  // Test 4: evalUArray matches evalU
  {
    const model = new BILOModel(4, DEFAULT_DEPTH, 42);
    const tArr = [0, 0.5, 1];
    const a = 1;
    const arr = model.evalUArray(tArr, a);
    const single = tArr.map(t => model.evalU(t, a));
    const match = arr.every((v, i) => Math.abs(v - single[i]) < 1e-12);
    if (!match) {
      pass = false;
      messages.push("FAIL: evalUArray != map evalU");
    } else {
      messages.push("PASS: evalUArray consistency");
    }
  }

  // Test 5: Depth 3 (two hidden layers) forward and gradients
  {
    const model = new BILOModel(4, 3, 43);
    const f = model.forward(0.5, 1);
    if (!isFinite(f.N) || !isFinite(f.u)) {
      pass = false;
      messages.push("FAIL: depth=3 forward produced non-finite");
    }
    const { losses, grads } = model.computeLossesAndGradients([0.5], [1], { w_res: 1, w_grad: 0.1 });
    const hasW1 = Array.isArray(grads.W1) && Array.isArray((grads.W1 as number[][])[0]);
    const hasW2 = Array.isArray(grads.W2);
    const hasW3 = Array.isArray(grads.W3) && !Array.isArray((grads.W3 as number[])[0]);
    if (!hasW1 || !hasW2 || !hasW3) {
      pass = false;
      messages.push("FAIL: depth=3 grads missing W1/W2/W3");
    } else {
      messages.push("PASS: depth=3 forward and grads");
    }
  }

  return { pass, messages };
}

// ---- PINN tests ----

export function runPinnTests(): { pass: boolean; messages: string[] } {
  const messages: string[] = [];
  let pass = true;

  {
    const model = new PINNModel(4, DEFAULT_DEPTH, 42);
    for (const a of [0, 0.5, 1, 2]) {
      const u0 = model.evalU(0.5, a);
      const u1 = model.evalU(0.5, 1);
      if (Math.abs(u0 - u1) > 1e-10) {
        pass = false;
        messages.push(`FAIL: PINN u(0.5,${a}) = ${u0} != u(0.5,1) = ${u1}`);
      }
    }
    if (pass) messages.push("PASS: PINN u(t) independent of a");
  }

  {
    const model = new PINNModel(4, DEFAULT_DEPTH, 42);
    const t_colloc = [0.25, 0.5, 0.75];
    const a_colloc = [1, 1, 1];
    const { losses: l0 } = model.computeLossesAndGradientsPinn(t_colloc, a_colloc, { w_res: 1 });
    for (let i = 0; i < 100; i++) {
      const { losses, grads } = model.computeLossesAndGradientsPinn(t_colloc, a_colloc, { w_res: 1 });
      applyGradients(model, grads, 0.01);
    }
    const { losses: l1 } = model.computeLossesAndGradientsPinn(t_colloc, a_colloc, { w_res: 1 });
    if (l1.L_res >= l0.L_res) {
      pass = false;
      messages.push(`FAIL: PINN L_res did not decrease: ${l0.L_res} -> ${l1.L_res}`);
    } else {
      messages.push("PASS: PINN L_res decreases");
    }
  }

  {
    const model = new PINNModel(4, DEFAULT_DEPTH, 43);
    const t_colloc = [0.2, 0.5, 0.8];
    const a_colloc = [1.2, 1.2, 1.2];
    const t_data = [0.3, 0.6];
    const u_data = [1.5, 2.2];
    const { losses, grads } = model.computeLossesAndGradientsPinn(t_colloc, a_colloc, {
      t_data, u_data, w_res: 1, w_data: 0.5,
    });
    const W1 = model._W[0] as number[][];
    const gW1 = grads.W1 as number[][];
    for (let j = 0; j < W1.length; j++) {
      if (Math.abs(gW1[j][1]) > 1e-12) {
        pass = false;
        messages.push(`FAIL: PINN grad W1[:,1] should be 0, got ${gW1[j][1]}`);
      }
    }
    if (typeof grads.a !== "number" || !isFinite(grads.a)) {
      pass = false;
      messages.push("FAIL: PINN grads.a missing or non-finite");
    }
    if (pass) messages.push("PASS: PINN grads (W1 col1=0, a present)");
  }

  return { pass, messages };
}

/** Snapshot shape for Python verification: weights, inputs, losses, gradients. */
export interface BiloSnapshot {
  model_type?: "bilo" | "pinn";
  n_hidden: number;
  depth: number;
  seed: number;
  /** Weights after init (or after some step); same layout as Python _W, _b */
  W: (number[][] | number[])[];
  b: (number[] | number)[];
  t_colloc: number[];
  a_colloc: number[];
  t_data?: number[];
  a_data?: number[];
  u_data?: number[];
  w_res: number;
  w_grad: number;
  w_data: number;
  losses: { L_res: number; L_grad: number; L_data: number };
  grads: Record<string, number[] | number[][] | number>;
}

/**
 * Build a snapshot of the current model and gradients for a given scenario.
 * Write the result to a JSON file (e.g. ts_snapshot.json) so that
 * verify_ts_gradients.py can load it and compare with PyTorch.
 */
export function buildSnapshotForVerification(opts: {
  n_hidden: number;
  depth: number;
  seed: number;
  t_colloc: number[];
  a_colloc: number[];
  t_data?: number[];
  a_data?: number[];
  u_data?: number[];
  w_res?: number;
  w_grad?: number;
  w_data?: number;
}): BiloSnapshot {
  const {
    n_hidden,
    depth,
    seed,
    t_colloc,
    a_colloc,
    t_data,
    a_data,
    u_data,
    w_res = 1,
    w_grad = 0.1,
    w_data = 0,
  } = opts;
  const model = new BILOModel(n_hidden, depth, seed);
  const { losses, grads } = model.computeLossesAndGradients(t_colloc, a_colloc, {
    t_data,
    a_data,
    u_data,
    w_res,
    w_grad,
    w_data,
  });
  const W = model._W.map(w =>
    Array.isArray(w[0]) ? (w as number[][]).map(row => [...row]) : [...(w as number[])]
  ) as (number[][] | number[])[];
  const b = model._b.map(bb => (typeof bb === "number" ? bb : [...bb]));
  const gradCopy: Record<string, number[] | number[][] | number> = {};
  for (const key of Object.keys(grads)) {
    const g = grads[key];
    if (typeof g === "number") gradCopy[key] = g;
    else if (Array.isArray(g[0])) gradCopy[key] = (g as number[][]).map(row => [...row]);
    else gradCopy[key] = [...(g as number[])];
  }
  return {
    model_type: "bilo",
    n_hidden,
    depth,
    seed,
    W,
    b,
    t_colloc: [...t_colloc],
    a_colloc: [...a_colloc],
    t_data: t_data ? [...t_data] : undefined,
    a_data: a_data ? [...a_data] : undefined,
    u_data: u_data ? [...u_data] : undefined,
    w_res,
    w_grad,
    w_data,
    losses: { ...losses },
    grads: gradCopy,
  };
}

/**
 * Build PINN snapshot for Python verification. Uses different number of
 * residual collocation points vs data points (to stress-test the verifier).
 */
export function buildSnapshotForVerificationPinn(opts: {
  n_hidden: number;
  depth: number;
  seed: number;
  t_colloc: number[];
  a_colloc: number[];
  t_data: number[];
  u_data: number[];
  w_res?: number;
  w_data?: number;
}): BiloSnapshot {
  const {
    n_hidden,
    depth,
    seed,
    t_colloc,
    a_colloc,
    t_data,
    u_data,
    w_res = 1,
    w_data = 0.5,
  } = opts;
  const model = new PINNModel(n_hidden, depth, seed);
  const { losses, grads } = model.computeLossesAndGradientsPinn(t_colloc, a_colloc, {
    t_data,
    u_data,
    w_res,
    w_data,
  });
  const W = model._W.map(w =>
    Array.isArray(w[0]) ? (w as number[][]).map(row => [...row]) : [...(w as number[])]
  ) as (number[][] | number[])[];
  const b = model._b.map(bb => (typeof bb === "number" ? bb : [...bb]));
  const gradCopy: Record<string, number[] | number[][] | number> = {};
  for (const key of Object.keys(grads)) {
    const g = grads[key];
    if (typeof g === "number") gradCopy[key] = g;
    else if (Array.isArray(g[0])) gradCopy[key] = (g as number[][]).map(row => [...row]);
    else gradCopy[key] = [...(g as number[])];
  }
  return {
    model_type: "pinn",
    n_hidden,
    depth,
    seed,
    W,
    b,
    t_colloc: [...t_colloc],
    a_colloc: [...a_colloc],
    t_data: [...t_data],
    a_data: a_colloc.length >= t_data.length ? a_colloc.slice(0, t_data.length) : undefined,
    u_data: [...u_data],
    w_res,
    w_grad: 0,
    w_data,
    losses: { ...losses },
    grads: gradCopy,
  };
}
