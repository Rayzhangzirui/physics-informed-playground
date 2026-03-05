/**
 * BILO/PINN TS tests: gradient verification via snapshots.
 *
 * - runBiloTests / runPinnTests: minimal sanity (boundary u(0)=u0, PINN u independent of a).
 * - getVerificationSnapshots(): returns snapshots for each scenario; verify against PyTorch
 *   by writing JSON (npm run write-ts-snapshots) then: pytest bilo_np/test_verify_ts_gradients.py
 *
 * Snapshot scenarios mirror playground/bilo_np/test_gradients.py and test_pinn_gradients.py.
 */

import { BILOModel, PINNModel, type OdeType } from "./bilo_nn";

const DEFAULT_DEPTH = 2;

export function runBiloTests(): { pass: boolean; messages: string[] } {
  const messages: string[] = [];
  let pass = true;

  // Boundary: u(0, a) = u0 for exponential (1) and logistic (0.1)
  {
    const expModel = new BILOModel(4, DEFAULT_DEPTH, 42, "exponential");
    const logModel = new BILOModel(4, DEFAULT_DEPTH, 42, "logistic");
    for (const a of [0.5, 1, 2]) {
      if (Math.abs(expModel.evalU(0, a) - 1) > 1e-10) {
        pass = false;
        messages.push(`FAIL: exponential u(0,${a}) != 1`);
      }
      if (Math.abs(logModel.evalU(0, a) - 0.1) > 1e-10) {
        pass = false;
        messages.push(`FAIL: logistic u(0,${a}) != 0.1`);
      }
    }
    if (pass) messages.push("PASS: boundary u(0,a)=u0");
  }

  return { pass, messages };
}

// ---- PINN tests ----

export function runPinnTests(): { pass: boolean; messages: string[] } {
  const messages: string[] = [];
  let pass = true;

  const model = new PINNModel(4, DEFAULT_DEPTH, 42, "exponential");
  for (const a of [0, 0.5, 1, 2]) {
    const u0 = model.evalU(0.5, a);
    const u1 = model.evalU(0.5, 1);
    if (Math.abs(u0 - u1) > 1e-10) {
      pass = false;
      messages.push(`FAIL: PINN u(0.5,${a}) != u(0.5,1)`);
    }
  }
  if (pass) messages.push("PASS: PINN u(t) independent of a");
  return { pass, messages };
}

/** Snapshot shape for Python verification: weights, inputs, losses, gradients. */
export interface BiloSnapshot {
  model_type?: "bilo" | "pinn";
  n_hidden: number;
  depth: number;
  seed: number;
  ode_type?: OdeType;
  u0?: number;
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
 * Used by verify_ts_gradients.py to compare TS gradients with PyTorch autograd.
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
  ode_type?: OdeType;
  u0?: number;
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
    ode_type = "exponential",
    u0,
  } = opts;
  const model = new BILOModel(n_hidden, depth, seed, ode_type, u0);
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
  const out: BiloSnapshot = {
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
  out.ode_type = ode_type;
  out.u0 = model.u0;
  return out;
}

/**
 * Build PINN snapshot for Python verification.
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
  ode_type?: OdeType;
  u0?: number;
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
    ode_type = "exponential",
    u0,
  } = opts;
  const model = new PINNModel(n_hidden, depth, seed, ode_type, u0);
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
  const out: BiloSnapshot = {
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
  out.ode_type = ode_type;
  out.u0 = model.u0;
  return out;
}

/**
 * All snapshot scenarios for gradient verification (mirror test_gradients.py and test_pinn_gradients.py).
 * Write to JSON with npm run write-ts-snapshots, then run: pytest bilo_np/test_verify_ts_gradients.py
 */
export function getVerificationSnapshots(): { name: string; snapshot: BiloSnapshot }[] {
  const n_hidden = 4;
  const list: { name: string; snapshot: BiloSnapshot }[] = [];

  // BILO: L_res + L_grad only (one collocation point; multiple depths and ode_types)
  for (const ode_type of ["exponential", "logistic"] as OdeType[]) {
    for (const depth of [2, 3]) {
      list.push({
        name: `bilo_residual_${ode_type}_depth${depth}`,
        snapshot: buildSnapshotForVerification({
          n_hidden,
          depth,
          seed: 42,
          t_colloc: [0.5],
          a_colloc: [1.0],
          w_res: 1,
          w_grad: 1,
          ode_type,
        }),
      });
    }
  }

  // BILO: L_data only (no collocation; gradient w.r.t. W and a)
  for (const ode_type of ["exponential", "logistic"] as OdeType[]) {
    list.push({
      name: `bilo_data_${ode_type}_depth2`,
      snapshot: buildSnapshotForVerification({
        n_hidden,
        depth: 2,
        seed: 43,
        t_colloc: [],
        a_colloc: [],
        t_data: [0.3],
        a_data: [1.2],
        u_data: [1.5],
        w_res: 0,
        w_grad: 0,
        w_data: 1,
        ode_type,
      }),
    });
  }

  // BILO: combined L_res + L_grad + L_data
  for (const ode_type of ["exponential", "logistic"] as OdeType[]) {
    list.push({
      name: `bilo_combined_${ode_type}_depth2`,
      snapshot: buildSnapshotForVerification({
        n_hidden,
        depth: 2,
        seed: 44,
        t_colloc: [0.25, 0.5, 0.75],
        a_colloc: [1, 1, 1],
        t_data: [0.5],
        a_data: [1],
        u_data: [1.65],
        w_res: 1,
        w_grad: 0.1,
        w_data: 0.5,
        ode_type,
      }),
    });
  }

  // PINN: L_res only
  for (const ode_type of ["exponential", "logistic"] as OdeType[]) {
    for (const depth of [2, 3]) {
      list.push({
        name: `pinn_residual_${ode_type}_depth${depth}`,
        snapshot: buildSnapshotForVerificationPinn({
          n_hidden,
          depth,
          seed: 42,
          t_colloc: [0.25, 0.5, 0.75],
          a_colloc: [1, 1, 1],
          t_data: [],
          u_data: [],
          w_res: 1,
          w_data: 0,
          ode_type,
        }),
      });
    }
  }

  // PINN: L_data only (empty colloc)
  list.push({
    name: "pinn_data_exponential_depth2",
    snapshot: buildSnapshotForVerificationPinn({
      n_hidden,
      depth: 2,
      seed: 43,
      t_colloc: [],
      a_colloc: [],
      t_data: [0.3, 0.6],
      u_data: [1.5, 2.2],
      w_res: 0,
      w_data: 1,
      ode_type: "exponential",
    }),
  });

  // PINN: combined L_res + L_data
  list.push({
    name: "pinn_combined_exponential_depth2",
    snapshot: buildSnapshotForVerificationPinn({
      n_hidden,
      depth: 2,
      seed: 45,
      t_colloc: [0.25, 0.5, 0.75],
      a_colloc: [1, 1, 1],
      t_data: [0.5],
      u_data: [1.65],
      w_res: 1,
      w_data: 0.5,
      ode_type: "exponential",
    }),
  });

  return list;
}
