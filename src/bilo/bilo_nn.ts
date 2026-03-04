/**
 * BILO model: manual backprop for PDE u' = a*u
 * General d-layer: input [t, a] -> hidden 1 .. hidden d-1 (tanh) -> N -> u = 1 + t*N
 * Same recurrence and backward as playground/bilo_np/model.py
 */

function mulberry32(seed: number): () => number {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function tanhDerivatives(z: number[]): {
  sigma: number[];
  sigma_p: number[];
  sigma_pp: number[];
  sigma_ppp: number[];
} {
  const n = z.length;
  const sigma = new Array<number>(n);
  const sigma_p = new Array<number>(n);
  const sigma_pp = new Array<number>(n);
  const sigma_ppp = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const s = Math.tanh(z[i]);
    sigma[i] = s;
    sigma_p[i] = 1 - s * s;
    sigma_pp[i] = -2 * s * sigma_p[i];
    sigma_ppp[i] = 2 * sigma_p[i] * (3 * s * s - 1);
  }
  return { sigma, sigma_p, sigma_pp, sigma_ppp };
}

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function ewMult(a: number[], b: number[]): number[] {
  return a.map((x, i) => x * b[i]);
}

/** Matrix-vector: W (rows x cols) @ h (cols) -> (rows). */
function matVec(W: number[][], h: number[]): number[] {
  return W.map(row => dot(row, h));
}

/** Row vector W (length n) dot h (length n) -> scalar. */
function dotRowVec(W: number[], h: number[]): number {
  return dot(W, h);
}

/** Outer product a (n) and b (m) -> (n x m). */
function outer(a: number[], b: number[]): number[][] {
  return a.map(ai => b.map(bj => ai * bj));
}

/** Transpose of W (n x m) -> (m x n). */
function transpose(W: number[][]): number[][] {
  const n = W.length;
  const m = W[0].length;
  const out: number[][] = [];
  for (let j = 0; j < m; j++) {
    out[j] = [];
    for (let i = 0; i < n; i++) out[j][i] = W[i][j];
  }
  return out;
}

/** Add two matrices (n x m) in place into first. */
function addMatInPlace(A: number[][], B: number[][]): void {
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < A[i].length; j++) A[i][j] += B[i][j];
}

/** Add two vectors in place into first. */
function addVecInPlace(a: number[], b: number[]): void {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
}

export type WeightMatrix = number[][] | number[];

export class BILOModel {
  n_hidden: number;
  depth: number;
  /** W[0]: (n x 2), W[1]..W[depth-2]: (n x n), W[depth-1]: (n,) output row */
  _W: WeightMatrix[];
  /** b[0]..b[depth-2]: (n,), b[depth-1]: scalar */
  _b: (number[] | number)[];

  constructor(n_hidden: number, depth: number = 2, seed?: number) {
    if (depth < 2) throw new Error("depth must be >= 2");
    this.n_hidden = n_hidden;
    this.depth = depth;
    const n = n_hidden;
    const d = depth;
    const raw = seed !== undefined ? mulberry32(seed) : () => Math.random();
    const rng = () => raw() - 0.5;

    this._W = [];
    this._b = [];
    // Layer 1: (n x 2)
    this._W.push(Array.from({ length: n }, () => Array.from({ length: 2 }, rng)));
    this._b.push(new Array(n).fill(0));
    // Layers 2 .. d-1: (n x n)
    for (let _ = 0; _ < d - 2; _++) {
      this._W.push(Array.from({ length: n }, () => Array.from({ length: n }, rng)));
      this._b.push(new Array(n).fill(0));
    }
    // Output layer d: (n,)
    this._W.push(Array.from({ length: n }, rng));
    this._b.push(0);
  }

  /** Forward + forward kinematics. Returns N, N_t, N_a, N_ta and layer state lists. */
  private forwardAndKinematics(
    t: number,
    a: number
  ): {
    N: number;
    N_t: number;
    N_a: number;
    N_ta: number;
    h_list: number[][];
    z_list: number[][];
    h_t_list: number[][];
    h_a_list: number[][];
    h_ta_list: number[][];
    z_t_list: number[][];
    z_a_list: number[][];
    z_ta_list: number[][];
    sigma_list: { sigma: number[]; sigma_p: number[]; sigma_pp: number[]; sigma_ppp: number[] }[];
  } {
    const x = [t, a];
    const h_0_t = [1, 0];
    const h_0_a = [0, 1];
    const h_0_ta = [0, 0];

    const h_list: number[][] = [x];
    const z_list: number[][] = [];
    const h_t_list: number[][] = [h_0_t];
    const h_a_list: number[][] = [h_0_a];
    const h_ta_list: number[][] = [h_0_ta];
    const z_t_list: number[][] = [];
    const z_a_list: number[][] = [];
    const z_ta_list: number[][] = [];
    const sigma_list: { sigma: number[]; sigma_p: number[]; sigma_pp: number[]; sigma_ppp: number[] }[] = [];

    for (let k = 0; k < this.depth - 1; k++) {
      const W_k = this._W[k] as number[][];
      const b_k = this._b[k] as number[];
      const h_prev = h_list[h_list.length - 1];
      const h_prev_t = h_t_list[h_t_list.length - 1];
      const h_prev_a = h_a_list[h_a_list.length - 1];
      const h_prev_ta = h_ta_list[h_ta_list.length - 1];

      const z_k = matVec(W_k, h_prev).map((v, i) => v + b_k[i]);
      const { sigma, sigma_p, sigma_pp, sigma_ppp } = tanhDerivatives(z_k);

      const z_k_t = matVec(W_k, h_prev_t);
      const z_k_a = matVec(W_k, h_prev_a);
      const z_k_ta = matVec(W_k, h_prev_ta);

      const h_k_t = ewMult(sigma_p, z_k_t);
      const h_k_a = ewMult(sigma_p, z_k_a);
      const h_k_ta = ewMult(ewMult(sigma_pp, z_k_t), z_k_a).map((v, i) => v + sigma_p[i] * z_k_ta[i]);

      z_list.push(z_k);
      h_list.push(sigma);
      h_t_list.push(h_k_t);
      h_a_list.push(h_k_a);
      h_ta_list.push(h_k_ta);
      z_t_list.push(z_k_t);
      z_a_list.push(z_k_a);
      z_ta_list.push(z_k_ta);
      sigma_list.push({ sigma, sigma_p, sigma_pp, sigma_ppp });
    }

    const W_d = this._W[this.depth - 1] as number[];
    const b_d = this._b[this.depth - 1] as number;
    const h_last = h_list[h_list.length - 1];
    const h_last_t = h_t_list[h_t_list.length - 1];
    const h_last_a = h_a_list[h_a_list.length - 1];
    const h_last_ta = h_ta_list[h_ta_list.length - 1];

    const N = dotRowVec(W_d, h_last) + b_d;
    const N_t = dotRowVec(W_d, h_last_t);
    const N_a = dotRowVec(W_d, h_last_a);
    const N_ta = dotRowVec(W_d, h_last_ta);

    return {
      N,
      N_t,
      N_a,
      N_ta,
      h_list,
      z_list,
      h_t_list,
      h_a_list,
      h_ta_list,
      z_t_list,
      z_a_list,
      z_ta_list,
      sigma_list,
    };
  }

  forward(t: number, a: number): {
    N: number;
    N_t: number;
    N_a: number;
    N_ta: number;
    u: number;
    u_t: number;
    u_a: number;
    u_ta: number;
    h_list: number[][];
    sigma_list: { sigma: number[]; sigma_p: number[]; sigma_pp: number[]; sigma_ppp: number[] }[];
  } {
    const f = this.forwardAndKinematics(t, a);
    const u = 1 + t * f.N;
    const u_t = f.N + t * f.N_t;
    const u_a = t * f.N_a;
    const u_ta = f.N_a + t * f.N_ta;
    return {
      N: f.N,
      N_t: f.N_t,
      N_a: f.N_a,
      N_ta: f.N_ta,
      u,
      u_t,
      u_a,
      u_ta,
      h_list: f.h_list,
      sigma_list: f.sigma_list,
    };
  }

  evalU(t: number, a: number): number {
    return this.forward(t, a).u;
  }

  evalUArray(tArr: number[], a: number): number[] {
    return tArr.map(t => this.evalU(t, a));
  }

  /** Returns sigma (hidden activation) of the first hidden layer on a (t,a) grid. [neuronIdx][ix][iy]. */
  getSigmaGrid(tGrid: number[], aGrid: number[]): number[][][] {
    const out: number[][][] = [];
    const n = this.n_hidden;
    for (let j = 0; j < n; j++) {
      out[j] = [];
      for (let ix = 0; ix < tGrid.length; ix++) {
        out[j][ix] = [];
        for (let iy = 0; iy < aGrid.length; iy++) {
          const f = this.forward(tGrid[ix], aGrid[iy]);
          out[j][ix][iy] = f.sigma_list[0] ? f.sigma_list[0].sigma[j] : 0;
        }
      }
    }
    return out;
  }

  getNGrid(tGrid: number[], aGrid: number[]): number[][] {
    const out: number[][] = [];
    for (let ix = 0; ix < tGrid.length; ix++) {
      out[ix] = [];
      for (let iy = 0; iy < aGrid.length; iy++) {
        out[ix][iy] = this.forward(tGrid[ix], aGrid[iy]).N;
      }
    }
    return out;
  }

  getUGrid(tGrid: number[], aGrid: number[]): number[][] {
    const out: number[][] = [];
    for (let ix = 0; ix < tGrid.length; ix++) {
      out[ix] = [];
      for (let iy = 0; iy < aGrid.length; iy++) {
        out[ix][iy] = this.forward(tGrid[ix], aGrid[iy]).u;
      }
    }
    return out;
  }

  private backwardOnePoint(
    t: number,
    a: number,
    R: number,
    R_a: number,
    delta_N_data: number,
    w_res: number,
    w_grad: number
  ): { grad_W: WeightMatrix[]; grad_b: (number[] | number)[] } {
    const f = this.forwardAndKinematics(t, a);
    const delta_N = w_res * R * (1 - a * t) - w_grad * R_a * t + delta_N_data;
    const delta_N_t = w_res * R * t;
    const delta_N_a = w_grad * R_a * (1 - a * t);
    const delta_N_ta = w_grad * R_a * t;

    const W_d = this._W[this.depth - 1] as number[];
    let delta_h = W_d.map(w => delta_N * w);
    let delta_h_t = W_d.map(w => delta_N_t * w);
    let delta_h_a = W_d.map(w => delta_N_a * w);
    let delta_h_ta = W_d.map(w => delta_N_ta * w);

    const grad_W: WeightMatrix[] = this._W.map(w => (Array.isArray(w[0]) ? (w as number[][]).map(row => row.map(() => 0)) : (w as number[]).map(() => 0))) as WeightMatrix[];
    const grad_b: (number[] | number)[] = this._b.map(b => (typeof b === "number" ? 0 : (b as number[]).map(() => 0)));

    // Output layer gradient
    const h_last = f.h_list[f.h_list.length - 1];
    const h_last_t = f.h_t_list[f.h_t_list.length - 1];
    const h_last_a = f.h_a_list[f.h_a_list.length - 1];
    const h_last_ta = f.h_ta_list[f.h_ta_list.length - 1];
    (grad_W[this.depth - 1] as number[]) = h_last.map((v, i) => delta_N * v + delta_N_t * h_last_t[i] + delta_N_a * h_last_a[i] + delta_N_ta * h_last_ta[i]);
    (grad_b[this.depth - 1] as number) = delta_N;

    for (let k = this.depth - 2; k >= 0; k--) {
      const z_k = f.z_list[k];
      const { sigma_p, sigma_pp, sigma_ppp } = f.sigma_list[k];
      const z_k_t = f.z_t_list[k];
      const z_k_a = f.z_a_list[k];
      const z_k_ta = f.z_ta_list[k];
      const h_prev = f.h_list[k];
      const h_prev_t = f.h_t_list[k];
      const h_prev_a = f.h_a_list[k];
      const h_prev_ta = f.h_ta_list[k];
      const W_k = this._W[k] as number[][];

      const delta_z_ta = ewMult(delta_h_ta, sigma_p);
      const delta_z_t = ewMult(delta_h_t, sigma_p).map((v, i) => v + delta_h_ta[i] * sigma_pp[i] * z_k_a[i]);
      const delta_z_a = ewMult(delta_h_a, sigma_p).map((v, i) => v + delta_h_ta[i] * sigma_pp[i] * z_k_t[i]);
      const delta_z =
        ewMult(delta_h, sigma_p)
          .map((v, i) => v + delta_h_t[i] * sigma_pp[i] * z_k_t[i])
          .map((v, i) => v + delta_h_a[i] * sigma_pp[i] * z_k_a[i])
          .map((v, i) => v + delta_h_ta[i] * (sigma_ppp[i] * z_k_t[i] * z_k_a[i] + sigma_pp[i] * z_k_ta[i]));

      const O = outer(delta_z, h_prev);
      const Ot = outer(delta_z_t, h_prev_t);
      const Oa = outer(delta_z_a, h_prev_a);
      const Ota = outer(delta_z_ta, h_prev_ta);
      (grad_W[k] as number[][]) = O.map((row, i) => row.map((v, j) => v + Ot[i][j] + Oa[i][j] + Ota[i][j]));
      (grad_b[k] as number[]) = delta_z;

      const Wt = transpose(W_k);
      delta_h = matVec(Wt, delta_z);
      delta_h_t = matVec(Wt, delta_z_t);
      delta_h_a = matVec(Wt, delta_z_a);
      delta_h_ta = matVec(Wt, delta_z_ta);
    }

    return { grad_W, grad_b };
  }

  computeLossesAndGradients(
    t_colloc: number[],
    a_colloc: number[],
    opts?: { t_data?: number[]; a_data?: number[]; u_data?: number[]; w_res?: number; w_grad?: number; w_data?: number }
  ): {
    losses: { L_res: number; L_grad: number; L_data: number };
    grads: Record<string, number[] | number[][] | number>;
  } {
    const w_res = opts?.w_res ?? 1;
    const w_grad = opts?.w_grad ?? 0.1;
    const w_data = opts?.w_data ?? 0;
    const t_data = opts?.t_data;
    const a_data = opts?.a_data;
    const u_data = opts?.u_data;

    let L_res = 0,
      L_grad = 0,
      L_data = 0;
    const grad_W_acc: WeightMatrix[] = this._W.map(w =>
      Array.isArray(w[0])
        ? (w as number[][]).map(row => row.map(() => 0))
        : (w as number[]).map(() => 0)
    ) as WeightMatrix[];
    const grad_b_acc: (number[] | number)[] = this._b.map(b =>
      typeof b === "number" ? 0 : (b as number[]).map(() => 0)
    );

    for (let i = 0; i < t_colloc.length; i++) {
      const t = t_colloc[i],
        a = a_colloc[i];
      const f = this.forward(t, a);
      const R = f.u_t - a * f.u;
      const R_a = f.u_ta - (f.u + a * f.u_a);
      L_res += 0.5 * R * R;
      L_grad += 0.5 * R_a * R_a;

      const { grad_W, grad_b } = this.backwardOnePoint(t, a, R, R_a, 0, w_res, w_grad);
      for (let k = 0; k < this.depth; k++) {
        if (Array.isArray(grad_W[k][0])) {
          addMatInPlace(grad_W_acc[k] as number[][], grad_W[k] as number[][]);
        } else {
          const g = grad_W[k] as number[];
          const acc = grad_W_acc[k] as number[];
          for (let j = 0; j < g.length; j++) acc[j] += g[j];
        }
        if (typeof grad_b[k] === "number") {
          (grad_b_acc[k] as number) += grad_b[k] as number;
        } else {
          addVecInPlace(grad_b_acc[k] as number[], grad_b[k] as number[]);
        }
      }
    }

    let dL_da = 0;
    if (t_data && a_data && u_data) {
      for (let i = 0; i < t_data.length; i++) {
        const t = t_data[i],
          a = a_data[i],
          u_target = u_data[i];
        const f = this.forward(t, a);
        const err = f.u - u_target;
        L_data += 0.5 * err * err;
        const delta_N_data = w_data * err * t;
        const { grad_W, grad_b } = this.backwardOnePoint(t, a, 0, 0, delta_N_data, 0, 0);
        for (let k = 0; k < this.depth; k++) {
          if (Array.isArray(grad_W[k][0])) {
            addMatInPlace(grad_W_acc[k] as number[][], grad_W[k] as number[][]);
          } else {
            const g = grad_W[k] as number[];
            const acc = grad_W_acc[k] as number[];
            for (let j = 0; j < g.length; j++) acc[j] += g[j];
          }
          if (typeof grad_b[k] === "number") {
            (grad_b_acc[k] as number) += grad_b[k] as number;
          } else {
            addVecInPlace(grad_b_acc[k] as number[], grad_b[k] as number[]);
          }
        }
        dL_da += w_data * err * f.u_a;
      }
    }

    const grads: Record<string, number[] | number[][] | number> = {};
    for (let k = 0; k < this.depth; k++) {
      grads[`W${k + 1}`] = grad_W_acc[k] as number[] | number[][];
      grads[`b${k + 1}`] = grad_b_acc[k] as number[] | number;
    }
    if (t_data && a_data && u_data) grads.a = dL_da;

    return { losses: { L_res, L_grad, L_data }, grads };
  }

  /** Backward compat for UI (depth=2 only): first-layer weights as W1t, W1a. */
  get W1t(): number[] {
    return this.depth === 2 ? (this._W[0] as number[][]).map(row => row[0]) : [];
  }
  get W1a(): number[] {
    return this.depth === 2 ? (this._W[0] as number[][]).map(row => row[1]) : [];
  }
  get b1(): number[] {
    return this.depth === 2 ? (this._b[0] as number[]) : [];
  }
  get W2(): number[] {
    return this.depth === 2 ? (this._W[1] as number[]) : [];
  }
  get b2(): number {
    return this.depth === 2 ? (this._b[1] as number) : 0;
  }

  /** Export weights and config for Python/PyTorch verification. */
  exportForVerification(): {
    n_hidden: number;
    depth: number;
    W: (number[][] | number[])[];
    b: (number[] | number)[];
  } {
    return {
      n_hidden: this.n_hidden,
      depth: this.depth,
      W: this._W.map(w => (Array.isArray(w[0]) ? (w as number[][]).map(row => [...row]) : [...(w as number[])])) as (number[][] | number[])[],
      b: this._b.map(b => (typeof b === "number" ? b : [...b])),
    };
  }
}
