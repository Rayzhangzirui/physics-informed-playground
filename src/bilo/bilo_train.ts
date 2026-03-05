/**
 * BILO training: pretrain (fix a=1) and optional finetune with data.
 * Supports SGD and ADAM (no AMSgrad); ADAM matches bilo_np/train.py defaults.
 */

import { BILOModel, WeightMatrix } from "./bilo_nn";

export const ADAM_BETA1 = 0.9;
export const ADAM_BETA2 = 0.999;
export const ADAM_EPS = 1e-8;

export interface AdamState {
  m_W: (number[][] | number[])[];
  v_W: (number[][] | number[])[];
  m_b: (number[] | number)[];
  v_b: (number[] | number)[];
  m_a?: number;
  v_a?: number;
}

export interface TrainStep {
  step: number;
  L_res: number;
  L_grad: number;
  L_data: number;
  L_total: number;
  a?: number;
}

function zerosLikeW(w: WeightMatrix): WeightMatrix {
  if (Array.isArray((w as number[][])[0])) {
    return (w as number[][]).map(row => row.map(() => 0));
  }
  return (w as number[]).map(() => 0);
}

function zerosLikeB(b: number[] | number): number[] | number {
  if (typeof b === "number") return 0;
  return b.map(() => 0);
}

/** One ADAM step for model weights (W and b). step is 1-based. Mutates model and state. */
export function adamStepModel(
  model: BILOModel,
  grads: Record<string, number[] | number[][] | number>,
  state: AdamState,
  step: number,
  lr: number,
  beta1: number = ADAM_BETA1,
  beta2: number = ADAM_BETA2,
  eps: number = ADAM_EPS
): void {
  if (state.m_W.length === 0) {
    state.m_W = model._W.map(w => zerosLikeW(w)) as (number[][] | number[])[];
    state.v_W = model._W.map(w => zerosLikeW(w)) as (number[][] | number[])[];
    state.m_b = model._b.map(b => zerosLikeB(b));
    state.v_b = model._b.map(b => zerosLikeB(b));
  }
  for (let k = 0; k < model.depth; k++) {
    const gW = grads[`W${k + 1}`] as number[] | number[][];
    const gb = grads[`b${k + 1}`] as number[] | number;
    const mW = state.m_W[k];
    const vW = state.v_W[k];
    const mb = state.m_b[k];
    const vb = state.v_b[k];

    const updateMoment = (
      m: number[][] | number[] | number,
      v: number[][] | number[] | number,
      g: number[][] | number[] | number,
      isScalar: boolean
    ) => {
      if (isScalar) {
        const mVal = (m as number) * beta1 + (1 - beta1) * (g as number);
        const vVal = (v as number) * beta2 + (1 - beta2) * (g as number) * (g as number);
        return { mHat: mVal / (1 - Math.pow(beta1, step)), vHat: vVal / (1 - Math.pow(beta2, step)) };
      }
      if (Array.isArray((g as number[][])[0])) {
        const G = g as number[][];
        const M = m as number[][];
        const V = v as number[][];
        for (let i = 0; i < G.length; i++)
          for (let j = 0; j < G[i].length; j++) {
            M[i][j] = beta1 * M[i][j] + (1 - beta1) * G[i][j];
            V[i][j] = beta2 * V[i][j] + (1 - beta2) * G[i][j] * G[i][j];
          }
        const mHat = (M as number[][]).map(row => row.map(x => x / (1 - Math.pow(beta1, step))));
        const vHat = (V as number[][]).map(row => row.map(x => x / (1 - Math.pow(beta2, step))));
        return { mHat, vHat };
      }
      const G = g as number[];
      const M = m as number[];
      const V = v as number[];
      for (let i = 0; i < G.length; i++) {
        M[i] = beta1 * M[i] + (1 - beta1) * G[i];
        V[i] = beta2 * V[i] + (1 - beta2) * G[i] * G[i];
      }
      const mHat = M.map(x => x / (1 - Math.pow(beta1, step)));
      const vHat = V.map(x => x / (1 - Math.pow(beta2, step)));
      return { mHat, vHat };
    };

    const isScalarB = typeof model._b[k] === "number";
    if (isScalarB) {
      (state.m_b[k] as number) = beta1 * (state.m_b[k] as number) + (1 - beta1) * (gb as number);
      (state.v_b[k] as number) = beta2 * (state.v_b[k] as number) + (1 - beta2) * (gb as number) * (gb as number);
    }
    const { mHat: mW_hat, vHat: vW_hat } = updateMoment(mW, vW, gW, false);
    if (Array.isArray((mW_hat as number[][])[0])) {
      const W = model._W[k] as number[][];
      const M = mW_hat as number[][];
      const V = vW_hat as number[][];
      for (let i = 0; i < W.length; i++)
        for (let j = 0; j < W[i].length; j++)
          W[i][j] -= (lr * M[i][j]) / (Math.sqrt(V[i][j]) + eps);
    } else {
      const W = model._W[k] as number[];
      const M = mW_hat as number[];
      const V = vW_hat as number[];
      for (let i = 0; i < W.length; i++) W[i] -= (lr * M[i]) / (Math.sqrt(V[i]) + eps);
    }

    const { mHat: mb_hat, vHat: vb_hat } = updateMoment(mb, vb, gb, isScalarB);
    if (isScalarB) {
      (model._b[k] as number) = (model._b[k] as number) - (lr * (mb_hat as number)) / (Math.sqrt(vb_hat as number) + eps);
    } else {
      const b = model._b[k] as number[];
      const Mb = mb_hat as number[];
      const Vb = vb_hat as number[];
      for (let i = 0; i < b.length; i++) b[i] -= (lr * Mb[i]) / (Math.sqrt(Vb[i]) + eps);
    }
  }
}

/** One ADAM step for scalar a (finetune). Returns new a. */
export function adamStepA(
  a: number,
  gradA: number,
  state: AdamState,
  step: number,
  lrA: number,
  beta1: number = ADAM_BETA1,
  beta2: number = ADAM_BETA2,
  eps: number = ADAM_EPS
): number {
  if (state.m_a === undefined) {
    state.m_a = 0;
    state.v_a = 0;
  }
  state.m_a = beta1 * state.m_a + (1 - beta1) * gradA;
  state.v_a = beta2 * state.v_a + (1 - beta2) * gradA * gradA;
  const mHat = state.m_a / (1 - Math.pow(beta1, step));
  const vHat = state.v_a / (1 - Math.pow(beta2, step));
  return a - (lrA * mHat) / (Math.sqrt(vHat) + eps);
}

export function train(
  model: BILOModel,
  t_colloc: number[],
  a_colloc: number[],
  opts: { n_iters: number; lr: number; w_res?: number; w_grad?: number; log_every?: number }
): TrainStep[] {
  const { n_iters, lr, w_res = 1, w_grad = 0.1, log_every = 100 } = opts;
  const history: TrainStep[] = [];
  for (let step = 0; step < n_iters; step++) {
    const { losses, grads } = model.computeLossesAndGradients(t_colloc, a_colloc, { w_res, w_grad, w_data: 0 });
    const L_total = w_res * losses.L_res + w_grad * losses.L_grad;

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

    history.push({ step, ...losses, L_total });
  }
  return history;
}
