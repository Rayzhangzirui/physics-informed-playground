# Higher-Order Derivatives in a $d$-Layer PINN

## 1. Forward Pass

Let the input be

\[
x =
\begin{bmatrix}
t \\
a
\end{bmatrix}
\in \mathbb{R}^2,
\]

and define

\[
h_0 = x.
\]

For hidden layers \(k \in \{1,2,\dots,d-1\}\):

- \(W_1 \in \mathbb{R}^{n \times 2}\)
- \(W_k \in \mathbb{R}^{n \times n}\) for \(k>1\)
- \(b_k \in \mathbb{R}^{n \times 1}\)

The output layer \(d\) has

- \(W_d \in \mathbb{R}^{1 \times n}\)
- \(b_d \in \mathbb{R}\)

The forward recurrence is

\[
z_k = W_k h_{k-1} + b_k
\]

\[
h_k = \sigma(z_k)
\]

\[
N(t,a; W) = z_d = W_d h_{d-1} + b_d
\]

---

# 2. Forward Propagation of Input Derivatives

Derivatives with respect to inputs are propagated alongside the forward pass.

Let

\[
h_{k,t} = \frac{\partial h_k}{\partial t}, \quad
h_{k,a} = \frac{\partial h_k}{\partial a}
\]

and

\[
h_{k,ta} = \frac{\partial^2 h_k}{\partial t \partial a}.
\]

### Base derivatives

\[
h_{0,t} =
\begin{bmatrix}
1 \\
0
\end{bmatrix},
\quad
h_{0,a} =
\begin{bmatrix}
0 \\
1
\end{bmatrix},
\quad
h_{0,ta} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
\]

---

### First derivatives

For \(k = 1,\dots,d-1\)

\[
z_{k,t} = W_k h_{k-1,t}
\]

\[
h_{k,t} = \sigma'(z_k) \odot z_{k,t}
\]

\[
z_{k,a} = W_k h_{k-1,a}
\]

\[
h_{k,a} = \sigma'(z_k) \odot z_{k,a}
\]

---

### Mixed derivative

\[
z_{k,ta} = W_k h_{k-1,ta}
\]

\[
h_{k,ta} =
\sigma''(z_k) \odot z_{k,t} \odot z_{k,a}
+
\sigma'(z_k) \odot z_{k,ta}
\]

---

### Final derivatives

\[
N_t = W_d h_{d-1,t}
\]

\[
N_a = W_d h_{d-1,a}
\]

\[
N_{ta} = W_d h_{d-1,ta}
\]

Each derivative at layer \(k\) depends only on derivatives from layer \(k-1\) and activations at layer \(k\).

---

# 3. Backward Dynamics (Adjoints)

The loss $\mathcal{L}$
depends on $N, N_t, N_a, N_{ta}$.

Define upstream gradients

\[
\delta_N = \frac{\partial \mathcal{L}}{\partial N},
\quad
\delta_{N_t} = \frac{\partial \mathcal{L}}{\partial N_t},
\quad
\delta_{N_a} = \frac{\partial \mathcal{L}}{\partial N_a},
\quad
\delta_{N_{ta}} = \frac{\partial \mathcal{L}}{\partial N_{ta}}.
\]

Each layer propagates four adjoint signals corresponding to

\[
z_k, \quad z_{k,t}, \quad z_{k,a}, \quad z_{k,ta}.
\]

---

# 4. Gradient Extraction

After backward propagation, gradients for layer \(k\) are

### Weights

\[
\frac{\partial \mathcal{L}}{\partial W_k}=\delta_{z_k} h_{k-1}^T+\delta_{z_{k,t}} h_{k-1,t}^T+\delta_{z_{k,a}} h_{k-1,a}^T+\delta_{z_{k,ta}} h_{k-1,ta}^T
\]

### Bias

\[
\frac{\partial \mathcal{L}}{\partial b_k} = \delta_{z_k}
\]

Bias does not appear in derivative tracks because

\[
\frac{\partial b_k}{\partial t} =
\frac{\partial b_k}{\partial a} = 0.
\]

---

# 5. Adjoint Recursions

At hidden layer \(k\), four adjoints arrive:

\[
\delta_{h_k},
\quad
\delta_{h_{k,t}},
\quad
\delta_{h_{k,a}},
\quad
\delta_{h_{k,ta}}.
\]

---

## Mixed adjoint

From

$$
h_{k,ta}= \sigma ''(z_k) \odot z_{k,t} \odot z_{k,a} + \sigma'(z_k) \odot z_{k,ta}
$$

we obtain

\[
\delta_{z_{k,ta}} =
\delta_{h_{k,ta}} \odot \sigma'(z_k)
\]

---

## First derivative adjoints

\[
\delta_{z_{k,t}}=
\delta_{h_{k,t}} \odot \sigma'(z_k)+
\delta_{h_{k,ta}} \odot \sigma''(z_k) \odot z_{k,a}
\]

\[
\delta_{z_{k,a}}=
\delta_{h_{k,a}} \odot \sigma'(z_k)+
\delta_{h_{k,ta}} \odot \sigma''(z_k) \odot z_{k,t}
\]

---

## Base adjoint

\[
\delta_{z_k}
=\underbrace{\delta_{h_k} \odot \sigma'(z_k)}_{\text{from } h_k}
+
\underbrace{\delta_{h_{k,t}} \odot \left(\sigma''(z_k) \odot z_{k,t}\right)}_{\text{from } h_{k,t}}
+
\underbrace{\delta_{h_{k,a}} \odot \left(\sigma''(z_k) \odot z_{k,a}\right)}_{\text{from } h_{k,a}}
+
\underbrace{\delta_{h_{k,ta}} \odot \left(\sigma'''(z_k) \odot z_{k,t} \odot z_{k,a} + \sigma''(z_k) \odot z_{k,ta}\right)}_{\text{from } h_{k,ta}}
\]

---

## Passing gradients downward

\[
\delta_{h_{k-1}} = W_k^T \delta_{z_k}
\]

\[
\delta_{h_{k-1,t}} = W_k^T \delta_{z_{k,t}}
\]

\[
\delta_{h_{k-1,a}} = W_k^T \delta_{z_{k,a}}
\]

\[
\delta_{h_{k-1,ta}} = W_k^T \delta_{z_{k,ta}}
\]

---

# 6. Example: Two Hidden Layers (\(d=3\))

Network structure:

Input → Layer 1 → Layer 2 → Output

---

## Step 1: Output layer initialization

\[
\delta_N =
\frac{\partial \mathcal{L}}{\partial N},
\quad
\delta_{N_t},
\quad
\delta_{N_a},
\quad
\delta_{N_{ta}}
\]

Since

\[
N = W_3 h_2 + b_3
\]

we obtain

\[
\delta_{h_2} = W_3^T \delta_N
\]

\[
\delta_{h_{2,t}} = W_3^T \delta_{N_t}
\]

\[
\delta_{h_{2,a}} = W_3^T \delta_{N_a}
\]

\[
\delta_{h_{2,ta}} = W_3^T \delta_{N_{ta}}
\]

---

## Step 2: Backprop through Layer 2

\[
\delta_{z_{2,ta}} =
\delta_{h_{2,ta}} \odot \sigma'(z_2)
\]

\[
\delta_{z_{2,t}} =
\delta_{h_{2,t}} \odot \sigma'(z_2)
+
\delta_{h_{2,ta}} \odot \sigma''(z_2) \odot z_{2,a}
\]

\[
\delta_{z_{2,a}} =
\delta_{h_{2,a}} \odot \sigma'(z_2)
+
\delta_{h_{2,ta}} \odot \sigma''(z_2) \odot z_{2,t}
\]

\[
\delta_{z_2} =
\delta_{h_2} \odot \sigma'(z_2)
+
\delta_{h_{2,t}} \odot \sigma''(z_2) \odot z_{2,t}
+
\delta_{h_{2,a}} \odot \sigma''(z_2) \odot z_{2,a}
+
\delta_{h_{2,ta}} \odot
\left(
\sigma'''(z_2) \odot z_{2,t} \odot z_{2,a}
+
\sigma''(z_2) \odot z_{2,ta}
\right)
\]

Gradient extraction

\[
\frac{\partial \mathcal{L}}{\partial W_2}=
\delta_{z_2} h_1^T
+
\delta_{z_{2,t}} h_{1,t}^T
+
\delta_{z_{2,a}} h_{1,a}^T
+
\delta_{z_{2,ta}} h_{1,ta}^T
\]

Pass error downward

\[
\delta_{h_1} = W_2^T \delta_{z_2}
\]

\[
\delta_{h_{1,t}} = W_2^T \delta_{z_{2,t}}
\]

\[
\delta_{h_{1,a}} = W_2^T \delta_{z_{2,a}}
\]

\[
\delta_{h_{1,ta}} = W_2^T \delta_{z_{2,ta}}
\]

---

## Step 3: Backprop through Layer 1

\[
\delta_{z_{1,ta}} = \delta_{h_{1,ta}} \odot \sigma'(z_1)
\]

\[
\delta_{z_{1,t}} = \delta_{h_{1,t}} \odot \sigma'(z_1)
+
\delta_{h_{1,ta}} \odot \sigma''(z_1) \odot z_{1,a}
\]

\[
\delta_{z_{1,a}} =
\delta_{h_{1,a}} \odot \sigma'(z_1)
+
\delta_{h_{1,ta}} \odot \sigma''(z_1) \odot z_{1,t}
\]

\[
\delta_{z_1}=
\delta_{h_1} \odot \sigma'(z_1)
+
\delta_{h_{1,t}} \odot \sigma''(z_1) \odot z_{1,t}
+
\delta_{h_{1,a}} \odot \sigma''(z_1) \odot z_{1,a}
+
\delta_{h_{1,ta}} \odot
\left(
\sigma'''(z_1) \odot z_{1,t} \odot z_{1,a}
+
\sigma''(z_1) \odot z_{1,ta}
\right)
\]

Gradient extraction

\[
\frac{\partial \mathcal{L}}{\partial W_1}=
\delta_{z_1} h_0^T
+
\delta_{z_{1,t}} h_{0,t}^T
+
\delta_{z_{1,a}} h_{0,a}^T
+
\delta_{z_{1,ta}} h_{0,ta}^T
\]

with

\[
h_0 =
\begin{bmatrix}
t \\
a
\end{bmatrix},
\quad
h_{0,t} =
\begin{bmatrix}
1 \\
0
\end{bmatrix},
\quad
h_{0,a} =
\begin{bmatrix}
0 \\
1
\end{bmatrix},
\quad
h_{0,ta} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}.
\]