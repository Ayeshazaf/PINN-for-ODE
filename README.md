

# 📘 README — Physics-Informed Neural Network (PINN) for ODE

## 📌 Overview

This project implements a **Physics-Informed Neural Network (PINN)** using PyTorch to solve the second-order differential equation:

[
y'' + 16y = 0
]

with initial conditions:
[
y(0) = 2,  y'(0) = -2
]

Instead of using traditional numerical solvers, the model learns the solution by embedding the **physics (differential equation)** directly into the loss function.

---

## ⚙️ Model Architecture

A fully connected neural network:

* Input: ( x (Real Number) )
* Output: ( y(x) )
* Layers:

  ```
  1 → 16 → 32 → 32 → 16 → 1
  ```
* Activation: `Tanh`

---

## 🧠 Key Idea (PINN)

The network is trained using two constraints:

### 1. Physics Loss

Ensures the model satisfies the differential equation:

[
Residual = y'' + 16y
]

Loss:
[
{L}_{physics} = MSE(y'' + 16y)
]

---

### 2. Initial Condition Loss

Forces the model to satisfy:

* ( y(0) = 2 )
* ( y'(0) = -2 )

Loss:
[
{L}_{IC} = (y(0) - 2)^2 + (y'(0) + 2)^2
]

---

### 3. Total Loss

[
{L} = {L}*{physics} + {L}*{IC}
]

---

## 🚀 Training Details

* Optimizer: Adam
* Learning rate: `5e-4`
* Epochs: `5000`
* Training domain: ( x \in [0, 2] )
* Collocation points per epoch: `256`

---

## 📈 Output

The script generates:

1. **PINN predicted solution plot**
2. **Exact analytical solution plot**

Exact solution:
[
y(x) = 2cos(4x) - 0.5sin(4x)
]

---

## 🧾 Code Explanation (Step-by-Step)

### 1. Model (`PINN`)

```python
class PINN(nn.Module):
```

* Standard feedforward network
* Takes `x` → outputs `y(x)`

---

### 2. Physics Loss

```python
u_x = grad(u, x)
u_xx = grad(u_x, x)
```

* Uses **autograd** to compute:

  * First derivative ( y' )
  * Second derivative ( y'' )
* Residual:

```python
residual = u_xx + 16.0 * u
```

---

### 3. Initial Condition Loss

```python
x0 = tensor([[0.0]])
```

* Computes:

  * ( y(0) )
  * ( y'(0) )
* Enforces constraints using MSE

---

### 4. Training Loop

```python
x = torch.rand(...) * 2.0
```

* Random sampling in domain (collocation points)
* Computes:

  * Physics loss
  * IC loss
* Backpropagation:

```python
loss.backward()
optimizer.step()
```

---

### 5. Prediction

```python
u_pred = model(x_test)
```

* Evaluates trained model on test grid

---

### 6. Visualization

* Plot 1: PINN solution
* Plot 2: Exact solution (for comparison)

---

