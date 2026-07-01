# Physics-Informed Neural Network for the 1D Viscous Burgers Equation

This repository contains a TensorFlow implementation of a **Physics-Informed Neural Network (PINN)** for solving the one-dimensional viscous Burgers equation.

The main goal of this project is educational: to understand how a neural network can be trained not only using data, but also using the governing differential equation as a physical constraint.

---

## 1. What Problem Is Solved?

This project solves the one-dimensional viscous Burgers equation using a Physics-Informed Neural Network.

The neural network learns the solution:

```math
u = u(x,t)
```

over the space-time domain:

```math
x \in [-1,1], \qquad t \in [0,1]
```

The model is trained using three types of information:

1. Initial-condition data
2. Boundary-condition data
3. Physics residual at collocation points inside the domain

Unlike a standard neural network that only fits labelled data, this PINN is also constrained to satisfy the governing partial differential equation.

---

## 2. Burgers Equation

The viscous Burgers equation is:

```math
u_t + u u_x - \nu u_{xx} = 0
```

where:

* (u(x,t)) is the unknown solution field
* (u_t) is the time derivative
* (u_x) is the spatial derivative
* (u_{xx}) is the second spatial derivative
* (\nu) is the viscosity or diffusion coefficient

In this implementation:

```math
\nu = \frac{0.01}{\pi}
```

The initial condition is:

```math
u(x,0) = -\sin(\pi x)
```

The boundary conditions are:

```math
u(-1,t) = 0
```

```math
u(1,t) = 0
```

Physically, the Burgers equation represents the competition between:

* nonlinear convection: (u u_x)
* viscous diffusion: (\nu u_{xx})

The nonlinear convection term tends to steepen the solution, while the viscous diffusion term smooths it.

---

## 3. PINN Idea

A Physics-Informed Neural Network approximates the unknown solution as:

```math
u_\theta(x,t)
```

where (\theta) represents the trainable weights and biases of the neural network.

The neural network takes two inputs:

```text
x, t
```

and gives one output:

```text
u(x,t)
```

The architecture used in this repository is:

```python
layers = [2, 20, 20, 20, 20, 20, 1]
```

This means:

* 2 input neurons: (x) and (t)
* 5 hidden layers
* 20 neurons per hidden layer
* 1 output neuron: (u(x,t))

The activation function used in the hidden layers is:

```python
tanh
```

This is useful because PINNs require smooth derivatives of the neural-network output.

---

## 4. Physics Residual

The key idea of the PINN is to convert the governing PDE into a residual.

For the Burgers equation:

```math
u_t + u u_x - \nu u_{xx} = 0
```

the residual is:

```math
f_\theta(x,t) = u_t + u_\theta u_x - \nu u_{xx}
```

For a perfect solution:

```math
f_\theta(x,t) = 0
```

at every point inside the domain.

In the code, TensorFlow automatic differentiation is used to compute:

```python
u_x
u_t
u_xx
```

The residual is then computed as:

```python
f = u_t + u * u_x - nu * u_xx
```

This allows the neural network to learn a solution that is consistent with the physics.

---

## 5. Loss Function

The total PINN loss contains two parts.

### Data loss

The data loss enforces the initial and boundary conditions:

```math
MSE_u = \frac{1}{N_u}\sum \left|u_\theta(x,t)-u_{\text{data}}\right|^2
```

### Physics loss

The physics loss enforces the Burgers equation at collocation points:

```math
MSE_f = \frac{1}{N_f}\sum \left|f_\theta(x,t)\right|^2
```

The total loss is:

```math
Loss = MSE_u + MSE_f
```

In code:

```python
total_loss = mse_u + mse_f
```

where:

* `mse_u` measures mismatch with initial and boundary data
* `mse_f` measures violation of the Burgers equation

---

## 6. Training Points

This implementation uses:

```python
N0 = 100
Nb = 100
Nf = 10000
```

where:

* `N0` = number of initial-condition points
* `Nb` = number of boundary-condition points on each boundary
* `Nf` = number of collocation points inside the domain

The supervised training data are built from:

```python
X0
X_left
X_right
```

The collocation data are built from randomly sampled points:

```python
X_f
```

The PDE residual is enforced at these collocation points.

---

## 7. How to Run the Notebook

Clone the repository:

```bash
git clone https://github.com/Rohit-lmmp/PINN-Burgers-Equation-.git
```

Enter the repository:

```bash
cd PINN-Burgers-Equation-
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then open and run:

```text
notebooks/burgers_pinn_tensorflow.ipynb
```

---

## 8. Requirements

The main Python packages are:

```text
numpy
matplotlib
tensorflow
jupyter
```

These are listed in:

```text
requirements.txt
```

---

## 9. Output Plots

The notebook generates several diagnostic plots:

1. Training-point distribution
2. Training loss history
3. Predicted solution contour plot
4. Solution profiles at selected time levels
5. Initial-condition check
6. Boundary-condition check

These plots are important because a PINN should not be judged only by the loss value. The solution must also satisfy the physical conditions visually and numerically.

---

## 10. Key Debugging Lesson: Use `X = [x, t]`, Not `X = [t, x]`

A very important lesson in this implementation is coordinate consistency.

The corrected convention used everywhere in this repository is:

```python
X[:, 0] = x
X[:, 1] = t
```

So every input point must be arranged as:

```python
X = [x, t]
```

For example:

```python
X0 = np.hstack([x0, t0])
X_left = np.hstack([x_left, tb])
X_right = np.hstack([x_right, tb])
X_f = np.hstack([x_f, t_f])
```

The residual function then correctly computes:

```python
u_x = grad_u[:, 0:1]
u_t = grad_u[:, 1:2]
```

If the input is accidentally arranged as:

```python
X = [t, x]
```

then the derivatives become physically wrong:

```python
u_x and u_t are swapped
```

This can lead to an incorrect PINN solution, even if the code runs without errors.

In the corrected implementation, the input bounds should be approximately:

```text
Lower bound: [-1.  0.]
Upper bound: [ 1.  1.]
```

This confirms that the first column is (x) and the second column is (t).

---

## 11. Why This Example Is Useful

The Burgers equation is simple enough to understand, but rich enough to demonstrate important PINN concepts:

* nonlinear convection
* diffusion
* automatic differentiation
* PDE residual construction
* initial and boundary condition enforcement
* collocation-point training
* physics-constrained learning

This makes it a good first problem before applying PINNs to more complex systems such as:

* Navier-Stokes equations
* multiphase flows
* heat transfer problems
* boiling simulations
* porous-media flow
* digital-twin surrogate models

---

## 12. Repository Structure

```text
PINN-Burgers-Equation-/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
└── notebooks/
    └── burgers_pinn_tensorflow.ipynb
```

---

## 13. Author

**Rohit Singh Gulia**

Research interests:

* Computational Fluid Dynamics
* Multiphase Flow
* Physics-Informed Machine Learning
* Scientific Machine Learning
* Digital Twin Modelling
* OpenFOAM and Python-based Simulation Workflows

---

## 14. Keywords

```text
Physics-Informed Neural Network
PINN
Burgers Equation
Scientific Machine Learning
TensorFlow
Automatic Differentiation
Computational Fluid Dynamics
Machine Learning
Partial Differential Equations
```
# PINN-Burgers-Equation-
TensorFlow implementation of a Physics-Informed Neural Network for solving the 1D viscous Burgers equation.
