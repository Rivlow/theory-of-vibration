# Main Code Documentation

## General Structure

The main code is divided into two primary parts:

1. Computation of K, M, and eigenvalues/modes
2. Transient response

### Part 1: Computation of K, M, and eigenvalues/modes

This section initializes geometry, creates elements, assembles matrices, and calculates eigenvalues and modes.

```markdown
1. Initialize parameters and geometry
2. Create elements
3. Assemble matrices
4. Add lumped masses
5. Remove clamped nodes
6. Extract K and M matrices
7. Solve for frequencies and modes
8. Normalize eigenvectors
```

### Part 2: Transient Response

This section calculates the transient response of the system using various methods.

```markdown
1. Define simulation parameters
2. Calculate damping matrix
3. Compute damping ratios
4. Define period and time span
5. Compute applied force
6. Apply mode displacement method
7. Apply mode acceleration method
8. Analyze transient response
9. Apply Newmark integration algorithm
10. Perform FFT analysis of the response
```

## Main Mathematical Functions

### 1. Force Computation (computeForce)

This function calculates the force induced by supporters on excitation nodes.

Main equation:
$$F(t) = -\frac{A}{n_{nodes}} \sin(\omega t)$$

where:
- $A = 0.8 \frac{mv}{\Delta t}$
- $v = \sqrt{2gh}$
- $\omega = 2\pi f$

### 2. Impulse Response Calculation (computeH)

This function computes the impulse response for all modes.

Equation:
$$h(t) = \frac{1}{\omega_d} e^{-\epsilon \omega_r t} \sin(\omega_d t)$$

where:
- $\omega_d$: damped frequency
- $\omega_r$: natural frequency
- $\epsilon$: damping ratio

### 3. Calculation of eta, phi, and mu (etaPhiMu)

This function calculates eta (modal displacement), phi (modal force), and mu (modal mass).

Main equations:
- $\mu_r = \mathbf{x}_r^T \mathbf{M} \mathbf{x}_r$
- $\phi_r(t) = \frac{\mathbf{x}_r^T \mathbf{F}(t)}{\mu_r}$
- $\eta_r(t) = \int_0^t h_r(t-\tau) \phi_r(\tau) d\tau$

### 4. Mode Displacement Method (modeDisplacementMethod)

This method calculates displacement using modal superposition.

Equation:
$$\mathbf{q}(t) = \sum_{r=1}^{n_{modes}} \eta_r(t) \mathbf{x}_r$$

### 5. Mode Acceleration Method (modeAccelerationMethod)

This method calculates displacement using the mode acceleration method.

Equation:
$$\mathbf{q}_{acc}(t) = \sum_{r=1}^{n_{modes}} \eta_r(t) \mathbf{x}_r + \mathbf{K}^{-1}\mathbf{F}(t) - \sum_{r=1}^{n_{modes}} \frac{\phi_r(t)}{\omega_r^2} \mathbf{x}_r$$

### 6. Newmark Integration (NewmarkIntegration)

This function implements the Newmark integration algorithm to solve the equation of motion.

Main equations:
- $\mathbf{v}_{i+1} = \mathbf{v}_i + [(1-\gamma)h]\mathbf{a}_i + \gamma h \mathbf{a}_{i+1}$
- $\mathbf{x}_{i+1} = \mathbf{x}_i + h\mathbf{v}_i + [(0.5-\beta)h^2]\mathbf{a}_i + \beta h^2 \mathbf{a}_{i+1}$

where $\gamma$ and $\beta$ are parameters of the Newmark algorithm.

This documentation provides an overview of the main functions and equations used in the code. For a deeper understanding, it is recommended to examine each function in detail and consult the specific documentation for each library used (numpy, scipy, etc.).