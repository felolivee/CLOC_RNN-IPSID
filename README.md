# CLOC Subgroup: Implementing an RNN-IPSID


## Jupyter Notebook: `gordon_jupyter.pynb`

### Overview

This Jupyter notebook contains all the testers for the behavior predictor algorithms I developed, based on the paper *"Where is all the nonlinearity: flexible nonlinear modeling of behaviorally relevant neural dynamics using recurrent neural networks (RNN-PSID)"*. The algorithms tested in this notebook are:

- Linear RNN
- Nonlinear RNN
- Linear RNN-IPSID
- Nonlinear RNN-IPSID

### Purpose

The primary goal of these tests is to determine whether latent factors (X) control behavior. To establish a causal relationship, we need to control the latent factors (X) and observe the resultant effects on behavior.

### Conceptual Background

#### Latent State Dynamics

Latent state dynamics involve a state space model where the state variables are unobservable. This can be likened to a Hidden Markov Model (HMM) where the system state is hidden and can only be inferred through observed outputs.

- **Example 1: Pendulum Model**  
  In a state space model of a pendulum, we know the state variables (e.g., position) over time. We can calculate first-order derivatives of position and plot the change in the states’ positions throughout time.

- **Example 2: Neuronal Populations Model**  
  In a latent space model of neuronal populations, we do not know the state variables over time. Instead, we have neural observations (e.g., brain waves) and behavioral observations (e.g., motor firing). These observations are used to create a latent state dynamics model, which abstractly represents brain activity. For instance:
  - Latent states may move together and then diverge at a specific time point, indicating an event (e.g., reaching over) that caused the divergence.
  - To ascertain if latent factors (X) control behavior, we must control (X) and observe if there is a causal relationship between the two.
  - RNN-IPSID allows us to model the latent state dynamics and introduce inputs (e.g., optogenetic control) to observe how adding input affects both latent state dynamics and observable behavioral output.

### Conclusion

A latent state dynamics model is essentially a state space model, which can be considered a partially observed HMM. The tests in this notebook aim to explore the behavior of such models and their ability to predict and control behavior through latent factors.
