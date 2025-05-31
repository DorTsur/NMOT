# Neural Entropic Multimarginal Optimal Transport (NEMOT)

Implementation of "Neural Estimation for Scaling Entropic Multimarginal Optimal Transport" by Dor Tsur, Ziv Goldfeld, Kristjan Greenewald, and Haim Permuter.

## Overview

Entropic Multimarginal Optimal Transport (EMOT) handles interactions among multiple probability distributions but scales poorly with the number of marginals \(k\) and dataset size \(n\). NEMOT replaces the classical Sinkhorn solver with a mini-batch, neural-network–based estimator:

- **Neural potentials** parametrize the dual variables via feedforward networks.
- **Mini-batch training** reduces per-epoch cost from \(O(nk)\) to \(O(b\,k^2)\) (or \(O(b\,k)\) when using sparse cost graphs), where \(b \ll n\).  
- **Non-asymptotic bounds** guarantee accuracy of both the estimated cost and induced transport plan.  
- **Empirical speedups** (especially when \(k>2\) or \(n \gg b\)) relative to classical multimarginal Sinkhorn.

This repository provides a reference Python implementation of NEMOT (for full and sparse cost graphs) and an extension to Multimarginal Gromov–Wasserstein (NEMGW).

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/DorTsur/NMOT.git
   cd NMOT

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
