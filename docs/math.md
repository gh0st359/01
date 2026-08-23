# Mathematical formulation

## Core

\[
x_{t+\Delta t}=F(x_t, p_t, m_t, \hat{w}_t, b_t, \mu_t, s_t, u_t)
\]

Implemented as a three-timescale GRU with fused readout.

## Workspace score

\[
s(c)=w^\top \phi(c)+\lambda^\top \phi(c)
\]

where \(\phi\) is salience, novelty, relevance, uncertainty, motivational weight, goal relevance, recency. Top-\(K=7\) winners are softmax-mixed into a broadcast vector. Ignition is the gap between the top two weights.

## World model

\[
z_t=\mathrm{enc}(p_t),\quad \hat{z}_{t+1}=\mathrm{GRU}([z_t;a_t], z_t),\quad \mathcal{L}=\|\hat{z}_{t+1}-z_{t+1}\|^2
\]

## Intrinsic value

Adaptive mix of prediction error, learning progress (declining error), novelty (familiarity reconstruction), uncertainty reduction, competence gain, social drive, and contradiction, gated by energy and rest need.

## Language

Intent \(I=(\alpha, v, R)\) is formed first. Surface form is \(\mathrm{arg\,sim}(v, \mathrm{lexicon})\) with learned construction counts. No \(P(\mathrm{token}_t\mid \mathrm{token}_{<t})\) language model is trained on internet text.
