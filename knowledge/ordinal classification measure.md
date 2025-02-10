Measures for ordinal classification

# 1. Defination

## 1.1. Notations

- $\{ V_1, V_2, \cdots, V_K \}, V_1 < V_2 \cdots < V_K$ : oridinal class labels
- $N$: sample size
- $p_k(x_n) = \Pr(Y_n = V_k | X_n) = \Pr(V_{k-1} < Y \leq V_k | X_n)$ : the probability of the sample point $x_n$ belongs to class $V_k$
- $\tilde{p}_k(x_n) = \sum_{j=1}^{k} \Pr(Y_n = V_j | X_n) = \Pr(Y \leq V_k | X_n)$ : the cumulative probability of $p_k(x_n)$, indicates the probability of the sample point $x_n$ belongs to the class $V_k$ or less.

## 1.2. Metrics

### (1) Ranked probability score (RPS)

- RPS is a generalization of the *Brier score* to multiple categories.
- The *predicted cumulative distribution function*, $\tilde{p}_k$, can be computed from class probabilities that are predicted by a model, that is the estimated probabilities of an observation belonging to classes ${k}$.
- The *true cumulative distribution function*, $I(Y_n \leq V_k)$
  , simplifies to a step function with a step from 0 to 1 at the true class value $Y_i \ ( \ Y_i \in \{ V_1, V_2, \cdots, V_K \} \ )$ for observation $x_i$.
- The RPS uses solely the ordering of the categories and does not require information on the distances between categories.
- A **smaller** RPS score represents a **better** prediction.

$$
RPS = \frac{1}{N} 
\sum_{i=1}^{N} \sum_{k=1}^{K}
\left[ \tilde{p}_k({x_n}) - I(Y_n \leq V_k) \right]^2
$$

- where $\sum_{k=1}^{K} \left| \tilde{p}_k({x_n}) - I(Y_n \leq k) \right|$ represent the area between the cumulative predicted and the cumulative true distribution functions

### (2) Concordance index (C-index)

- The C-index is computed as the proportion of concordant pairs among the comparable pairs.
- A higher C-index indicated a better performance.
- C-index varies between 0 and 1.
- When there are only two different class labels, the C-index reduces to the area under the Receiver Operating Characteristic (ROC) curve

$$
\text{C-index}
$$

### (3) Spearman's rank correlation coefficient

- *Spearman's rank correlation coefficient* is a nonparametric measure of rank correlation
- varies in the range between -1 and 1



$$
\rho = 1 - \frac{6 \sum_{i=1}^N d_i^2 }{N(N-1)}
$$

- where $d_i = \text{rank}(\hat{y}_i) - \text{rank}(\hat{y})$ is the difference between the two ranks of the predicted and true labels of the sample

# Reference

Epstein, E.S., 1969. A scoring system for probability forecasts of ranked categories. J. Appl. Meteorol. 8 (6), 985–987, doi: [10.1175/1520-0450(1969)008[0985:ASSFPF](0985:ASSFPF)2.0.CO;2](https://doi.org/10.1175/1520-0450(1969)008[0985:ASSFPF](0985:ASSFPF)2.0.CO;2)

Janitza, S., Tutz, G. and Boulesteix, A.L., 2016. Random forest for ordinal responses: prediction and variable selection. *Computational Statistics & Data Analysis*, 96, pp.57-73, doi: [10.1016/j.csda.2015.10.005](https://doi.org/10.1016/j.csda.2015.10.005)

Tang, M., Perez-Fernandez, R. and De Baets, B., 2021. A comparative study of machine learning methods for ordinal classification with absolute and relative information. Knowledge-Based Systems, 230, p.107358. doi: [10.1016/j.knosys.2021.107358](https://doi.org/10.1016/j.knosys.2021.107358)

Tutz, G., 2022, Ordinal Trees and Random Forests: Score-Free Recursive Partitioning and Improved Ensembles. *J Classif*, 39, pp.241–263, doi: [10.1007/s00357-021-09406-4](https://doi.org/10.1007/s00357-021-09406-4)

Waegeman, W., De Baets, B. and Boullart, L., 2008. ROC analysis in ordinal regression learning. Pattern Recognition Letters, 29(1), pp.1-9, doi: [10.1016/j.patrec.2007.07.019](https://doi.org/10.1016/j.patrec.2007.07.019)

