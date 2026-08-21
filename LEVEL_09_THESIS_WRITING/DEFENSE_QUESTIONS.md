# Likely defense questions

1. How does your work differ from Huang et al.'s 2019 causal state-space forecasting paper?
2. Why is an observed predictive lagged coefficient not automatically a causal effect?
3. What assumptions justify edge direction in your generator and real-world data?
4. Why can a lagged causal graph have a cycle after collapsing time indices?
5. Why is the standard NOTEARS acyclicity penalty not a generic `O(D²)` operation?
6. Is your complexity linear in context length, number of variables, or both?
7. Why is online VAR a necessary baseline for a graph-conditioned state-space model?
8. How do you know the graph is helping rather than merely adding parameters or adaptation?
9. What happens when the estimated graph is missing, reversed, dense, or confounded?
10. How do you prevent the current target from leaking into its own prediction?
11. Why can you report graph SHD on synthetic data but not ordinary ETT observations?
12. How do you distinguish a true mechanism change from a noise-scale change?
13. What does the Page-Hinkley detector actually observe?
14. What is the difference between graph uncertainty and forecast noise?
15. How would you evaluate probabilistic coverage after a change point?
16. Why does your implementation remain stable when graph coefficients change?
17. How would you scale dense `D²` graph operations to hundreds of variables?
18. Which part of the proposed method is genuinely new or empirically informative?
19. What negative result would falsify your main hypothesis?
20. How would you improve the method if given intervention data or known physical graph structure?
