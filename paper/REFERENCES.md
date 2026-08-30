# References

Keys match `references.bib`. Unmarked entries are verified against the arXiv/publisher page; *check* = cited from memory, confirm before submission; *TODO* = information only Marco has.

## Core method

1. V. Kurtz, A. Castro. [CENIC: Convex Error-controlled Numerical Integration for Contact](https://arxiv.org/abs/2511.08771). arXiv:2511.08771, 2025. `cenic`
2. A. M. Castro, X. Han, J. Masterjohn. [Irrotational Contact Fields](https://arxiv.org/abs/2312.03908). arXiv:2312.03908, 2023. `icf`
3. A. Castro, F. Permenter, X. Han. [An Unconstrained Convex Formulation of Compliant Contact](https://arxiv.org/abs/2110.10107). IEEE T-RO 39(2), 2023. `sap`

## Contact models and dissipation

4. E. Todorov. Convex and analytically-invertible dynamics with contacts and constraints: theory and implementation in MuJoCo. ICRA 2014. `todorov2014` — *check*
5. E. Todorov, T. Erez, Y. Tassa. MuJoCo: A physics engine for model-based control. IROS 2012. `mujoco` — *check*
6. [MuJoCo documentation, solver parameters](https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters). `mujoco_docs`
7. K. H. Hunt, F. R. E. Crossley. Coefficient of restitution interpreted as damping in vibroimpact. J. Appl. Mech. 42(2), 1975. `huntcrossley` — *check*
8. J. Masterjohn, D. Guoy, J. Shepherd, A. Castro. [Velocity Level Approximation of Pressure Field Contact Patches](https://arxiv.org/abs/2110.04157). IEEE RA-L, 2022. `hydroelastic`
9. Q. Le Lidec, W. Jallet, L. Montaut, I. Laptev, C. Schmid, J. Carpentier. [Contact Models in Robotics: a Comparative Analysis](https://arxiv.org/abs/2304.06372). IEEE T-RO, 2024. `lelidec2024`
10. T. A. Howell et al. [Dojo: A Differentiable Physics Engine for Robotics](https://arxiv.org/abs/2203.00806). arXiv:2203.00806, 2022. `dojo`

## Error-controlled integration

11. E. Hairer, G. Wanner. Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems. Springer, 1996. `hairer` — *check*
12. R. Tedrake and the Drake Development Team. [Drake: Model-based design and verification for robotics](https://drake.mit.edu). 2019. `drake` — *check*
13. [Drake issue #14694](https://github.com/RobotLocomotion/drake/issues/14694). `drake14694` — *TODO: confirm number and title*

## Benchmarking physics engines

14. T. Erez, Y. Tassa, E. Todorov. Simulation tools for model-based robotics: comparison of Bullet, Havok, MuJoCo, ODE and PhysX. ICRA 2015. `erez2015` — *check*
15. D. Kang, J. Hwangbo. [SimBenchmark](https://leggedrobotics.github.io/SimBenchmark/). 2018. `simbenchmark` — *check*
16. B. Acosta, W. Yang, M. Posa. [Validating Robotics Simulators on Real-World Impacts](https://arxiv.org/abs/2110.00541). IEEE RA-L, 2022. `acosta2022`

## GPU simulation and robot learning

17. V. Makoviychuk et al. [Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470). arXiv:2108.10470, 2021. `isaacgym`
18. M. Mittal et al. [Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning](https://arxiv.org/abs/2511.04831). arXiv:2511.04831, 2025. `isaaclab`
19. M. Mittal et al. Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments. IEEE RA-L 8(6), 2023. `orbit` — *check*
20. N. Rudin, D. Hoeller, P. Reist, M. Hutter. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. CoRL 2021. `rudin2021` — *check*
21. [rsl_rl](https://github.com/leggedrobotics/rsl_rl). leggedrobotics, GitHub. `rsl_rl`
22. [Newton](https://github.com/newton-physics/newton). Newton Physics, GitHub. `newton`
23. [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp). Google DeepMind, GitHub. `mujoco_warp`
24. M. Macklin. [Warp: A high-performance Python framework for GPU simulation and graphics](https://github.com/NVIDIA/warp). NVIDIA GTC 2022. `warp` — *check*
25. Y. Narang et al. [Factory: Fast Contact for Robotic Assembly](https://arxiv.org/abs/2205.03532). RSS 2022. `factory`
26. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017. `ppo` — *check*

## Policies exploiting simulator artifacts

27. J. Lehman et al. [The Surprising Creativity of Digital Evolution](https://arxiv.org/abs/1803.03453). arXiv:1803.03453, 2018. `lehman2018`
28. N. Cheney, R. MacCurdy, J. Clune, H. Lipson. Unshackling evolution: evolving soft robots with multiple materials and a powerful generative encoding. GECCO 2013. `cheney2013` — *check*

## Curriculum and reward machinery

29. C. Florensa, D. Held, M. Wulfmeier, M. Zhang, P. Abbeel. [Reverse Curriculum Generation for Reinforcement Learning](https://arxiv.org/abs/1707.05300). CoRL 2017. `florensa2017`
30. D. Pavlichenko, S. Behnke. [Deep Reinforcement Learning of Dexterous Pre-Grasp Manipulation for Human-Like Functional Categorical Grasping](https://www.ais.uni-bonn.de/papers/CASE_2023_Pavlichenko.pdf). IEEE CASE 2023. `pavlichenko2023`
31. H. Yuan et al. [DemoGrasp: Universal Dexterous Grasping from a Single Demonstration](https://arxiv.org/abs/2509.22149). arXiv:2509.22149, 2025. `demograsp`
32. dexsuite, the PI's fork. `dexsuite` — *TODO: URL and citation form*

## Smooth policies and hardware

33. E. Aljalbout, F. Frank, M. Karl, P. van der Smagt. [On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer](https://arxiv.org/abs/2312.03673). arXiv:2312.03673, 2023. `aljalbout2023`
34. S. Mysore, B. Mabsout, R. Mancuso, K. Saenko. [Regularizing Action Policies for Smooth Control with Reinforcement Learning](https://arxiv.org/abs/2012.06644). ICRA 2021. `caps`
35. Z. Chen et al. [Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies](https://arxiv.org/abs/2410.11825). arXiv:2410.11825, 2024. `lcp`
36. [Trossen AI documentation](https://docs.trossenrobotics.com). Trossen Robotics. `trossen`
