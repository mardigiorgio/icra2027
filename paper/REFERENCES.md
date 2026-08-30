# References for the CENIC-on-GPU paper

Status key: **V** = title/authors/year verified this session against the arXiv or publisher page; **M** = from memory of a well-known work, check the exact citation before submission; **TODO** = missing information only you have. "Where" = the paper section that leans on it (I = Intro, II = Background, III = Method, IV = Results I, V = Results II, VI = IRL). Every entry has a key in `references.bib`.

## Core method
| key | reference | where | status |
|---|---|---|---|
| cenic | Kurtz, V., Castro, A. **CENIC: Convex Error-controlled Numerical Integration for Contact.** arXiv:2511.08771, 2025. https://arxiv.org/abs/2511.08771 | everywhere; test cases (IV), constants (III), Fig. 8/10 layouts | V (title from the PDF) |
| icf | Castro, A. M., Han, X., Masterjohn, J. **Irrotational Contact Fields.** arXiv:2312.03908, 2023 (rev. 2025). https://arxiv.org/abs/2312.03908 | II (contact model, Lagged friction, gliding offset), IV-stiffness (Fig. 18 protocol) | V |
| sap | Castro, A., Permenter, F., Han, X. **An Unconstrained Convex Formulation of Compliant Contact.** IEEE T-RO 39(2), 2023; arXiv:2110.10107. https://arxiv.org/abs/2110.10107 | II (convex time stepping, closed-form resting penetration) | V (arXiv); journal volume/pages M |

## Contact models, MuJoCo, dissipation
| key | reference | where | status |
|---|---|---|---|
| todorov2014 | Todorov, E. **Convex and analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo.** ICRA 2014. | II (soft constraints, solref/solimp) | M |
| mujoco | Todorov, E., Erez, T., Tassa, Y. **MuJoCo: A physics engine for model-based control.** IROS 2012. | I, II | M |
| mujoco_docs | MuJoCo documentation, *Computation → Solver parameters* (solref, solimp, refsafe). https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters | II, Appendix F formulas (matched to mujoco_warp constraint.py) | V (the code) |
| huntcrossley | Hunt, K. H., Crossley, F. R. E. **Coefficient of restitution interpreted as damping in vibroimpact.** J. Appl. Mech. 42(2):440–445, 1975. | II (dissipation model), IV (the d = 1 s/m assumption) | M |
| hydroelastic | Masterjohn, J., Guoy, D., Shepherd, J., Castro, A. **Velocity Level Approximation of Pressure Field Contact Patches.** arXiv:2110.04157, 2021 (RA-L 2022). https://arxiv.org/abs/2110.04157 | II (why hydroelastic cases are out of scope) | V |
| lelidec2024 | Le Lidec, Q., Jallet, W., Montaut, L., Laptev, I., Schmid, C., Carpentier, J. **Contact Models in Robotics: a Comparative Analysis.** arXiv:2304.06372, 2023 (T-RO 2024). https://arxiv.org/abs/2304.06372 | II, IV-consistency (short re-anchored windows) | V |
| dojo | Howell, T. A., Le Cleac'h, S., Brüdigam, J., Chen, Q., Sun, J., Kolter, J. Z., Schwager, M., Manchester, Z. **Dojo: A Differentiable Physics Engine for Robotics.** arXiv:2203.00806, 2022. https://arxiv.org/abs/2203.00806 | II (penetration reported in mm) | V |

## Error-controlled integration and reporting
| key | reference | where | status |
|---|---|---|---|
| hairer | Hairer, E., Wanner, G. **Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems.** Springer, 1996. | II (step control, work-precision diagrams) | M |
| drake | Tedrake, R. and the Drake Development Team. **Drake: Model-based design and verification for robotics.** 2019. https://drake.mit.edu | II, III (the step-size controller constants: Drake's IntegratorBase) | M (URL V) |
| drake14694 | Drake issue #14694 (light body under high gain). https://github.com/RobotLocomotion/drake/issues/14694 | IV-actuated design | TODO: confirm the issue number/title before citing |

## Benchmarking physics engines
| key | reference | where | status |
|---|---|---|---|
| erez2015 | Erez, T., Tassa, Y., Todorov, E. **Simulation tools for model-based robotics: Comparison of Bullet, Havok, MuJoCo, ODE and PhysX.** ICRA 2015. | IV-consistency (the consistency test), momentum drift | M |
| simbenchmark | Kang, D., Hwangbo, J. **SimBenchmark.** 2018. https://leggedrobotics.github.io/SimBenchmark/ | IV (determinism, momentum) | M (URL) |
| acosta2022 | Acosta, B., Yang, W., Posa, M. **Validating Robotics Simulators on Real-World Impacts.** arXiv:2110.00541, 2021 (RA-L/IROS 2022). https://arxiv.org/abs/2110.00541 | II, IV ("decreasing the timestep further did not improve prediction") | V |

## GPU simulation and robot learning
| key | reference | where | status |
|---|---|---|---|
| isaacgym | Makoviychuk, V., et al. **Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning.** arXiv:2108.10470, 2021. https://arxiv.org/abs/2108.10470 | I, II | V |
| isaaclab | Mittal, M., et al. **Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning.** arXiv:2511.04831, 2025. https://arxiv.org/abs/2511.04831 | I, V (platform) | V |
| orbit | Mittal, M., et al. **Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments.** IEEE RA-L 8(6), 2023. arXiv:2301.04195 | V (platform lineage) | M |
| rudin2021 | Rudin, N., Hoeller, D., Reist, P., Hutter, M. **Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning.** CoRL 2021; arXiv:2109.11978. | V (rsl_rl's paper) | M |
| rsl_rl | leggedrobotics/rsl_rl (PPO implementation used for training). https://github.com/leggedrobotics/rsl_rl | V | V (URL) |
| newton | Newton Physics. **Newton: a GPU-accelerated physics simulation engine built on NVIDIA Warp.** https://github.com/newton-physics/newton | I, III | V (URL) |
| mujoco_warp | Google DeepMind. **MuJoCo Warp.** https://github.com/google-deepmind/mujoco_warp | III (the MuJoCo arm) | V (URL) |
| warp | Macklin, M. **Warp: A high-performance Python framework for GPU simulation and graphics.** NVIDIA GTC 2022. https://github.com/NVIDIA/warp | III | M |
| factory | Narang, Y., et al. **Factory: Fast Contact for Robotic Assembly.** RSS 2022; arXiv:2205.03532. https://arxiv.org/abs/2205.03532 | I (RL "exploits any inaccuracies") | V |
| ppo | Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. **Proximal Policy Optimization Algorithms.** arXiv:1707.06347, 2017. | V | M |

## Policies exploiting simulator artifacts
| key | reference | where | status |
|---|---|---|---|
| lehman2018 | Lehman, J., Clune, J., Misevic, D., et al. **The Surprising Creativity of Digital Evolution.** arXiv:1803.03453, 2018. https://arxiv.org/abs/1803.03453 | I (motivation) | V |
| cheney2013 | Cheney, N., MacCurdy, R., Clune, J., Lipson, H. **Unshackling evolution: evolving soft robots with multiple materials and a powerful generative encoding.** GECCO 2013. | I (creatures gaming an adaptive-δt heuristic) | M |

## Approach and curriculum (Part 2)
| key | reference | where | status |
|---|---|---|---|
| florensa2017 | Florensa, C., Held, D., Wulfmeier, M., Zhang, M., Abbeel, P. **Reverse Curriculum Generation for Reinforcement Learning.** CoRL 2017; arXiv:1707.05300. https://arxiv.org/abs/1707.05300 | V (grasp bank / reverse curriculum) | V |
| pavlichenko2023 | Pavlichenko, D., Behnke, S. **Deep Reinforcement Learning of Dexterous Pre-Grasp Manipulation for Human-Like Functional Categorical Grasping.** IEEE CASE 2023. https://www.ais.uni-bonn.de/papers/CASE_2023_Pavlichenko.pdf | V (pre-grasp) | V |

## Reward machinery (Part 2)
| key | reference | where | status |
|---|---|---|---|
| demograsp | Yuan, H., Huang, Z., Wang, Y., Mao, C., Xu, C., Lu, Z. **DemoGrasp: Universal Dexterous Grasping from a Single Demonstration.** arXiv:2509.22149, 2025. https://arxiv.org/abs/2509.22149 | V | V |
| dexsuite | dexsuite / the PI's fork (progress ratchet, contact gating, cross-solver evaluation). | V | TODO: repository URL and citation form from you |

## Smooth policies and hardware (VI)
| key | reference | where | status |
|---|---|---|---|
| aljalbout2023 | Aljalbout, E., Frank, F., Karl, M., van der Smagt, P. **On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer.** arXiv:2312.03673, 2023. https://arxiv.org/abs/2312.03673 | VI (action parameterization) | V |
| caps | Mysore, S., Mabsout, B., Mancuso, R., Saenko, K. **Regularizing Action Policies for Smooth Control with Reinforcement Learning.** ICRA 2021; arXiv:2012.06644. https://arxiv.org/abs/2012.06644 | VI (CAPS) | V |
| lcp | Chen, Z., et al. **Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies.** arXiv:2410.11825, 2024. https://arxiv.org/abs/2410.11825 | VI (gradient penalty in rsl_rl) | V |
| trossen | Trossen Robotics, **Trossen AI / WidowX AI documentation** (control modes, goal_time, effort limits). https://docs.trossenrobotics.com | VI | V (URL) |
