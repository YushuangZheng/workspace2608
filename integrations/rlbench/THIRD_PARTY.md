# Third-Party Components

The RLBench experiments require the following external components. They are not distributed with this repository.

| Component | Revision | Purpose |
|---|---|---|
| `vonHartz/RLBench` | `tapas@a51b4e609dc5c3e1a8c06046bd87a9da24723da4` | Dual-Panda environment and tasks |
| `robot-learning-freiburg/TAPAS` | `52e35214b9baa7b190b87196c36b9e98f4006149` | Segmentation and resampling reference |
| `vonHartz/PyRep` | `b8bd1d7a3182adcd570d001649c0849047ebf197` | CoppeliaSim Python interface |
| `chenhaox/pytracik` | `v0.0.3@8c8fd2d8ca70334af9b747987f1156ebb1da25cc` plus the repository's bounded-Cartesian text patch | ROS-free TRAC-IK Distance fallback |
| CoppeliaSim Edu 4.1 | Ubuntu 20.04 build | Simulator |

Use each component under its upstream license. Demonstration pickles, camera observations, simulator assets, trained checkpoints, and evaluation outputs should be distributed only when their respective licenses permit it.

The pytracik source is MIT licensed; its notice is retained at
`third_party/pytracik/LICENSE`.  No pytracik shared object or wheel is checked
in.  `build_pytracik_bounded.sh` downloads the exact source archive, verifies
its digest, applies the reviewable text patch, and installs into a caller-owned
isolated prefix.  The native extension also links against system Boost,
Eigen3, Orocos KDL, and NLopt packages, whose licenses and ABI compatibility
remain the deployer's responsibility.
