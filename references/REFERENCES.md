# References

Grouped by where they are used. Papers are cited, not redistributed.

## The material and the constitutive model

- P. Sedlák, M. Frost, B. Benešová, T. Ben Zineb, P. Šittner (2012). Thermomechanical model for NiTi-based shape
  memory alloys including R-phase and material anisotropy under multi-axial loadings. *International Journal of
  Plasticity* 39, 132–151. [doi:10.1016/j.ijplas.2012.06.008](https://doi.org/10.1016/j.ijplas.2012.06.008).
  **The model this project starts from.**
- M. Frost, B. Benešová, H. Seiner, M. Kružík, P. Šittner, P. Sedlák (2020). Thermomechanical model for
  NiTi-based shape memory alloys covering macroscopic localization of martensitic transformation.
  [arXiv:1912.01356](https://arxiv.org/abs/1912.01356). States that the energy is convex in the volume fraction
  and the inelastic strain $\xi\varepsilon^{\mathrm{tr}}$, and not in $(\xi, \varepsilon^{\mathrm{tr}})$.
- A. Moskovka, M. Horák, J. Valdman, M. Knapek, M. Janeček, P. Sedlák, M. Frost (2025). Finite-strain
  constitutive model for shape memory alloys formulated in the logarithmic strain space.
  [arXiv:2504.16629](https://arxiv.org/abs/2504.16629). Uses Nelder–Mead for the state update and names bundle
  and non-smooth Newton methods as future work.
- B. Halphen, Q. S. Nguyen (1975). Sur les matériaux standards généralisés. *Journal de Mécanique* 14, 39–63.
  The framework: a free energy and a dissipation potential define the material.
- K. Otsuka, X. Ren (2005). Physical metallurgy of Ti–Ni-based shape memory alloys. *Progress in Materials
  Science* 50(5), 511–678. [doi:10.1016/j.pmatsci.2004.10.001](https://doi.org/10.1016/j.pmatsci.2004.10.001).
- D. C. Lagoudas, ed. (2008). *Shape Memory Alloys: Modeling and Engineering Applications*. Springer.

## The reference solver

- J. A. Nelder, R. Mead (1965). A simplex method for function minimization. *The Computer Journal* 7(4), 308–313.
  [doi:10.1093/comjnl/7.4.308](https://doi.org/10.1093/comjnl/7.4.308).
- K. I. M. McKinnon (1998). Convergence of the Nelder–Mead simplex method to a nonstationary point. *SIAM Journal
  on Optimization* 9(1), 148–158. A smooth, strictly convex function of two variables on which the method fails.
- J. C. Lagarias, J. A. Reeds, M. H. Wright, P. E. Wright (1998). Convergence properties of the Nelder–Mead
  simplex method in low dimensions. *SIAM Journal on Optimization* 9(1), 112–147.

## Neural surrogates of path-dependent materials (steps 01 to 02)

- M. Mozaffar, R. Bostanabad, W. Chen, K. Ehmann, J. Cao, M. A. Bessa (2019). Deep learning predicts
  path-dependent plasticity. *PNAS* 116(52), 26414–26420. [doi:10.1073/pnas.1911815116](https://doi.org/10.1073/pnas.1911815116).
- F. Masi, I. Stefanou, P. Vannucci, V. Maffi-Berthier (2021). Thermodynamics-based artificial neural networks
  for constitutive modeling. *Journal of the Mechanics and Physics of Solids* 147, 104277.
  [doi:10.1016/j.jmps.2020.104277](https://doi.org/10.1016/j.jmps.2020.104277).
- C. Bonatti, D. Mohr (2022). On the importance of self-consistency in recurrent neural network models
  representing elasto-plastic solids. *Journal of the Mechanics and Physics of Solids* 158, 104697.
  [doi:10.1016/j.jmps.2021.104697](https://doi.org/10.1016/j.jmps.2021.104697).
- S. Ross, G. J. Gordon, J. A. Bagnell (2011). A reduction of imitation learning and structured prediction to
  no-regret online learning. *AISTATS*, 627–635. Exposure bias and DAgger.
- S. Bengio, O. Vinyals, N. Jaitly, N. Shazeer (2015). Scheduled sampling for sequence prediction with recurrent
  neural networks. *Advances in Neural Information Processing Systems* 28.
- K. He, X. Zhang, S. Ren, J. Sun (2016). Deep residual learning for image recognition. *CVPR*, 770–778.
- P. Ramachandran, B. Zoph, Q. V. Le (2017). Searching for activation functions.
  [arXiv:1710.05941](https://arxiv.org/abs/1710.05941).
- D. P. Kingma, J. Ba (2015). Adam: a method for stochastic optimization. *ICLR*.
- P. J. Werbos (1990). Backpropagation through time: what it does and how to do it. *Proceedings of the IEEE*
  78(10), 1550–1560.
- T. Miyato, T. Kataoka, M. Koyama, Y. Yoshida (2018). Spectral normalization for generative adversarial
  networks. *ICLR*.

## Architectures, committees and data collection (step 03)

- B. Settles (2009). Active learning literature survey. Computer Sciences Technical Report 1648, University of
  Wisconsin–Madison.
- H. S. Seung, M. Opper, H. Sompolinsky (1992). Query by committee. *COLT*, 287–294.
- B. Lakshminarayanan, A. Pritzel, C. Blundell (2017). Simple and scalable predictive uncertainty estimation
  using deep ensembles. *Advances in Neural Information Processing Systems* 30.

## Proximal methods (step 05) and algorithm unrolling (planned)

- N. Parikh, S. Boyd (2014). Proximal algorithms. *Foundations and Trends in Optimization* 1(3), 127–239.
- J.-J. Moreau (1965). Proximité et dualité dans un espace hilbertien. *Bulletin de la Société Mathématique de
  France* 93, 273–299. The proximal map.
- I. Daubechies, M. Defrise, C. De Mol (2004). An iterative thresholding algorithm for linear inverse problems
  with a sparsity constraint. *Communications on Pure and Applied Mathematics* 57(11), 1413–1457. ISTA.
- J. D. Lee, Y. Sun, M. A. Saunders (2014). Proximal Newton-type methods for minimizing composite functions.
  *SIAM Journal on Optimization* 24(3), 1420–1443.
- A. Beck, M. Teboulle (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems.
  *SIAM Journal on Imaging Sciences* 2(1), 183–202.
- K. Gregor, Y. LeCun (2010). Learning fast approximations of sparse coding. *ICML*. The first unrolled ISTA
  (LISTA).
- J. R. Hershey, J. Le Roux, F. Weninger (2014). Deep unfolding: model-based inspiration of novel deep
  architectures. [arXiv:1409.2574](https://arxiv.org/abs/1409.2574).
- Y. Yang, J. Sun, H. Li, Z. Xu (2016). Deep ADMM-Net for compressive sensing MRI. *Advances in Neural
  Information Processing Systems* 29.
- Y. Yang, J. Sun, H. Li, Z. Xu (2018). ADMM-CSNet: a deep learning approach for image compressive sensing.
  *IEEE Transactions on Pattern Analysis and Machine Intelligence* 42(3), 521–538.
- D. Ito, S. Takabe, T. Wadayama (2019). Trainable ISTA for sparse signal recovery. *IEEE Transactions on Signal
  Processing* 67(12), 3113–3125.
- V. Monga, Y. Li, Y. C. Eldar (2021). Algorithm unrolling: interpretable, efficient deep learning for signal and
  image processing. *IEEE Signal Processing Magazine* 38(2), 18–44. The survey.
- J. C. Simo, T. J. R. Hughes (1998). *Computational Inelasticity*. Springer. Return-mapping algorithms: the
  projection onto the elastic domain that the proximal step turns out to be.

