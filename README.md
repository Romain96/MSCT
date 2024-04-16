# Multi-Scale Component-Tree

The __Multi-Scale Component-Tree__ [1] (__MSCT__ for short) is an extension of the concept of _component-tree_ [2] applied to multiple scales of the same image.
It has been presented in the DGMM 2024 conference.
The slides are available [here](Perrin_DGMM_2024.pdf).

Instead of building a component-tree on the original image, a set of downsampled images is build using a max-pooling operation.
A component-tree, called the _base component-tree_, is built on the smallest downsampled image.
At each step of the enrichment path, the MSCT _G(i)_ of maximum scale _i_ built on _f(i)_ is filtered to extract a subset of nodes of interest.
From each of these subtrees, the corresponding flatzone _fz(i)_ is projected and upscaled to fit in the higher resolution image _f(i-1)_.
A local component-tree is then built using this projected upscaled flatzone _fz(i-1)_ as a mask, thus yielding to the computation of a component-tree on _f(i-1)_ union _fz(i-1)_.
All local hierarchies at higher scale (the partial component-trees computed previously) are then merged into the MSCT at scale i _G(i)_ enriching it to become the MSCT _G(i-1)_ of highest scale _i-1_.
At the final iteration, we obtain _G(0)_ the MSCT in which the highest amount of detail is at the original scale of the image _f(0)_.

---
[1] R. Perrin, A. Leborgne, N. Passat, B. Naegel, C. Wemmert, _Multi-Scale Component-Tree: An Hierarchical Representation of Sparse Objects_, IAPR Third International Conference on Discrete Geometry and Mathematical Morphology (DGMM 2024), Florence, Italy, april 2024.

[2] P. Salembier, A. Oliveras, L. Garrido, _Anti-extensive connected operators for image and sequence processing_, IEEE Transactions on Image Processing, vol. 7, pp. 555-570, 1998.
