# Research Basis

The implementation choices were checked against the following primary papers,
official project repositories, and challenge resources. These sources are design
references; no pretrained weights or external training data are used.

## Point and mesh backbones

1. [PointNet++](https://arxiv.org/abs/1706.02413)
2. [PointNeXt](https://arxiv.org/abs/2206.04670)
3. [Official PointNeXt implementation](https://github.com/guochengqian/PointNeXt)
4. [Official PointNeXt-S configuration](https://raw.githubusercontent.com/guochengqian/PointNeXt/master/cfgs/modelnet40ply2048/pointnext-s.yaml)
5. [DGCNN](https://arxiv.org/abs/1801.07829)
6. [Official DGCNN implementation](https://github.com/WangYueFt/dgcnn)
7. [KPConv](https://arxiv.org/abs/1904.08889)
8. [Official KPConv implementation](https://github.com/HuguesTHOMAS/KPConv)
9. [Point Transformer](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)
10. [MeshNet](https://arxiv.org/abs/1811.11424)
11. [Official MeshNet implementation](https://github.com/iMoonLab/MeshNet)
12. [MeshCNN](https://arxiv.org/abs/1809.05910)
13. [Official MeshCNN implementation](https://github.com/ranahanocka/MeshCNN)

These comparisons motivated retaining PointNet++ as the baseline, implementing a
portable C32/B0-style PointNeXt candidate, and placing mesh methods behind a strict
all-crop connectivity and simplification gate.

## Landmark localization and refinement

14. [HPoint103 / Dual Cascade Point Transformer](https://arxiv.org/abs/2401.07251)
15. [3D landmark heatmap GCN](https://ojs.aaai.org/index.php/AAAI/article/view/20161)
16. [PAL-Net](https://github.com/Ali5hadman/PAL-Net-A-Point-Wise-CNN-with-Patch-Attention)
17. [Structure-Aware LSTM](https://pubmed.ncbi.nlm.nih.gov/35130151/)
18. [CHaRNet and CHaRM](https://arxiv.org/abs/2501.13073)
19. [nnLandmark](https://proceedings.mlr.press/v315/ertl26a.html)

These methods support the coarse-to-fine locator/regressor split, ordered structural
heads, and the optional local KNN offset-refinement experiment.

## Shape correspondence, surface constraints, and ear context

20. [Point2SSM](https://arxiv.org/abs/2305.14486)
21. [Point2SSM++](https://arxiv.org/abs/2405.09707)
22. [PyTorch3D geometric loss definitions](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html)
23. [HRTF individualization from 3D-head anthropometry](https://air.unimi.it/retrieve/dfa8b9aa-2484-748b-e053-3a05fe0a3a96/IEEE-HRTF_Individualization_Based_on_Anthropometric_Measurements_Extracted_from_3D_Head_Meshes.pdf)
24. [AudioEar](https://arxiv.org/abs/2301.12613)
25. [York Ear Model](https://www-users.york.ac.uk/~np7/research/YEM/)
26. [Official Tech Arena topic description](https://huawei.agorize.com/en/challenges/2026-munich-tech-arena/pages/topic-description?lang=en)

These sources motivated explicit correspondence-aware losses, predicted-to-surface
regularization, exact optional surface projection, and strict preservation of the
challenge's coordinate and ordered-output contract.
