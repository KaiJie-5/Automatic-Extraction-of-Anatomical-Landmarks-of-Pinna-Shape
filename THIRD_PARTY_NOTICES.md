# Third-Party Notices

## yanx27/Pointnet_Pointnet2_pytorch

`src/pointnet2_utils.py` adapts PointNet++ set abstraction utilities from:

https://github.com/yanx27/Pointnet_Pointnet2_pytorch

The upstream project is distributed under the MIT License:

```text
MIT License

Copyright (c) 2019 benny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## PointNeXt design provenance

`src/pointnext_model.py` is a repository-native PyTorch implementation informed by
the PointNeXt C32/B0 architecture and configuration published at:

https://github.com/guochengqian/PointNeXt

No OpenPoints source code or compiled operators are copied into this repository.
PointNeXt is distributed under the MIT License. Copyright (c) 2022 Guocheng Qian.
The MIT grant and warranty disclaimer reproduced above apply to that upstream work.

## Pointcept/PointTransformerV3

`src/third_party/pointtransformerv3/` contains the official detached Point
Transformer V3 implementation and serialization functions from:

https://github.com/Pointcept/PointTransformerV3

Xiaoyang Wu et al., "Point Transformer V3: Simpler, Faster, Stronger,"
CVPR 2024, https://arxiv.org/abs/2312.10035.

The source is pinned to revision
`3229e9b7de1770c8ad17c316f8e349982de509f8`. The learned architecture is kept
unchanged. The vendored constructors have one integration-only modification:
they accept and forward an optional spconv algorithm argument. The adapter fixes
that argument to `spconv.ConvAlgo.Native` to avoid the upstream mixed-precision
evaluation tuner failure. Repository-specific voxel preparation, global max
pooling, and training-only serialization-order shuffling are implemented in
`src/pointtransformerv3_model.py` outside the vendored source.

Point Transformer V3 is distributed under the MIT License. Copyright (c) 2023
Pointcept. The complete upstream licence is retained at
`src/third_party/pointtransformerv3/LICENSE`.
