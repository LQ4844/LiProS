# Third-party notices

## Multistream fusion

The multistream fusion code in `layers/GNN.py` includes MIT-licensed material covered by the following copyright and permission notices.

```text
MIT License

Copyright (c) 2025 filipovicSamuel

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

## Multi-stage training and graph components

`SARGCN`, `RGCN`, and `HGT` in `layers/GNN.py`, the classifier in `layers/Classifier.py`, and parts of the multi-stage training and metric code in `models/lipros.py` and `src/utils.py` retain code from the M3S PyTorch implementation. `ESAConsEncoder` also uses `SARGCN`. The public upstream repository lists no project license. Confirm permission with the rights holders before redistributing the retained code. [Upstream repository](https://github.com/Junseok0207/M3S_Pytorch) · [Paper](https://arxiv.org/abs/1902.11038)
