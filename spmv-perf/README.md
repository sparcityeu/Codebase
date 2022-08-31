# Parallel Sparse Matrix-Vector Multiplication Performance Evaluation Tool

This performance tool is inspired by the [merge-spmv](https://github.com/dumerrill/merge-spmv/) performance evaluation tool.
The tool was modified in order to support input in form of binary-converted CSR matrices, provide stable and reproducible performance data and support different additional SpMV kernels.


##References

> Merrill, Duane, and Michael Garland. "Merge-based parallel sparse matrix-vector multiplication." In SC'16: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 678-689. IEEE, 2016.

## License

`merge-spmv` is available under the "New BSD" open-source license:

```
Copyright (c) 2020-2022, Simula Research laboratory.  All rights reserved.

Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   *  Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   *  Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   *  Neither the name of the Simule Research Laboratory and NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL SIMULA RESEARCH LABORATORY AND NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
