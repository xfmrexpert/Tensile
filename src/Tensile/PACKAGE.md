# Tensile

Dense linear algebra for .NET, real and complex, in managed code. No native
dependencies.

```csharp
using System.Numerics;
using Tensile;

var a = Matrix.FromRows(new[,] { { 4.0, 1.0 }, { 1.0, 3.0 } });
var b = Matrix.FromColumnMajor<double>(2, 1, [1.0, 2.0]);

Matrix<double> x = a.Solve(b);
double rcond = a.FactorLu().ReciprocalCondition();
Matrix<double> e = a.Expm();                         // the matrix exponential

var z = Matrix.FromRows(new Complex[,] { { new(0, 1), 2.0 }, { -2.0, new(0, 1) } });
Matrix<Complex> y = z.Expmv(z.ReadOnlyView, t: 0.5); // exp(0.5 Z) Z, without forming exp
```

- **Arithmetic**: `a + b`, `a - b`, `2.5 * a`, the matrix product `a * b`,
  in-place forms over matrices and slices, transpose and adjoint.
- **GEMM** with BLIS-style blocking and AVX-512, AVX2/FMA and scalar
  micro-kernels chosen per host; single- and multi-threaded. At parity with
  BLIS on one core from n = 256 up.
- **Complex products** by the 4M method over the real GEMM.
- **LU with partial pivoting**, real and complex, with solves against A and
  its adjoint.
- **Condition estimation**: Higham–Tisseur block 1-norm estimation and the
  `dgecon`/`zgecon` equivalent.
- **The matrix exponential** (Al-Mohy & Higham 2009) and **its action**
  exp(tA)B (Al-Mohy & Higham 2011), real and complex, dense and matrix-free.
- **Secure by design**: the public assembly compiles with no unsafe code and
  checked arithmetic, and every request-sized allocation passes a
  configurable ceiling.

Requires .NET 10. Pre-1.0: the API may still change between minor versions.

[API guide](https://github.com/xfmrexpert/Tensile/blob/main/docs/api.md) ·
[Source and measurements](https://github.com/xfmrexpert/Tensile) ·
BSD-3-Clause
