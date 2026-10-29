using System.Numerics;

namespace Tensile.Tests;

/// <summary>
/// Element-wise arithmetic, transposes and the product operator.
///
/// Element-wise results are asserted exactly, not by tolerance: each element is
/// one IEEE operation, the same one whether it lands in a vector or in the
/// scalar tail, so anything but bit-for-bit agreement with the scalar formula
/// is a defect. The sizes are chosen to straddle every vector width (2, 4 and
/// 8 doubles; 4, 8 and 16 floats), so both paths and their seam are reached.
/// </summary>
public class ArithmeticTests
{
    public static TheoryData<int, int, int> Shapes => new()
    {
        { 1, 1, 0 },
        { 3, 2, 0 },
        { 7, 3, 9 },
        { 8, 4, 0 },
        { 17, 5, 20 },
        { 33, 2, 0 },
    };

    // ---- operators ---------------------------------------------------------

    [Theory]
    [MemberData(nameof(Shapes))]
    public void OperatorsMatchTheScalarFormulaExactly(int rows, int columns, int stride)
    {
        var a = ComplexLuTests.RandomReal(rows, columns, seed: rows, stride);
        var b = ComplexLuTests.RandomReal(rows, columns, seed: rows + 1, stride);
        const double Alpha = 0.3;

        var sum = a + b;
        var difference = a - b;
        var negated = -a;
        var left = Alpha * a;
        var right = a * Alpha;

        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                Assert.Equal(a[i, j] + b[i, j], sum[i, j]);
                Assert.Equal(a[i, j] - b[i, j], difference[i, j]);
                Assert.Equal(-a[i, j], negated[i, j]);
                Assert.Equal(Alpha * a[i, j], left[i, j]);
                Assert.Equal(Alpha * a[i, j], right[i, j]);
            }
        }
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void ComplexOperatorsMatchTheScalarFormulaExactly(int rows, int columns, int stride)
    {
        var a = ComplexLuTests.Random(rows, columns, seed: rows, stride);
        var b = ComplexLuTests.Random(rows, columns, seed: rows + 1, stride);
        var alpha = new Complex(0.3, -1.7);

        var sum = a + b;
        var difference = a - b;
        var negated = -a;
        var scaled = alpha * a;

        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                Assert.Equal(a[i, j] + b[i, j], sum[i, j]);
                Assert.Equal(a[i, j] - b[i, j], difference[i, j]);
                Assert.Equal(-a[i, j], negated[i, j]);
                Assert.Equal(alpha * a[i, j], scaled[i, j]);
            }
        }
    }

    /// <summary>
    /// Element-wise arithmetic needs nothing but INumberBase, so it works for
    /// element types that have no GEMM. float takes the vector path too.
    /// </summary>
    [Fact]
    public void FloatAndHalfMatricesAddAndScale()
    {
        var f = new Matrix<float>(19, 3);
        for (int j = 0; j < 3; j++) for (int i = 0; i < 19; i++) f[i, j] = (i * 0.37f) - j;

        var doubled = f + f;
        var scaled = 0.5f * f;

        for (int j = 0; j < 3; j++)
        {
            for (int i = 0; i < 19; i++)
            {
                Assert.Equal(f[i, j] + f[i, j], doubled[i, j]);
                Assert.Equal(0.5f * f[i, j], scaled[i, j]);
            }
        }

        var h = Matrix.FromRows(new Half[,] { { (Half)1, (Half)2 }, { (Half)3, (Half)4 } });
        Assert.Equal((Half)6, (h + h)[1, 0]);
    }

    /// <summary>
    /// An integer matrix takes the scalar path, which this assembly compiles
    /// checked, so overflow throws. A vector path would wrap silently, which
    /// is why integers do not get one. 64 rows is a whole number of vectors
    /// at every width, so no element would fall to a scalar tail that throws
    /// anyway.
    /// </summary>
    [Fact]
    public void IntegerOverflowThrowsRatherThanWraps()
    {
        var a = new Matrix<int>(64, 1);
        a.Fill(int.MaxValue);
        var one = new Matrix<int>(64, 1);
        one.Fill(1);

        Assert.Throws<OverflowException>(() => a + one);
        Assert.Throws<OverflowException>(() => 2 * a);
    }

    /// <summary>Negation is not scaling by minus one, which an unsigned type cannot form.</summary>
    [Fact]
    public void AnUnsignedZeroMatrixNegates()
    {
        var zero = new Matrix<uint>(3, 3);

        Assert.All((-zero).ToArray(), value => Assert.Equal(0u, value));
    }

    [Fact]
    public void ScalarsConvertToTheElementType()
    {
        var a = Matrix.FromRows(new[,] { { 1.0, 2.0 } });
        var z = Matrix.FromRows(new Complex[,] { { new(1, 1) } });

        Assert.Equal(4.0, (2 * a)[0, 1]);
        Assert.Equal(new Complex(2, 2), (z * 2.0)[0, 0]);
    }

    [Fact]
    public void MismatchedShapesAreAnArgumentError()
    {
        var a = new Matrix<double>(3, 2);
        var b = new Matrix<double>(2, 3);

        Assert.Throws<ArgumentException>(() => a + b);
        Assert.Throws<ArgumentException>(() => a - b);
    }

    [Fact]
    public void NullOperandsAreAnArgumentError()
    {
        var a = new Matrix<double>(2, 2);
        Matrix<double> none = null!;

        Assert.Throws<ArgumentNullException>(() => a + none);
        Assert.Throws<ArgumentNullException>(() => none - a);
        Assert.Throws<ArgumentNullException>(() => -none);
        Assert.Throws<ArgumentNullException>(() => 2.0 * none);
        Assert.Throws<ArgumentNullException>(() => a * none);
        Assert.Throws<ArgumentNullException>(() => none.AddInPlace(a));
    }

    [Fact]
    public void OperatorResultsAreNewStorage()
    {
        var a = Matrix.Identity<double>(3);
        var sum = a + a;

        sum[0, 0] = 99.0;

        Assert.Equal(1.0, a[0, 0]);
        Assert.False(sum.View.Overlaps(a.View));
    }

    // ---- in place ----------------------------------------------------------

    [Theory]
    [MemberData(nameof(Shapes))]
    public void InPlaceFormsMatchTheScalarFormulaExactly(int rows, int columns, int stride)
    {
        var a = ComplexLuTests.RandomReal(rows, columns, seed: rows + 2, stride);
        var x = ComplexLuTests.RandomReal(rows, columns, seed: rows + 3);
        const double Alpha = -2.5;

        var added = a.Clone();
        added.AddInPlace(x);
        var subtracted = a.Clone();
        subtracted.SubtractInPlace(x);
        var accumulated = a.Clone();
        accumulated.AddScaledInPlace(Alpha, x);
        var scaled = a.Clone();
        scaled.ScaleInPlace(Alpha);

        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                Assert.Equal(a[i, j] + x[i, j], added[i, j]);
                Assert.Equal(a[i, j] - x[i, j], subtracted[i, j]);
                Assert.Equal(a[i, j] + (Alpha * x[i, j]), accumulated[i, j]);
                Assert.Equal(Alpha * a[i, j], scaled[i, j]);
            }
        }
    }

    [Fact]
    public void ComplexAccumulationMatchesTheScalarFormulaExactly()
    {
        var a = ComplexLuTests.Random(6, 4, seed: 41);
        var x = ComplexLuTests.Random(6, 4, seed: 42);
        var alpha = new Complex(-0.25, 3.0);

        var accumulated = a.Clone();
        accumulated.AddScaledInPlace(alpha, x);

        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 6; i++)
                Assert.Equal(a[i, j] + (alpha * x[i, j]), accumulated[i, j]);
    }

    /// <summary>A slice is a view, and the in-place forms write through it and nowhere else.</summary>
    [Fact]
    public void InPlaceOnASliceTouchesOnlyTheSlice()
    {
        var a = new Matrix<double>(6, 5);
        a.Fill(1.0);
        var x = new Matrix<double>(2, 3);
        x.Fill(4.0);

        MatrixView<double> block = a.Slice(1, 1, 2, 3);
        block.AddInPlace(x);
        block.ScaleInPlace(2.0);

        for (int j = 0; j < 5; j++)
            for (int i = 0; i < 6; i++)
                Assert.Equal(i is >= 1 and <= 2 && j is >= 1 and <= 3 ? 10.0 : 1.0, a[i, j]);
    }

    /// <summary>Stride padding is storage the view does not own, and must never be written.</summary>
    [Fact]
    public void PaddingIsNeverWritten()
    {
        var shape = new MatrixShape(5, 3, stride: 8);
        double[] storage = new double[shape.RequiredExtent];
        Array.Fill(storage, double.NaN);

        MatrixView<double> target = MatrixView<double>.Bind(storage, shape);
        target.Fill(1.0);

        var x = new Matrix<double>(5, 3);
        x.Fill(2.0);

        target.AddInPlace(x);
        target.SubtractInPlace(x);
        target.AddScaledInPlace(3.0, x);
        target.ScaleInPlace(0.5);
        x.TransposeInto(new Matrix<double>(3, 5).View);

        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < shape.Stride && (j * shape.Stride) + k < storage.Length; k++)
            {
                double value = storage[(j * shape.Stride) + k];
                if (k < 5) Assert.Equal(3.5, value);
                else Assert.True(double.IsNaN(value), $"padding at column {j}, offset {k} was written");
            }
        }
    }

    [Fact]
    public void InPlaceShapeMismatchLeavesTheTargetUntouched()
    {
        var a = Matrix.Identity<double>(3);
        var x = new Matrix<double>(3, 2);

        Assert.Throws<ArgumentException>(() => a.AddInPlace(x));
        Assert.Throws<ArgumentException>(() => a.SubtractInPlace(x));
        Assert.Throws<ArgumentException>(() => a.AddScaledInPlace(2.0, x));

        Assert.Equal(Matrix.Identity<double>(3).ToArray(), a.ToArray());
    }

    /// <summary>The exact alias is safe without a copy: every element is read before it is written.</summary>
    [Fact]
    public void AMatrixCombinedWithItself()
    {
        var a = ComplexLuTests.RandomReal(9, 4, seed: 43);
        var original = a.Clone();

        a.AddInPlace(a);
        for (int j = 0; j < 4; j++) for (int i = 0; i < 9; i++) Assert.Equal(original[i, j] + original[i, j], a[i, j]);

        a.SubtractInPlace(a);
        Assert.All(a.ToArray(), value => Assert.Equal(0.0, value));
    }

    /// <summary>
    /// Two windows of one matrix, shifted against each other. With the operand
    /// ahead of the target, a forward walk happens to read every operand
    /// element before writing over it; with the operand behind, writing the
    /// target's first column (or row) overwrites the operand's second before it
    /// is read. Both directions must give what a separate copy of the operand
    /// would.
    /// </summary>
    [Theory]
    [InlineData(0, 1, false)]   // a column apart
    [InlineData(1, 0, false)]   // a row apart
    [InlineData(1, 1, false)]
    [InlineData(0, 1, true)]
    [InlineData(1, 0, true)]
    [InlineData(1, 1, true)]
    public void OverlappingWindowsAreStaged(int rowShift, int columnShift, bool operandBehind)
    {
        var a = ComplexLuTests.RandomReal(12, 8, seed: 44);
        var expected = a.Clone();

        (int targetRow, int targetColumn, int operandRow, int operandColumn) =
            operandBehind ? (rowShift, columnShift, 0, 0) : (0, 0, rowShift, columnShift);

        var operand = Matrix.From(a.ReadOnlyView.Slice(operandRow, operandColumn, 10, 6));
        expected.Slice(targetRow, targetColumn, 10, 6).AddScaledInPlace(0.5, operand);

        a.Slice(targetRow, targetColumn, 10, 6).AddScaledInPlace(0.5, a.ReadOnlyView.Slice(operandRow, operandColumn, 10, 6));

        Assert.Equal(expected.ToArray(), a.ToArray());
    }

    // ---- transposes --------------------------------------------------------

    [Theory]
    [MemberData(nameof(Shapes))]
    public void TransposeMovesEveryElement(int rows, int columns, int stride)
    {
        var a = ComplexLuTests.RandomReal(rows, columns, seed: rows + 5, stride);

        var t = a.Transpose();

        Assert.Equal(columns, t.Rows);
        Assert.Equal(rows, t.Columns);
        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                Assert.Equal(a[i, j], t[j, i]);

        Assert.Equal(Matrix.From(a.ReadOnlyView).ToArray(), t.Transpose().ToArray());
    }

    /// <summary>
    /// Transposing a square window onto itself is the overlap that even exact
    /// alignment does not make safe, so it is staged too.
    /// </summary>
    [Fact]
    public void TransposeInPlaceOfASquareMatrix()
    {
        var a = ComplexLuTests.RandomReal(7, 7, seed: 45);
        var expected = a.Transpose();

        a.TransposeInto(a.View);

        Assert.Equal(expected.ToArray(), a.ToArray());
    }

    [Fact]
    public void TransposeIntoAWindowOfTheWrongShapeIsAnArgumentError()
    {
        var a = new Matrix<double>(3, 2);

        Assert.Throws<ArgumentException>(() => a.TransposeInto(new Matrix<double>(3, 2).View));
    }

    [Fact]
    public void TransposeIntoFromEitherViewKind()
    {
        var a = ComplexLuTests.RandomReal(4, 3, seed: 46);
        var fromWritable = new Matrix<double>(3, 4);
        var fromReadOnly = new Matrix<double>(3, 4);

        a.View.TransposeInto(fromWritable.View);
        a.ReadOnlyView.TransposeInto(fromReadOnly.View);

        Assert.Equal(a.Transpose().ToArray(), fromWritable.ToArray());
        Assert.Equal(a.Transpose().ToArray(), fromReadOnly.ToArray());
    }

    [Fact]
    public void ComplexAdjointConjugatesAndTransposes()
    {
        var a = ComplexLuTests.Random(5, 3, seed: 47);

        var h = a.Adjoint();

        Assert.Equal(ComplexLuTests.Adjoint(a).ToArray(), h.ToArray());
    }

    [Fact]
    public void RealAdjointIsTheTranspose()
    {
        var a = ComplexLuTests.RandomReal(5, 3, seed: 48);

        Assert.Equal(a.Transpose().ToArray(), a.Adjoint().ToArray());
    }

    // ---- the product operator ---------------------------------------------

    [Fact]
    public void TheProductOperatorIsMultiply()
    {
        var a = ComplexLuTests.RandomReal(9, 5, seed: 49);
        var b = ComplexLuTests.RandomReal(5, 4, seed: 50);
        var za = ComplexLuTests.Random(9, 5, seed: 51);
        var zb = ComplexLuTests.Random(5, 4, seed: 52);

        Assert.Equal(a.Multiply(b).ToArray(), (a * b).ToArray());
        Assert.Equal(za.Multiply(zb).ToArray(), (za * zb).ToArray());
    }

    [Fact]
    public void AProductOfNonconformableMatricesIsAnArgumentError()
    {
        var a = new Matrix<double>(3, 2);

        Assert.Throws<ArgumentException>(() => a * a);
    }

    /// <summary>Operators compose: a residual reads like the mathematics.</summary>
    [Fact]
    public void OperatorsCompose()
    {
        var a = Matrix.FromRows(new[,] { { 4.0, 1.0 }, { 1.0, 3.0 } });
        var b = Matrix.FromColumnMajor<double>(2, 1, [1.0, 2.0]);

        var residual = (a * a.Solve(b)) - b;

        Assert.True(residual.FrobeniusNorm() < 1e-15);
    }
}

/// <summary>
/// Staging under the allocation limit: an overlapping operand that would need a
/// copy larger than the limit is refused before the target is touched.
/// </summary>
[Collection(nameof(Invariants.TensileLimitsCollection))]
public sealed class ArithmeticLimitTests : IDisposable
{
    public void Dispose() => TensileLimits.Reset();

    [Fact]
    public void ARefusedStagingCopyLeavesTheTargetUntouched()
    {
        var a = ComplexLuTests.RandomReal(40, 40, seed: 53);
        var before = a.Clone();

        TensileLimits.MaxElements = 100;

        Assert.Throws<AllocationLimitException>(
            () => a.Slice(0, 0, 39, 39).AddInPlace(a.ReadOnlyView.Slice(1, 1, 39, 39)));
        Assert.Throws<AllocationLimitException>(() => a.TransposeInto(a.View));

        TensileLimits.Reset();
        Assert.Equal(before.ToArray(), a.ToArray());
    }

    /// <summary>The exact alias needs no copy, so the limit does not apply to it.</summary>
    [Fact]
    public void TheExactAliasNeedsNoAllocation()
    {
        var a = ComplexLuTests.RandomReal(40, 40, seed: 54);

        TensileLimits.MaxElements = 100;

        a.AddInPlace(a);
    }
}
