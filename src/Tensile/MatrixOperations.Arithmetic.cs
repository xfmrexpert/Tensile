using System.Numerics;

namespace Tensile;

// Element-wise arithmetic, transposes and the product operator.
//
// Unlike the rest of this class, the element-wise operations are generic: they
// need nothing from an element type but its own +, - and *, which
// INumberBase<T> supplies, so they hold for every Matrix<T> there is -- float,
// Half, an integer type -- and not only for the two types with a GEMM. The
// closed-type rule exists so that a type with no arithmetic fails to compile;
// here every admissible type has the arithmetic.
//
// Each operation that writes into storage the caller owns comes in two
// receivers, a view and a matrix: an extension method's receiver does not take
// the implicit conversion from Matrix<T> to MatrixView<T>, so without the
// second one `m.AddInPlace(x)` would not compile.
//
// The allocating forms are the operators on Matrix<T> (+, -, unary -, scalar
// *), and a matrix product * here for double and Complex.
//
// The analyzers do not yet understand C# 14 extension blocks: they see each
// block as a nested public type (CA1034), and two blocks as two such types
// whose generated names differ only by case (CA1708). Neither is a type in
// the language. Both are silenced at the one place each fires, not globally.
#pragma warning disable CA1708
public static partial class MatrixOperations
#pragma warning restore CA1708
{
    /// <summary>target := target + x, element by element.</summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="target">The window to add into.</param>
    /// <param name="x">The window to add; the same shape as <paramref name="target"/>. May overlap it.</param>
    /// <exception cref="ArgumentException">The shapes differ.</exception>
    /// <exception cref="AllocationLimitException"><paramref name="x"/> overlaps the target in a different position and staging it would exceed <see cref="TensileLimits.MaxElements"/>.</exception>
    public static void AddInPlace<T>(this MatrixView<T> target, ReadOnlyMatrixView<T> x)
        where T : unmanaged, INumberBase<T>
    {
        RequireSameShape(target.Shape, x.Shape, nameof(x));
        if (target.IsEmpty) return;

        x = Unaliased(target, x);
        for (int j = 0; j < target.Columns; j++) Elementwise.Add(x.Column(j), target.Column(j));
    }

    /// <inheritdoc cref="AddInPlace{T}(MatrixView{T}, ReadOnlyMatrixView{T})"/>
    public static void AddInPlace<T>(this Matrix<T> target, ReadOnlyMatrixView<T> x)
        where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(target);
        target.View.AddInPlace(x);
    }

    /// <summary>target := target - x, element by element.</summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="target">The window to subtract from.</param>
    /// <param name="x">The window to subtract; the same shape as <paramref name="target"/>. May overlap it.</param>
    /// <exception cref="ArgumentException">The shapes differ.</exception>
    /// <exception cref="AllocationLimitException"><paramref name="x"/> overlaps the target in a different position and staging it would exceed <see cref="TensileLimits.MaxElements"/>.</exception>
    public static void SubtractInPlace<T>(this MatrixView<T> target, ReadOnlyMatrixView<T> x)
        where T : unmanaged, INumberBase<T>
    {
        RequireSameShape(target.Shape, x.Shape, nameof(x));
        if (target.IsEmpty) return;

        x = Unaliased(target, x);
        for (int j = 0; j < target.Columns; j++) Elementwise.Subtract(x.Column(j), target.Column(j));
    }

    /// <inheritdoc cref="SubtractInPlace{T}(MatrixView{T}, ReadOnlyMatrixView{T})"/>
    public static void SubtractInPlace<T>(this Matrix<T> target, ReadOnlyMatrixView<T> x)
        where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(target);
        target.View.SubtractInPlace(x);
    }

    /// <summary>
    /// target := target + alpha*x, element by element: BLAS's axpy, over a
    /// window. Each element is a separate multiply and add, not a fused one.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="target">The window to accumulate into.</param>
    /// <param name="alpha">The scalar on <paramref name="x"/>.</param>
    /// <param name="x">The window to scale and add; the same shape as <paramref name="target"/>. May overlap it.</param>
    /// <exception cref="ArgumentException">The shapes differ.</exception>
    /// <exception cref="AllocationLimitException"><paramref name="x"/> overlaps the target in a different position and staging it would exceed <see cref="TensileLimits.MaxElements"/>.</exception>
    public static void AddScaledInPlace<T>(this MatrixView<T> target, T alpha, ReadOnlyMatrixView<T> x)
        where T : unmanaged, INumberBase<T>
    {
        RequireSameShape(target.Shape, x.Shape, nameof(x));
        if (target.IsEmpty) return;

        x = Unaliased(target, x);
        for (int j = 0; j < target.Columns; j++) Elementwise.AddScaled(alpha, x.Column(j), target.Column(j));
    }

    /// <inheritdoc cref="AddScaledInPlace{T}(MatrixView{T}, T, ReadOnlyMatrixView{T})"/>
    public static void AddScaledInPlace<T>(this Matrix<T> target, T alpha, ReadOnlyMatrixView<T> x)
        where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(target);
        target.View.AddScaledInPlace(alpha, x);
    }

    /// <summary>target := alpha*target, element by element.</summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="target">The window to scale.</param>
    /// <param name="alpha">The scalar.</param>
    public static void ScaleInPlace<T>(this MatrixView<T> target, T alpha)
        where T : unmanaged, INumberBase<T>
    {
        if (target.IsEmpty) return;

        for (int j = 0; j < target.Columns; j++) Elementwise.Scale(alpha, target.Column(j));
    }

    /// <inheritdoc cref="ScaleInPlace{T}(MatrixView{T}, T)"/>
    public static void ScaleInPlace<T>(this Matrix<T> target, T alpha)
        where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(target);
        target.View.ScaleInPlace(alpha);
    }

    /// <summary>
    /// The adjoint (conjugate transpose) A^H, as a new matrix. For a real
    /// matrix that is the transpose; this overload exists so code written
    /// against A^H reads the same for both element types.
    /// </summary>
    /// <param name="a">The matrix.</param>
    public static Matrix<double> Adjoint(this Matrix<double> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        return a.Transpose();
    }

    /// <summary>The adjoint (conjugate transpose) A^H, as a new matrix.</summary>
    /// <param name="a">The matrix.</param>
    public static Matrix<Complex> Adjoint(this Matrix<Complex> a)
    {
        ArgumentNullException.ThrowIfNull(a);

        var result = a.Transpose();
        for (int j = 0; j < result.Columns; j++)
        {
            Span<Complex> column = result.Column(j);
            for (int i = 0; i < column.Length; i++) column[i] = Complex.Conjugate(column[i]);
        }

        return result;
    }

    // C# 14 extension operators: the product exists only for the element types
    // that have a GEMM, so it cannot be declared on Matrix<T> itself.
#pragma warning disable CA1034
    extension(Matrix<double>)
#pragma warning restore CA1034
    {
        /// <summary>
        /// A*B, as a new matrix, on <see cref="Workspace.Shared"/>: the same as
        /// <see cref="Multiply(Matrix{double}, ReadOnlyMatrixView{double}, Workspace)"/>
        /// with no workspace.
        /// </summary>
        /// <param name="a">Left operand, m x k.</param>
        /// <param name="b">Right operand, k x n.</param>
        /// <exception cref="ArgumentException">The inner dimensions disagree.</exception>
        public static Matrix<double> operator *(Matrix<double> a, Matrix<double> b)
        {
            ArgumentNullException.ThrowIfNull(b);
            return a.Multiply(b.ReadOnlyView);
        }
    }

#pragma warning disable CA1034
    extension(Matrix<Complex>)
#pragma warning restore CA1034
    {
        /// <summary>
        /// A*B for complex operands, as a new matrix, on
        /// <see cref="Workspace.Shared"/>: the same as
        /// <see cref="Multiply(Matrix{Complex}, ReadOnlyMatrixView{Complex}, Workspace)"/>
        /// with no workspace.
        /// </summary>
        /// <param name="a">Left operand, m x k.</param>
        /// <param name="b">Right operand, k x n.</param>
        /// <exception cref="ArgumentException">The inner dimensions disagree.</exception>
        public static Matrix<Complex> operator *(Matrix<Complex> a, Matrix<Complex> b)
        {
            ArgumentNullException.ThrowIfNull(b);
            return a.Multiply(b.ReadOnlyView);
        }
    }

    internal static void RequireSameShape(MatrixShape target, MatrixShape operand, string parameter)
    {
        if (target.Rows != operand.Rows || target.Columns != operand.Columns)
        {
            throw new ArgumentException(
                $"Shape mismatch: target is {target.Rows}x{target.Columns}, operand is {operand.Rows}x{operand.Columns}.",
                parameter);
        }
    }

    /// <summary>
    /// <paramref name="x"/>, or a packed copy of it if writing the target
    /// column by column could change an element of x before it is read.
    ///
    /// The one overlap that is safe without a copy is the exact one -- the
    /// same window, as in <c>m.AddInPlace(m)</c> -- where every element is
    /// read before it is written. Anything else, a slice shifted by a row or
    /// a column, is staged, as <see cref="MatrixView{T}.CopyTo"/> does.
    /// </summary>
    private static ReadOnlyMatrixView<T> Unaliased<T>(MatrixView<T> target, ReadOnlyMatrixView<T> x)
        where T : unmanaged
    {
        if (!target.Buffer.Overlaps(x.Buffer, out int offset)) return x;
        if (offset == 0 && target.Stride == x.Stride) return x;

        return ReadOnlyMatrixView<T>.Bind(x.ToArray(), new MatrixShape(x.Rows, x.Columns));
    }
}
