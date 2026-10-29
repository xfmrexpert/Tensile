using System.Numerics;

namespace Tensile;

/// <summary>
/// Element-wise arithmetic over one column at a time: the loops behind
/// addition, subtraction and scaling.
///
/// These are streamed, not packed, and memory-bound, so they live here in safe
/// code rather than behind the kernel seam -- there is nothing a pointer would
/// buy. The vector path is taken only for <see cref="double"/> and
/// <see cref="float"/>: <see cref="Vector{T}"/> arithmetic on an integer type
/// wraps where the scalar loop, compiled checked, throws, and an operation
/// whose overflow behaviour depended on the hardware would be a defect.
///
/// The vector and scalar paths round identically: each element is computed by
/// the same separate multiply and add, never a fused one, so where an element
/// falls -- in a vector or in the tail -- does not change its value.
/// </summary>
internal static class Elementwise
{
    private static bool Vectorized<T>() =>
        Vector.IsHardwareAccelerated && (typeof(T) == typeof(double) || typeof(T) == typeof(float));

    /// <summary>y := y + x.</summary>
    public static void Add<T>(ReadOnlySpan<T> x, Span<T> y) where T : unmanaged, INumberBase<T>
    {
        int i = 0;

        if (Vectorized<T>())
        {
            for (int width = Vector<T>.Count; i <= y.Length - width; i += width)
                (new Vector<T>(y.Slice(i)) + new Vector<T>(x.Slice(i))).CopyTo(y.Slice(i));
        }

        for (; i < y.Length; i++) y[i] += x[i];
    }

    /// <summary>y := y - x.</summary>
    public static void Subtract<T>(ReadOnlySpan<T> x, Span<T> y) where T : unmanaged, INumberBase<T>
    {
        int i = 0;

        if (Vectorized<T>())
        {
            for (int width = Vector<T>.Count; i <= y.Length - width; i += width)
                (new Vector<T>(y.Slice(i)) - new Vector<T>(x.Slice(i))).CopyTo(y.Slice(i));
        }

        for (; i < y.Length; i++) y[i] -= x[i];
    }

    /// <summary>y := y + alpha*x.</summary>
    public static void AddScaled<T>(T alpha, ReadOnlySpan<T> x, Span<T> y) where T : unmanaged, INumberBase<T>
    {
        int i = 0;

        if (Vectorized<T>())
        {
            for (int width = Vector<T>.Count; i <= y.Length - width; i += width)
                (new Vector<T>(y.Slice(i)) + (new Vector<T>(x.Slice(i)) * alpha)).CopyTo(y.Slice(i));
        }

        for (; i < y.Length; i++) y[i] += alpha * x[i];
    }

    /// <summary>y := alpha*y.</summary>
    public static void Scale<T>(T alpha, Span<T> y) where T : unmanaged, INumberBase<T>
    {
        int i = 0;

        if (Vectorized<T>())
        {
            for (int width = Vector<T>.Count; i <= y.Length - width; i += width)
                (new Vector<T>(y.Slice(i)) * alpha).CopyTo(y.Slice(i));
        }

        for (; i < y.Length; i++) y[i] = alpha * y[i];
    }

    /// <summary>
    /// y := -y. Separate from scaling by minus one, which an unsigned element
    /// type cannot form, though it can negate a zero.
    /// </summary>
    public static void Negate<T>(Span<T> y) where T : unmanaged, INumberBase<T>
    {
        int i = 0;

        if (Vectorized<T>())
        {
            for (int width = Vector<T>.Count; i <= y.Length - width; i += width)
                (-new Vector<T>(y.Slice(i))).CopyTo(y.Slice(i));
        }

        for (; i < y.Length; i++) y[i] = -y[i];
    }
}
