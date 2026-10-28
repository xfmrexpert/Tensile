using System.Numerics;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using Tensile;

// Exercises the public surface through the package, and fails loudly. What it
// proves is that the package installs and runs -- both assemblies present,
// the kernels reachable through the public one -- not the numerics, which the
// unit suite owns. Each check is therefore loose but unmistakable.

int failures = 0;

void Check(bool condition, string what)
{
    Console.WriteLine($"{(condition ? "ok  " : "FAIL")}  {what}");
    if (!condition) failures++;
}

// The kernel assembly travels inside the package, and nothing public from it
// is visible: its types are all internal.
Assembly kernels = Assembly.Load("Tensile.Kernels");
Check(kernels.GetExportedTypes().Length == 0, "Tensile.Kernels is loaded and exposes no public types");

// Both assemblies carry their PDB embedded, with a SourceLink map, so a
// consumer's debugger can step into the exact source the package was built
// from. GitHub Packages hosts no symbol packages, so this is the only route.
foreach (Assembly assembly in new[] { typeof(Matrix).Assembly, kernels })
{
    Check(HasSourceLink(assembly.Location), $"{assembly.GetName().Name} has an embedded PDB with SourceLink");
}

// Real: a solve, its residual, the condition estimate, and a product.
var a = Matrix.FromRows(new[,] { { 4.0, 1.0, 0.0 }, { 1.0, 3.0, 1.0 }, { 0.0, 1.0, 2.0 } });
var b = Matrix.FromColumnMajor<double>(3, 1, [1.0, 2.0, 3.0]);
Matrix<double> x = a.Solve(b);
Matrix<double> ax = a.Multiply(x);
double residual = Math.Max(Math.Abs(ax[0, 0] - 1.0), Math.Max(Math.Abs(ax[1, 0] - 2.0), Math.Abs(ax[2, 0] - 3.0)));
Check(residual < 1e-12, $"real solve, residual {residual:E2}");

double rcond = a.FactorLu().ReciprocalCondition();
Check(rcond > 0.1 && rcond <= 1.0, $"reciprocal condition {rcond:F4}");

// The exponential of a diagonal matrix is the diagonal of exponentials.
var d = Matrix.FromRows(new[,] { { 1.0, 0.0 }, { 0.0, -2.0 } });
Matrix<double> e = d.Expm();
Check(Math.Abs(e[0, 0] - Math.E) < 1e-13 && Math.Abs(e[1, 1] - Math.Exp(-2.0)) < 1e-14, "real expm");

// A product large enough to go through the blocked GEMM and its micro-kernel.
const int N = 96;
var g = new Matrix<double>(N, N);
for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) g[i, j] = (i == j) ? 2.0 : 0.0;
Matrix<double> g2 = g.Multiply(g);
Check(g2[17, 17] == 4.0 && g2[17, 18] == 0.0, $"blocked GEMM on the {Workspace.Shared.KernelName} kernel");

// Complex: exp(i*theta*I) is e^(i theta) on the diagonal, and its action.
var z = Matrix.FromRows(new Complex[,] { { new(0, 0.5), 0.0 }, { 0.0, new(0, 0.5) } });
Matrix<Complex> ez = z.Expm();
Check(Complex.Abs(ez[0, 0] - Complex.FromPolarCoordinates(1.0, 0.5)) < 1e-14, "complex expm");

var v = Matrix.FromRows(new Complex[,] { { 1.0 }, { 0.0 } });
Matrix<Complex> ev = z.Expmv(v.ReadOnlyView);
Check(Complex.Abs(ev[0, 0] - Complex.FromPolarCoordinates(1.0, 0.5)) < 1e-14, "complex expmv");

Console.WriteLine(failures == 0 ? "package smoke test passed" : $"package smoke test FAILED ({failures})");
return failures == 0 ? 0 : 1;

static bool HasSourceLink(string path)
{
    // The SourceLink custom debug information kind, from the Portable PDB specification.
    var sourceLink = new Guid("CC110556-A091-4D38-9FEC-25AB9A351A6A");

    using var pe = new PEReader(File.OpenRead(path));
    DebugDirectoryEntry embedded = pe.ReadDebugDirectory()
        .FirstOrDefault(entry => entry.Type == DebugDirectoryEntryType.EmbeddedPortablePdb);
    if (embedded.DataSize == 0) return false;

    using MetadataReaderProvider provider = pe.ReadEmbeddedPortablePdbDebugDirectoryData(embedded);
    MetadataReader pdb = provider.GetMetadataReader();

    foreach (CustomDebugInformationHandle handle in pdb.CustomDebugInformation)
    {
        if (pdb.GetGuid(pdb.GetCustomDebugInformation(handle).Kind) == sourceLink) return true;
    }

    return false;
}
