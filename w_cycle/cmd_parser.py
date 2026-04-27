
"""Command-line argument parser for CSL GMG test harness.

Arguments:

   --blockSize <int>     Size of temporary communication buffers(default: 2)
   --latestlink <str>    Directory to store log files (default: latest)
   -d, --driver <str>    Path to CSL compiler(default: cslc)
   --fabric-dims <str>   Fabric dimensions of the WSE (not always required)
   --cmaddr <str>        IP address and port of the WSE (format: <IP>:<port>)
   --channels <int>      Number of I/O channels (1 <= channels <= 16)(default: 1)
   --width-west-buf <int>  Width (columns) of buffer on the west side of the core rectangle(default: 0)
   --width-east-buf <int>  Width (columns) of buffer on the east side of the core rectangle(default: 0)
   --compile-only        Only compile the ELFs, do not run(default: False)
   --run-only            Only run with precompiled binary, do not compile(default: False)

GMG-specific arguments:
   -m <int>              Number of PE rows (core rectangle height)
   -n <int>              Number of PE columns (core rectangle width)
   -k <int>              Size of local tensor (minimum 2)
   --zDim <int>          Number of elements for y = A*x computation (domain of Laplacian)
   -l, --levels <int>        Number of multigrid levels (default: 2)
   -v, --verbose             Print detailed level information
   --tolerance <float>       Convergence tolerance (default: 1e-6)
   --pre-iter <int>          Number of pre-smoothing iterations (default: 6)
   --post-iter <int>         Number of post-smoothing iterations (default: 6)
   --bottom-iter <int>       Number of bottom solver iterations (default: 100)
   --max-ite <int>           Maximum number of GMG iterations (default: 1)
"""


import os
import argparse

def channels_type(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError("number of I/O channels must be at least 1")
    if x > 16:
        raise argparse.ArgumentTypeError("only support up to 16 I/O channels")
    return x

def print_arguments(args):
  print("=" * 60)
  print("Arguments:")
  for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
  print("=" * 60)
  
# PE rows/columns will be equal to problem size to only the single column will map on a single PE.
def parse_args():
  parser = argparse.ArgumentParser(description='Simple GMG Solver')
  # GMG related arguments
  # -s 32,32,32, this is also the problem size, nx,ny,nz == m,n,k
  parser.add_argument("-m", default=1, type=int, help="number of PE rows")
  parser.add_argument("-n", default=1, type=int, help="number of PE columns")
  parser.add_argument("-k", default=1, type=int, help="size of local tensor, no less than 2")
  parser.add_argument("--zDim", default=2, type=int, help="[0 zDim-1) is the domain of Laplacian")
  parser.add_argument("--max-ite", default=200, type=int, help="maximum number of iterations of GMG (default: 100)")
  parser.add_argument('-l', '--levels', type=int, default=2, help='Maximum number of multigrid levels (default: 2)')
  parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed level information')
  parser.add_argument('--tolerance', type=float, default=1e-4, help='Convergence tolerance (default: 1e-4)')
  parser.add_argument('--pre-iter', type=int, default=6, help='Number of pre-smoothing iterations (default: 6)')
  parser.add_argument('--post-iter', type=int, default=6, help='Number of post-smoothing iterations (default: 6)')
  parser.add_argument('--bottom-iter', type=int, default=1, help='Number of bottom solver iterations (default: 10)')

  # Other CSL related arguments
  parser.add_argument("--blockSize", default=2, type=int, help="the size of temporary buffers for communication")
  parser.add_argument("--latestlink", help="folder to contain the log files (default: latest)")
  parser.add_argument("-d", "--driver", help="The path to the CSL compiler")
  parser.add_argument("--compile-only", help="Compile only", action="store_true")
  parser.add_argument("--cmaddr", help="CM address and port, i.e. <IP>:<port>")
  parser.add_argument("--run-only", help="Run only", action="store_true")
  parser.add_argument("--fabric-dims", help="Fabric dimension, i.e. <W>,<H>")  
  parser.add_argument("--arch", help="wse2 or wse3. Default is wse2 when not supplied.")
  parser.add_argument("--width-west-buf", default=0, type=int, help="width of west buffer")
  parser.add_argument("--width-east-buf", default=0, type=int, help="width of east buffer")
  parser.add_argument("--channels", default=1, type=channels_type, help="number of I/O channels, between 1 and 16")


  args = parser.parse_args()

  logs_dir = "latest"
  if args.latestlink:
    logs_dir = args.latestlink

  dir_exist = os.path.isdir(logs_dir)
  if dir_exist:
    print(f"{logs_dir} already exists")
  else:
    print(f"create {logs_dir} to store log files")
    os.mkdir(logs_dir)

  return args, logs_dir
