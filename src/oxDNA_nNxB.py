import argparse
import os
from os.path import abspath
from libs.oxDNA_to_nNxB_utils import oxDNA_to_nNxB


# Sphinx can autogenerate cli documentation if you have a function which returns your parser.
# To get it to show up in the documentation you need to add the script to oxDNA/docs/oat/cli.md
def cli_parser(prog="oxDNA_to_nNxB.py"):
    # A standard way to create and parse command line arguments.
    parser = argparse.ArgumentParser(prog = prog, description="Converts oxDNA file to nNxB format.")
    parser.add_argument('particles_per_course_bead', type=int, nargs=1, help='The number of particles that will be used per course bead in the nNxB file')
    parser.add_argument('conf', type=str, nargs=1, help='The  oxDNA configuration file that will be converted to nNxB format')
    parser.add_argument('top', type=str, nargs=1, help='oxDNA topology file')
    parser.add_argument('input', type=str, nargs=1, help='oxDNA input file')
    parser.add_argument('traj', type=str, nargs=1, help='The trajectory used to find which nuc are bonded and unbonded')
    parser.add_argument('material', type=str, nargs=1, help='Specify if DNA or RNA')
    parser.add_argument('-p', metavar='num_cpus', nargs=1, type=int, dest='parallel', help="(optional) How many cores to use")
    parser.add_argument('-o', '--output', metavar='output_file', nargs=1, help='The filename to save the mean structure to')
    parser.add_argument('-r', metavar='remainder_modifier', nargs=1, type=float, dest='remainder_modifier', help="If particles_per_course_bead * remainder_modifier <= num_particle_to_bead_remainder append remainder to previous bead")

    return parser


def main():
    parser = cli_parser(os.path.basename(__file__))
    args = parser.parse_args()
   
    # Verify that dependencies are installed and a good version
    from oxDNA_analysis_tools.config import check
    check(["python", "numpy"])
    
    particles_per_course_bead = args.particles_per_course_bead[0]
    path_to_conf = abspath(args.conf[0])
    path_to_top = abspath(args.top[0])
    path_to_input = abspath(args.input[0])
    path_to_traj = abspath(args.traj[0])
    material = abspath(args.material[0])
    
    n_cpus = args.parallel[0] if args.parallel else 1
    remainder_modifier = args.remainder_modifier[0] if args.remainder_modifier else 0.25
    system_name = args.output[0] if args.output else None
    

    oxDNA_to_nNxB(particles_per_course_bead,
                  path_to_conf,
                  path_to_top,
                  path_to_input,
                  path_to_traj,
                  material,
                  remainder_modifier,
                  n_cpus=n_cpus,
                  system_name=system_name)
    
if __name__ == "__main__":
    main()