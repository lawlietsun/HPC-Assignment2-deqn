# deqn - a simple 2D heat diffusion simulation

`deqn` is a program that solves the 2D diffusion equation on a structured mesh.

## Building

- Just type `make`!

## Running

Using your favourite MPI flavour:

    mpirun -n 4 ./deqn <input-file>

## Input Files

The file `test/square.in` demonstrates the supported input parameters, most importantly:

- `vis_frequency <n>` controls how often visualisation files are written out.
- `subregion <xmin> <ymin> <xmax> <ymax>` specifies the region of the problem domain that will be initially heated.
