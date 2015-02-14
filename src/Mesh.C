#include "Mesh.h"

#include <cstdlib>
#include <iostream>

#define POLY2(i, j, imin, jmin, ni) (((i) - (imin)) + ((j)-(jmin)) * (ni))

Mesh::Mesh(const InputFile* input):
input(input)
{
    allocated = false;

    NDIM = 2;

    n = new int[NDIM];
    min = new int[NDIM];
    max = new int[NDIM];
    dx = new double[NDIM];

    int nx = input->getInt("nx", 0);
    int ny = input->getInt("ny", 0);

    min_coords = new double[NDIM];
    max_coords = new double[NDIM];

    min_coords[0] = input->getDouble("xmin", 0.0);
    max_coords[0] = input->getDouble("xmax", 1.0);
    min_coords[1] = input->getDouble("ymin", 0.0);
    max_coords[1] = input->getDouble("ymax", 1.0);

    // setup first dimension.
    n[0] = nx;
    min[0] = 1;
    max[0] = nx;

    dx[0] = ((double) max_coords[0]-min_coords[0])/nx;

    // setup second dimension.
    n[1] = ny;
    min[1] = 1;
    max[1] = ny;

    dx[1] = ((double) max_coords[1]-min_coords[1])/ny;

    comm = MPI_COMM_WORLD;

    neighbours = new int[4];
    neighbours[0] = MPI_PROC_NULL;
    neighbours[1] = MPI_PROC_NULL;
    neighbours[2] = MPI_PROC_NULL;
    neighbours[3] = MPI_PROC_NULL;

    global_mesh = this;
}

Mesh::Mesh(MPI_Comm comm,
    int x_min,
    int y_min,
    int x_max,
    int y_max,
    double xmin,
    double xmax,
    double ymin,
    double ymax,
    Mesh* global_mesh) :
comm(comm),
global_mesh(global_mesh)
{
    allocated = false;

    NDIM = 2;

    n = new int[NDIM];
    min = new int[NDIM];
    max = new int[NDIM];
    dx = new double[NDIM];

    min_coords = new double[NDIM];
    max_coords = new double[NDIM];

    min_coords[0] = xmin;
    max_coords[0] = xmax;
    min_coords[1] = ymin;
    max_coords[1] = ymax;

    int nx = x_max - x_min + 1;
    int ny = y_max - y_min + 1;

    /* setup for first dimension */
    n[0] = nx;
    min[0] = x_min;
    max[0] = x_max;

    dx[0] = ((double) max_coords[0]-min_coords[0])/nx;

    /* setup second dimension */
    n[1] = ny;
    min[1] = y_min;
    max[1] = y_max;

    dx[1] = ((double) max_coords[1]-min_coords[1])/ny;

    /* setup neighbours */
    int nghbrs[4];

    MPI_Cart_shift(comm, 0, 1, &nghbrs[0], &nghbrs[1]);
    MPI_Cart_shift(comm, 1, 1, &nghbrs[2], &nghbrs[3]);

    neighbours = new int[4];

    neighbours[0] = nghbrs[3];
    neighbours[1] = nghbrs[1];
    neighbours[2] = nghbrs[2];
    neighbours[3] = nghbrs[0];
}

Mesh* Mesh::partition()
{
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs < 2) {
        allocate();
        return this;
    } else {
        // partition mesh here

        double ratio = (double) n[0] / (double) n[1];

        int blocks = nprocs;

        int px = blocks;
        int py = 1;

        int t_px, t_py = -1;

        bool split_found = false;

        for(int i = 1; i <= blocks; i++) {
            if(blocks % i == 0) {
                t_px = blocks / ( double) i;
                t_py = (double) i;

                if(t_px/t_py <= ratio) {
                    py = i;
                    px = blocks/i;
                    split_found = true;
                    break;
                }
            }
        }

        if (split_found == false || py == blocks) {
            if(ratio >= 1.0) {
                px = blocks;
                py = 1;
            } else {
                px = 1;
                py = blocks;
            }
        }

        int periods[2] = {0,0};
        int dimensions[2] = {px, py};

        MPI_Comm cart_comm;
        int cart_coords[2];

        MPI_Cart_create(
            MPI_COMM_WORLD,
            2,
            dimensions,
            periods,
            0,
            &cart_comm);

        int cart_rank;
        int rank;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        MPI_Comm_rank(cart_comm, &cart_rank);
        MPI_Cart_coords(cart_comm, cart_rank, 2, cart_coords);

        int local_nx = n[0]/px;
        int local_ny = n[1]/py;

        int xmin = min[0] + local_nx*cart_coords[0];
        int ymin = min[1] + local_ny*cart_coords[1];

        int xmax = min[0] + local_nx*(cart_coords[0]+1) - 1;
        int ymax = min[1] + local_ny*(cart_coords[1]+1) - 1;

        double xmn = min_coords[0] + dx[0]*(xmin-1);
        double xmx = min_coords[0] + dx[0]*(xmax);
        double ymn = min_coords[1] + dx[1]*(ymin-1);
        double ymx = min_coords[1] + dx[1]*(ymax);

        Mesh* mesh_partition = new Mesh(cart_comm, xmin, ymin, xmax, ymax,
            xmn, xmx, ymn, ymx, this);

        mesh_partition->allocate();
        return mesh_partition;
    }
}

void Mesh::allocate()
{
    allocated = true;

    int nx = n[0];
    int ny = n[1];

    /* Allocate arrays */
    u1 = new double[(nx+2) * (ny+2)];
    u0 = new double[(nx+2) * (ny+2)];
    
    /* Allocate and initialise coordinate arrays */
    cellx = new double[nx+2];
    celly = new double[ny+2];

    double xmin = min_coords[0];
    double ymin = min_coords[1];

    for (int i=0; i < nx+2; i++) {
        cellx[i]=xmin+dx[0]*(i-1);
    }

    for (int i = 0; i < ny+2; i++) {
        celly[i]=ymin+dx[1]*(i-1);
    }
}

double* Mesh::getU0()
{
    return u0;
}

double* Mesh::getU1()
{
    return u1;
}

double* Mesh::getDx()
{
    return dx;
}

int* Mesh::getMin()
{
    return min;
}

int* Mesh::getMax()
{
    return max;
}

int Mesh::getDim()
{
    return NDIM;
}

int* Mesh::getNx()
{
    return n;
}

MPI_Comm Mesh::getCommunicator()
{
    return comm;
}

int* Mesh::getNeighbours()
{
    return neighbours;
}

double* Mesh::getCellX()
{
    return cellx;
}

double* Mesh::getCellY()
{
    return celly;
}

Mesh* Mesh::getGlobalMesh()
{
    return global_mesh;
}

double Mesh::getTotalTemperature()
{
    if(allocated) {
        double temperature = 0.0;
        int x_min = min[0];
        int x_max = max[0];
        int y_min = min[1]; 
        int y_max = max[1]; 

        int nx = n[0]+2;

        // int my_rank;
        // int p;
        // int source;
        // int tag = 0;
        // int dest = 0;

        // MPI_Status status;

        // MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        // MPI_Comm_size(MPI_COMM_WORLD, &p);

        double start = MPI_Wtime();

        for(int k=y_min; k <= y_max; k++) {
            for(int j=x_min; j <= x_max; j++) {

                int n1 = POLY2(j,k,x_min-1,y_min-1,nx);

                temperature += u0[n1];

                // MPI_Reduce(&u0[n1], &temperature, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
                // if(my_rank == 0)
                // {
                //     temperature = u0[n1];

                //     for(source = 1; source < p; source++)
                //     {
                //         MPI_Recv(&u0[n1], 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);

                //         temperature = temperature + u0[n1];
                //     }
                // }
                // else {
                //     MPI_Send(&u0[n1], 1, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
                // }

            }
        }

        double end = MPI_Wtime();
        // printf("time = %f\n", end - start);

        return temperature;
    } else {
        return 0.0;
    }
}
