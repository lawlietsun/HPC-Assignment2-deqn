#ifndef DIFFUSION_MESH_H_
#define DIFFUSION_MESH_H_

#include "InputFile.h"

#include "mpi.h"

class Mesh {
    private:
        const InputFile* input;

        double* u1;
        double* u0;
        double* cellx;
        double* celly;

        double* min_coords;
        double* max_coords;

        int NDIM;

        int* n; 
        int* min;
        int* max;

        double* dx;

        /*
         * A mesh has four neighbours, and they are 
         * accessed in the following order:
         * - top
         * - right
         * - bottom
         * - left
         */
        int* neighbours;

        // mpi communicator
        MPI_Comm comm;

        Mesh* global_mesh;

        void allocate();
        bool allocated;
    public:
        Mesh(const InputFile* input);

        Mesh(MPI_Comm cart_comm,
                int x_min,
                int y_min,
                int x_max,
                int y_max,
                double xmin,
                double xmax,
                double ymin,
                double ymax,
                Mesh* global);

        Mesh* partition();

        Mesh* getGlobalMesh();
                
        double* getU0();
        double* getU1();

        double* getDx();
        int* getNx();
        int* getMin();
        int* getMax();
        int getDim();

        double* getCellX();
        double* getCellY();

        int* getNeighbours();

        double getTotalTemperature();

        MPI_Comm getCommunicator();
};
#endif
