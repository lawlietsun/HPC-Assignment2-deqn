#include "Driver.h"

#include <iostream>

Driver::Driver(const InputFile* input, const std::string& pname)
    : problem_name(pname)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "+++++++++++++++++++++" << std::endl;
        std::cout << "  Running deqn v0.1  " << std::endl;
#ifdef DEBUG
        std::cout << "- input file: " << problem_name << std::endl;
#endif
    }

    dt_max = input->getDouble("dt_max",  0.2);
    dt = input->getDouble("initial_dt", 0.2);

    t_start = input->getDouble("start_time", 0.0);
    t_end = input->getDouble("end_time", 10.0);

    vis_frequency = input->getInt("vis_frequency",-1);
    summary_frequency = input->getInt("summary_frequency", 1);

    if(rank == 0) {
#ifdef DEBUG
        std::cout << "- dt_max: " << dt_max << std::endl;
        std::cout << "- initial_dt: " << dt << std::endl;
        std::cout << "- start_time: " << t_start << std::endl;
        std::cout << "- end_time: " << t_end << std::endl;
        std::cout << "- vis_frequency: " << vis_frequency << std::endl;
        std::cout << "- summary_frequency: " << summary_frequency << std::endl;
#endif
        std::cout << "+++++++++++++++++++++" << std::endl;
        std::cout << std::endl;
    }

    global_mesh = new Mesh(input);
    local_mesh = global_mesh->partition();

    diffusion = new Diffusion(input, local_mesh);
    writer = new VtkWriter(pname, local_mesh);

    /* Initial mesh dump */
    if(vis_frequency != -1)
        writer->write(0, 0.0);
}

Driver::~Driver() {
    if(!(local_mesh == global_mesh))
        delete local_mesh;
    delete global_mesh;
    delete diffusion;
    delete writer;
}

void Driver::run() {

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int step = 0;
    double t_current;

    for(t_current = t_start; t_current < t_end; t_current += dt) {
        if (rank == 0) {
            std::cout << "+ step: " << step << ", dt:   " << dt << std::endl;
        }

        diffusion->doCycle(dt);

        if(step % vis_frequency == 0 && vis_frequency != -1)
            writer->write(step, t_current);
        if(step % summary_frequency == 0 && summary_frequency != -1) {
            double temperature = local_mesh->getTotalTemperature();
            if (rank == 0) {
                std::cout << "+\tcurrent total temperature: " << temperature << std::endl;
            }
        }

        step++;
    }

    if(step % vis_frequency != 0 && vis_frequency != -1)
        writer->write(step, t_current);

    if (rank == 0) {
        std::cout << std::endl;
        std::cout << "+++++++++++++++++++++" << std::endl;
        std::cout << "   Run completete.   " << std::endl;
        std::cout << "+++++++++++++++++++++" << std::endl;
    }
}
