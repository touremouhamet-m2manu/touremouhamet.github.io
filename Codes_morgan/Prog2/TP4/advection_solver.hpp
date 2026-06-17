// advection_solver.hpp (for Exercice 5,6,7)
// New file, assuming separate header

#include "class_mesh_up.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <iostream>

class advection_solver {
protected:
    std::vector<double> u_;
    triangulation& mesh_;

public:
    advection_solver(triangulation& mesh) : mesh_(mesh), u_(mesh.n_elements(), 0.0) {}

    void set_initial_data(point2d Xc, double amp, double sig) {
        for (size_t i = 0; i < mesh_.n_elements(); ++i) {
            point2d cent = mesh_.triangles()[i].centroid();
            double dist_sq = (cent[0] - Xc[0]) * (cent[0] - Xc[0]) + (cent[1] - Xc[1]) * (cent[1] - Xc[1]);
            u_[i] = amp * std::exp(-dist_sq / (2.0 * sig * sig));
        }
    }

    void advance(double T_final, point2d velocity) {
        double vel_norm = std::sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1]);
        double dt = (vel_norm > 0) ? 0.5 * mesh_.mesh_size() / vel_norm : 1.0; // Avoid div by 0
        double t = 0.0;
        int step = 0;

        while (t < T_final) {
            std::vector<double> next_u = u_;
            // Exercice 7: edge-loop implementation
            for (const auto& e : mesh_.edges()) {
                int left = e.neighbor(0);
                int right = e.neighbor(1);
                double vn = velocity[0] * e.normal()[0] + velocity[1] * e.normal()[1];
                double gtf;
                double flux_val;
                if (right >= 0) {
                    // internal
                    gtf = std::max(vn, 0.0) * u_[left] + std::min(vn, 0.0) * u_[right];
                    flux_val = e.length() * gtf;
                    next_u[left] -= dt / mesh_.triangles()[left].area() * flux_val;
                    next_u[right] += dt / mesh_.triangles()[right].area() * flux_val;
                } else {
                    // boundary
                    gtf = std::max(vn, 0.0) * u_[left] + std::min(vn, 0.0) * 0.0;
                    flux_val = e.length() * gtf;
                    next_u[left] -= dt / mesh_.triangles()[left].area() * flux_val;
                }
            }
            u_ = next_u;
            t += dt;
            if (++step % 10 == 0) {
                std::cout << "Step " << step << ", t = " << t << std::endl;
            }
        }
    }

    void export_to_VTK(std::string const& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return;

        file << "# vtk DataFile Version 3.0" << std::endl;
        file << "Advection Initial Condition" << std::endl;
        file << "ASCII" << std::endl;
        file << "DATASET UNSTRUCTURED_GRID" << std::endl;

        // 1. write the vertices
        file << "POINTS " << mesh_.n_vertices() << " double" << std::endl;
        for (const auto& v : mesh_.vertices()) {
            file << v->location()[0] << " " << v->location()[1] << " 0.0" << std::endl;
        }

        // 2. write the cells - format: CELLS [nb_cellules] [nb_total_entiers]
        // for each triangle: 3 + index1 + index2 + index3 (4 ints)
        file << "CELLS " << mesh_.n_elements() << " " << mesh_.n_elements() * 4 << std::endl;
        for (triangulation* scanner = &mesh_; scanner; scanner = (triangulation*)scanner->p_next()) {
            file << "3 " << scanner->item().vertex(0)->index() << " "
                 << scanner->item().vertex(1)->index() << " "
                 << scanner->item().vertex(2)->index() << std::endl;
        }

        // 3. cells type (5 for triangle)
        file << "CELL_TYPES " << mesh_.n_elements() << std::endl;
        for (auto i = 0; i < mesh_.n_elements(); ++i) {
            file << "5" << std::endl;
        }

        // 4. values at cells
        file << "CELL_DATA " << u_.size() << std::endl;
        file << "SCALARS U_Initial double 1" << std::endl;
        file << "LOOKUP_TABLE default" << std::endl;
        for (double val : u_) {
            file << val << std::endl;
        }

        file.close();
    }
};