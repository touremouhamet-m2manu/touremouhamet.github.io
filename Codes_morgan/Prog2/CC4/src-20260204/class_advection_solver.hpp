#pragma once
#include "class_mesh.hpp"
#include <iostream>
#include <vector>

class advection_solver
{
protected:
    std::vector<double> u_;
    triangulation& mesh_;
    double dt;

public:
    triangulation& mesh() { return mesh_; };

    advection_solver(triangulation& mesh) : mesh_(mesh), dt_(0.0) {};

    void set_initial_data(point2d Xc, double amp, double sig)
    {
        u_.clear();
        for (triangulation* scanner = &mesh_; scanner; scanner = (triangulation*)scanner->p_next())
        {
            double x = scanner->item().centroid().x();
            double y = scanner->item().centroid().y();

            // Calcul de la distance au carré par rapport au centre
            double r_squared = std::pow(x - Xc.x(), 2) + std::pow(y - Xc.y(), 2);

            // Valeur de la gaussienne au centre du triangle
            u_.push_back(amp * std::exp(-r_squared / (2.0 * std::pow(sig, 2))));
        }
    }

    void advance(double T_final, point2d velocity)
    {
        double t{0.};
        int step{0};
        std::size_t n_elements = mesh_.n_elements();

        double dt = 0.5 * mesh().mesh_size() / sqrt(velocity * velocity);
        std::cout << "dt= " << dt << std::endl;

        std::vector<double> next_u(n_elements, 0.0); // Solution at t + dt

        set_initial_data(point2d(0., 0.), 1., 2.);
        export_to_VTK("initial.vtk");

        while (t < T_final)
        {
            if (t + dt > T_final)
                dt = T_final - t;

            int tri_index{0};

            for (triangulation* p_scan = &mesh(); p_scan; p_scan = (triangulation*)p_scan->p_next())
            {
                double flux_total = 0;

                for (auto kk = 0; kk < 3; ++kk)
                {
                    int neighbor_index = p_scan->item().neighbor(kk);

                    if (neighbor_index != -1)
                    {
                        double vn = velocity * p_scan->item().normal(kk);

                        // upwind flux
                        flux_total += vn > 0 ? vn * u_[tri_index] * p_scan->item().faces_length(kk) : vn * u_[neighbor_index] * p_scan->item().faces_length(kk);
                    }
                }

                next_u[tri_index] = u_[tri_index] - (dt / p_scan->item().area()) * flux_total;
                tri_index++;
            }

            // prepare next time iteration
            u_ = next_u;
            t += dt;
            if (++step % 1 == 0)
            {
                std::cout << "Temps---: " << t << std::endl;
            }
        }
        export_to_VTK("final.vtk");
    }

    void advance_edge(double T_final, point2d velocity)
    {
        double t = 0.0;
        int step = 0;

        mesh_.compute_edges_centroids();
        mesh_.compute_edges_unit_normals();
        double min_edge_length = mesh_.compute_edges_length();
        
        double v_norm = sqrt(velocity.x() * velocity.x() + velocity.y() * velocity.y());
        dt_ = 0.5 * min_edge_length / (v_norm + 1e-12);
        std::cout << "dt (edge-based) = " << dt_ << std::endl;
        
        if (u_.empty()) 
        {
            set_initial_data(point2d(0., 0.), 1., 2.);
        }
        
        std::cout << "Starting edge-based advection..." << std::endl;
        
        while (t < T_final) 
        {
            double dt_local = dt_;
            if (t + dt_local > T_final) 
            {
                dt_local = T_final - t;
            }

            std::vector<double> Phi(mesh_.n_elements(), 0.0);
            
            const auto& edges = mesh_.edges();
            for (const auto& e : edges) 
            {
               
                double vn = velocity.x() * e.normal().x() + velocity.y() * e.normal().y();

                int T_L = e.neighbor(0);
                int T_R = e.neighbor(1);
                
                double flux = 0.0;
                
                if (vn > 0) 
                {
                    flux = vn * u_[T_L] * e.length();
                } 
                else 
                {
                    if (T_R != -1) 
                    {
                        flux = vn * u_[T_R] * e.length();
                    } 
                    else 
                    {
                        flux = 0.0;
                    }
                }
                
                Phi[T_L] += flux;
                if (T_R != -1) 
                {
                    Phi[T_R] -= flux; 
                }
            }
            
            int tri_index = 0;
            for (triangulation* p_scan = &mesh_; p_scan; p_scan = (triangulation*)p_scan->p_next()) 
            {
                double area = p_scan->item().area();
                if (area > 1e-12) {
                    u_[tri_index] = u_[tri_index] - (dt_local / area) * Phi[tri_index];
                }
                tri_index++;
            }
            
            t += dt_local;
            step++;
            
            if (step % 10 == 0) 
            {
                std::cout << "Step " << step << ", t = " << t << std::endl;
            }
        }
        
        std::cout << "Edge-based advection completed after " << step << " steps." << std::endl;
    }

    void advance_edge_optimized(double T_final, point2d velocity)
    {
        double t = 0.0;
        int step = 0;
        
        mesh_.compute_edges_centroids();
        mesh_.compute_edges_unit_normals();
        double min_edge_length = mesh_.compute_edges_length();
        
        double v_norm = sqrt(velocity.x() * velocity.x() + velocity.y() * velocity.y());
        dt_ = 0.5 * min_edge_length / (v_norm + 1e-12);
        
        std::vector<double> inv_areas(mesh_.n_elements());
        int idx = 0;
        for (triangulation* p_scan = &mesh_; p_scan; p_scan = (triangulation*)p_scan->p_next()) 
        {
            double area = p_scan->item().area();
            inv_areas[idx] = (area > 1e-12) ? 1.0 / area : 0.0;
            idx++;
        }
        
        std::cout << "Starting OPTIMIZED edge-based advection..." << std::endl;
        
        while (t < T_final) 
        {
            double dt_local = dt_;
            if (t + dt_local > T_final) 
            {
                dt_local = T_final - t;
            }
            
            std::vector<double> Phi(mesh_.n_elements(), 0.0);
            
            const auto& edges = mesh_.edges();
            for (const auto& e : edges) 
            {
                double vn = velocity.x() * e.normal().x() + velocity.y() * e.normal().y();
                int T_L = e.neighbor(0);
                int T_R = e.neighbor(1);
                
                double flux = 0.0;
                if (vn > 0) {
                    flux = vn * u_[T_L] * e.length();
                } else if (T_R != -1) {
                    flux = vn * u_[T_R] * e.length();
                }
                
        
                if (T_R == -1) 
                {
                    u_[T_L] -= dt_local * inv_areas[T_L] * flux;
                } 
                else 
                {
                    u_[T_L] -= dt_local * inv_areas[T_L] * flux;
                    u_[T_R] += dt_local * inv_areas[T_R] * flux;
                }
            }
            
            t += dt_local;
            step++;
        }
        
        std::cout << "Optimized edge-based advection completed." << std::endl;
    }

    void export_to_VTK(std::string const& filename)
    {
        std::ofstream file(filename);
        if (!file.is_open())
            return;

        file << "# vtk DataFile Version 3.0" << std::endl;
        file << "Advection Initial Condition" << std::endl;
        file << "ASCII" << std::endl;
        file << "DATASET UNSTRUCTURED_GRID" << std::endl;

        // 1. Écriture des points (sommets)
        file << "POINTS " << mesh_.n_vertices() << " double" << std::endl;
        for (const auto& v : mesh_.vertices())
        {
            file << v->x() << " " << v->y() << " 0.0" << std::endl; // Z=0 pour la 2D
        }

        // 2. Écriture des cellules (triangles)
        // Format : CELLS [nb_cellules] [nb_total_entiers]
        // Pour chaque triangle : 3 (nb de sommets) + index1 + index2 + index3 = 4 entiers
        file << "CELLS " << mesh_.n_elements() << " " << mesh_.n_elements() * 4 << std::endl;

        for (triangulation* scanner = &mesh_; scanner; scanner = (triangulation*)scanner->p_next())
        {
            file << "3 " << scanner->item().vertex(0)->index() << " " << scanner->item().vertex(1)->index() << " " << scanner->item().vertex(2)->index() << std::endl;
        }
        // 3. Type de cellules (5 correspond au triangle dans VTK)
        file << "CELL_TYPES " << mesh_.n_elements() << std::endl;
        for (size_t i = 0; i < mesh_.n_elements(); ++i)
        {
            file << "5" << std::endl;
        }

        // 4. Données aux cellules
        file << "CELL_DATA " << u_.size() << std::endl;
        file << "SCALARS U_Initial double 1" << std::endl;
        file << "LOOKUP_TABLE default" << std::endl;
        for (double val : u_)
        {
            file << val << std::endl;
        }

        file.close();
        std::cout << "Fichier VTK genere : " << filename << std::endl;
    }

    
    const std::vector<double>& solution() const { return u_; }
    std::vector<double>& solution() { return u_; }
    

    double dt() const { return dt_; }
};