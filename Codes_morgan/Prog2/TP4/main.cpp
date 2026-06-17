// main.cpp example (driver)
#include "class_mesh_up.hpp"
#include "advection_solver.hpp" // Assume the new file
#include <vector>

int main() {
    std::vector<vertex*> vertices;
    std::vector<triangle> triangles;

    // load mesh data from .mesh file
    std::string meshFile = "ex.mesh";
    bool res = mesh_reader(meshFile, vertices, triangles);

    triangulation mesh(triangles, vertices);

    // Build and compute all necessary data
    mesh.index_vertices();
    mesh.index_elements();
    mesh.build_edges();
    mesh.compute_elements_areas();
    mesh.compute_elements_centroids();
    mesh.compute_faces_lengths();
    mesh.compute_faces_outgoing_unit_normals();
    mesh.populate_faces_neighbors_indices();
    mesh.compute_edges_normals_and_lengths();
    mesh.compute_mesh_size();

    advection_solver gaussienne(mesh);

    point2d Xc(0.0, 0.0);
    double amp = 1.0;
    double sig = 2.0; // As per PDF, but adjust if needed
    gaussienne.set_initial_data(Xc, amp, sig);

    point2d velocity(1.0, 0.0);
    double Tmax = 3.0;
    gaussienne.advance(Tmax, velocity);

    gaussienne.export_to_VTK("solution.vtk");

    return 0;
}