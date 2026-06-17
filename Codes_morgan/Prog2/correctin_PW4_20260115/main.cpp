#include "class_element.hpp"
#include "class_mesh.hpp"
#include "class_node.hpp"
#include "class_point.hpp"

#include <iostream>

int main()
{
    std::vector<vertex*> vertices;
    std::vector<triangle> triangles;

    // loads mesh data from .mesh file
    std::string meshFile = "first.mesh";
    // std::string meshFile = "ell.mesh";
    //std::string meshFile = "ex.mesh";

    bool res               = mesh_reader(meshFile, vertices, triangles);

    std::size_t n_vertices = vertices.size();
    std::size_t n_elements = triangles.size();

    std::cout << "number of vertices= " << vertices.size() << std::endl;
    std::cout << "number of elements= " << triangles.size() << std::endl;

    // form the mesh
    triangulation ell(triangles[0]); // first item will be doubled

    // populate mesh with triangles and free copied triangles
    for (auto new_triangle : triangles)
    {
        ell.append(new_triangle);
        new_triangle.~triangle();
    }
    ell.drop_first_item(); // first item is removed

    std::cout << "number of triangles= " << ell.indexing_elements() << std::endl;
    std::cout << "number of vertices = " << ell.indexing_vertices() << std::endl;

    ell.compute_elements_areas();
    ell.compute_elements_centroids();
    ell.compute_faces_lengths();
    ell.compute_faces_outgoing_unit_normals();

    ell.build_edges(ell.indexing_vertices());

    ell.populate_faces_neighbors_indices();

    return 0;
}