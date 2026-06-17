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
    //std::string meshFile = "ell.mesh";
    //std::string meshFile = "ex.mesh";
    bool res               = mesh_reader(meshFile, vertices, triangles);

    std::size_t n_vertices = vertices.size();
    std::size_t n_elements = triangles.size();

    std::cout << "number of vertices from mesh file= " << vertices.size() << std::endl;
    std::cout << "number of elements from mesh file= " << triangles.size() << std::endl;

    // form the mesh
    triangulation ell(triangles[0]); // first item will be doubled

    // populate mesh with triangles and free copied triangles
    for (auto new_triangle : triangles)
    {
        ell.append(new_triangle);
        new_triangle.~triangle();
    }

    ell.drop_first_item(); // first item is removed

    std::cout << "number of triangles from mesh structure= " << ell.length() << std::endl;
    std::cout << "number of triangles obtained from indexing in mesh structure= " << ell.indexing_elements() << std::endl;
    std::cout << "number of vertices from mesh structure= " << ell.indexing_vertices() << std::endl;

    int index = ell.find_neighbor_index(0, triangles[0](2), triangles[0](1));
    std::cout << "index neighbor of t0 for first face is = " << index << std::endl;
    index = ell.find_neighbor_index(0, triangles[0](1), triangles[0](0));
    std::cout << "index neighbor of t0 for second face is = " << index << std::endl;
    index = ell.find_neighbor_index(0, triangles[0](0), triangles[0](2));
    std::cout << "index neighbor of t0 for third face is = " << index << std::endl;

    index = ell.find_neighbor_index(1, triangles[1](2), triangles[1](1));
    std::cout << "index neighbor of t1 for first face is = " << index << std::endl;
    index = ell.find_neighbor_index(1, triangles[1](1), triangles[1](0));
    std::cout << "index neighbor of t1 for second face is = " << index << std::endl;
    index = ell.find_neighbor_index(1, triangles[1](0), triangles[1](2));
    std::cout << "index neighbor of t1 for third face is = " << index << std::endl;

    return 0;
}