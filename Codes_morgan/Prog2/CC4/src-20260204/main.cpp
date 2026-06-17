#include "class_element.hpp"
#include "class_mesh.hpp"
#include "class_node.hpp"
#include "class_point.hpp"
#include "class_advection_solver.hpp"

#include <iostream>

int main()
{

    std::vector<vertex*> vertices;
    std::vector<triangle> triangles;

    // loads mesh data from .mesh file
    std::string meshFile = "first.mesh";
    //std::string meshFile = "ex.mesh";

    bool res               = mesh_reader(meshFile, vertices, triangles);

    std::size_t n_vertices = vertices.size();
    std::size_t n_elements = triangles.size();

    std::cout << "number of vertices= " << vertices.size() << std::endl;
    std::cout << "number of elements= " << triangles.size() << std::endl;

    triangulation mesh(triangles, vertices);
    
    return 0;
}