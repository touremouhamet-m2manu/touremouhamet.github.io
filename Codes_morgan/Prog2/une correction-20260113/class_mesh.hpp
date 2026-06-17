#pragma once
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "class_element.hpp"
#include "class_face.hpp"
#include "class_linked_list.hpp"
#include "class_list.hpp"
#include "class_node.hpp"

// temp structures for storing read values from files
typedef std::array<int, 3> itriplet;

struct coords
{
    double x;
    double y;
    int index;
};

template <class T>
class mesh : public linked_list<T>
{
protected:
    std::size_t n_vertices_;
    std::size_t n_elements_;

    // easy data-structure for FV needs
    std::vector<edge> edges_;

    // arbitrary oriented (but outgoing for boundary edges)
    //std::vector<point2d> edges_unit_normals_;

public:
    mesh() {}                         //  default constructor
    mesh(T& e) : linked_list<T>(e) {} //  constructor

    int indexing_vertices();
    int indexing_elements();

    void build_edges(std::size_t n_vertices_);

    size_t n_vertices() { return n_vertices_; };
    size_t n_elements() { return n_elements_; };

    int find_neighbor_index(int index_triangle, vertex& nI, vertex& nJ);
};

// shortcut for triangulation (2d)
typedef mesh<triangle> triangulation;

template <>
int triangulation::find_neighbor_index(int index_triangle, vertex& nI, vertex& nJ)
{
    int index_neighbor = -1; // meaningless value meaning "hey, I'm on the boundary !!"

    // scan all the mesh elements
    for (triangulation* scanner = this; scanner; scanner = (triangulation*)scanner->p_next())
    {
        // the arguments ”nI” and ”nJ” are the node objects corresponding
        // to the endpoints nI and nJ respectively.
        // If both nI and nJ are indeed vertices in the first triangle in the mesh,
        // 'item', then the integers 'ni' and ”nj” are the corresponding indices
        // of nI and nJ in the list of vertices in ”item” plus 1.
        int ni = nI < scanner->item();
        int nj = nJ < scanner->item();

        // These integers are now used to identify the neighbor of *this:
        if (ni && nj && (index_triangle != scanner->item().index()))
        {
            index_neighbor = scanner->item().index();
            break;
        }
        // If the two vertices are indeed in the triangle "item” then one should
        //  check if this triangle is not equal to this
        // if not, then "item” is indeed the required neighbor triangle
        // that shares both nI and nJ
    }
    return index_neighbor;
}

template <class T>
int mesh<T>::indexing_vertices()
{
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().reset_indices();
        // p_scan->item().print_vertices_indices();
    }

    int count = 0;
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().indexing_vertices(count);
        // p_scan->item().print_vertices_indices();
    }
    return count;
} //  indexing the nodes in the mesh

template <class T>
int mesh<T>::indexing_elements()
{
    int count = 0;
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().set_index(count++);
        // p_scan->item().print_vertices_indices();
    }
    return count--;
} //  indexing the nodes in the mesh

// populate vertices and triangles from file's data
bool mesh_reader(std::string const& fname,
                 std::vector<vertex*>& vertices,
                 std::vector<triangle>& triangles)
{
    std::ifstream mesh_file(fname, std::ios::in);
    if (!mesh_file)
    {
        std::cerr << "mesh_reader ==> failed to load mesh file" << std::endl;
        return -1;
    }

    std::string marker;
    std::string line_;

    while (marker.compare("Vertices"))
    {
        mesh_file >> marker;
        assert(!mesh_file.eof());
    }

    int n_nodes;
    mesh_file >> n_nodes;
    
    coords node;
    std::vector<point2d> points;

    for (auto in = 0; in < n_nodes; in++)
    {
        mesh_file >> node.x >> node.y >> node.index;
        points.push_back(point2d(node.x, node.y));
    }

    // form vertices from points
    for (point2d pt : points)
        vertices.push_back(new vertex(pt));

    // set indices
    for (auto iv = 0; iv < n_nodes; iv++)
        vertices[iv]->index() = iv;

    while (marker.compare("Triangles"))
    {
        mesh_file >> marker;
        assert(!mesh_file.eof());
    }
    
    int n_elements;
    mesh_file >> n_elements;

    int n1; // useless label for the time being

    std::vector<itriplet> triplets;

    for (auto ie = 0; ie < n_elements; ie++)
    {
        itriplet tmp;
        mesh_file >> tmp[0] >> tmp[1] >> tmp[2] >> n1;

        // set 0-based indices
        tmp[0]--;
        tmp[1]--;
        tmp[2]--;
        // cout << tmp[0] << " " << tmp[1] << " " << tmp[2] <<endl;
        triplets.push_back(tmp);
        // std::cout << " triplet " << tmp[0] << tmp[1] << tmp[2] << std::endl;
    }

    // form triangles from triplets
    for (itriplet it : triplets)
    {
        triangles.push_back(
            triangle(*vertices[it[0]], *vertices[it[1]], *vertices[it[2]]));
    }

    while (marker.compare("End"))
    {
        mesh_file >> marker;
        assert(!mesh_file.eof());
    }

    mesh_file.close(); // Finished reading from the file stream.
    std::cout << "mesh_reader : mesh file loaded successfully" << std::endl;

    return 1; // mesh data loaded successfully
}



template <class T>
void mesh<T>::build_edges(std::size_t n_vertices)
{
    int n_elements = this->length();
    std::cout << "n_vertices= " << n_vertices << std::endl;

    // data structure to store vertices-to-edges data
    list<linked_list<std::size_t>> hashvd(n_vertices);

    std::vector<std::size_t> adj_vertices(n_vertices, 0);
    std::size_t n_edges = 0;

    // pointers to iterate through linked_list<size_t>
    linked_list<std::size_t>* pc_cell      = nullptr;
    linked_list<std::size_t>* pc_cell_prev = nullptr;

    // pointers to vertices
    vertex* scan_min = nullptr;
    vertex* scan_max = nullptr;

    size_t ismin;
    size_t ismax;

    size_t edges_counter{0};
    //size_t elements_counter{0};

    // connectivity_data connect;

    // looping on elements->next
    for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next())
    {
        //auto current_element_index  = scanner->item().index();
        triangle* p_current_element = &scanner->item();

        for (auto kk = 0; kk < 3; kk++)
        {
            size_t next = (kk + 1) % 3;

            if (scanner->item().vertex(kk)->index() <= scanner->item().vertex(next)->index())
            {
                scan_min = scanner->item().vertex(kk);
                scan_max = scanner->item().vertex(next);
                ismin    = scan_min->index();
                ismax    = scan_max->index();
            }
            else
            {
                scan_min = scanner->item().vertex(next);
                scan_max = scanner->item().vertex(kk);
                ismin    = scan_min->index();
                ismax    = scan_max->index();
            }

            // std::cout << " ismin= " << ismin << std::endl;
            // std::cout << " ismax= " << ismax << std::endl;

            if (!hashvd.item(ismin))
            {
                hashvd.item(ismin) = new linked_list<std::size_t>(ismin);
            }

            // std::cout << "  place pointers at the begining of the linked_list  " << std::endl;
            pc_cell_prev = hashvd.item(ismin);
            pc_cell      = hashvd.item(ismin)->p_next();

            while (pc_cell && (pc_cell->item() <= ismax))
            {
                pc_cell_prev = pc_cell;
                pc_cell      = pc_cell_prev->p_next();
            }

            if (pc_cell_prev->item() < ismax)
            {
                // std::cout << "  position found " << std::endl;
                // new edge is created / edge index is modified
                //connectivity_data tmp(ismax, edges_counter, current_element_index);
                pc_cell_prev->insert_next_item(ismax);

                // add this edge as an element's face with local index kk
                //scanner->item().face(kk) = edges_counter;

                ++adj_vertices[ismin];
                ++n_edges;
                // std::cout << "edges_counter= " << edges_counter << std::endl;
                // std::cout << "current_element_index= " << current_element_index << std::endl;
                edges_.push_back(edge(*scan_min, *scan_max));
            }
            else
            {
                
            }
        }
    }

    // a pass on edges_ to compute their length and normals
    // for (auto& this_edge : edges_)
    // {
    //     this_edge.compute_length();
    //     std::cout << "edge with index= " << this_edge.index() << " has a length of " << this_edge.length() << std::endl << std::endl;

    //     this_edge.compute_centroid();
    //     std::cout << "edge with index= " << this_edge.index() << " has a centroid (xGF, yGF)= " << this_edge.centroid() << std::endl
    //               << std::endl;

    //     this_edge.compute_unit_normal();
    //     std::cout << "edge with index= " << this_edge.index() << " has a normal (nx, ny)= " << this_edge.normal() << std::endl
    //               << std::endl;
    // }

    // // check faces as edges ont-to-one relation
    // for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next())
    // {
    //     auto current_element_index = scanner->item().index();
    //     std::cout << "current element index= " << current_element_index << std::endl;

    //     std::cout << "indices of faces as edges_ indices= " << scanner->item().face(0) << " "
    //               << scanner->item().face(1) << " "
    //               << scanner->item().face(2) << " " << std::endl;
    // }

    std::ofstream os;
    os.open("segment.txt");

    for (auto& this_edge : edges_)
    {
        os << this_edge[0].location() << std::endl;
        os << this_edge[1].location() << std::endl;
        os << std::endl;
    }
    os.close();

    std::cout << "n_edges= " << n_edges << std::endl;

    //std::cout << "n_edges (bis)= " << edges_counter << std::endl << std::endl;

    std::cout << " -- exit build_edges -- " << std::endl<<std::endl;
} //  list of edges
