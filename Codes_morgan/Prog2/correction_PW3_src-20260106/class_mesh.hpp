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
private:
    std::size_t n_vertices_;
    std::size_t n_elements_;

public:
    mesh() {}                         //  default constructor
    mesh(T& e) : linked_list<T>(e) {} //  constructor

    int indexing();

    
    void build_edges(std::vector<edge>& edges, std::size_t n_vertices_);
    size_t n_vertices(){return n_vertices_;}
    size_t n_elements(){return n_elements_;}
    // bool mesh_reader(std::string const& fname);
    
    // template<typename S>
    // friend std::ostream& operator<<(std::ostream& os, const mesh<S>& m);
};

// shortcut for triangulation (2d)
typedef mesh<triangle> triangulation;

template <class T>
int mesh<T>::indexing()
{
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().reset_indices();
        // p_scan->item().print_vertices_indices();
    }

    int count = 0;
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().indexing(count);
        // p_scan->item().print_vertices_indices();
    }
    return count;
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

    // // Skip 7 first lines
    // for (int irow = 1; irow <= 7; ++irow)
    //     getline(mesh_file, line_);

    while (marker.compare("Vertices"))
    {
        mesh_file >> marker;
        assert(!mesh_file.eof());
    }

    int n_nodes;
    mesh_file >> n_nodes;
    // getline(mesh_file, line_);

    coords node;
    std::vector<point2d> points;

    for (auto in = 0; in < n_nodes; in++)
    {
        mesh_file >> node.x >> node.y >> node.index;
        points.push_back(point2d(node.x, node.y));
        // std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        // getline(mesh_file, line_);
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
    // // Skip 2 lines
    // for (int irow = 1; irow <= 2; ++irow)
    //     getline(mesh_file, line_);

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

// ajout

class connectivity_data
{

protected:
int vertex_index_;
int edge_index_;
int sharing_1_index_;
int sharing_2_index_;

public:
connectivity_data() : vertex_index_(-1), edge_index_(-1), sharing_1_index_(-1), sharing_2_index_(-1) {};
connectivity_data(int vv, int ed = -1, int e1 = -1, int e2 = -1) : vertex_index_(vv), edge_index_(ed), sharing_1_index_(e1), sharing_2_index_(e2) {};
inline int vertex_index() const { return vertex_index_; };
inline int& vertex_index() { return vertex_index_; };
inline int edge_index() const { return edge_index_; };
inline int& edge_index() { return edge_index_; };
inline int sharing_elements(size_t ii) const { return ii == 1 ? sharing_1_index_ : sharing_2_index_; };
inline int& sharing_elements(size_t ii) { return ii == 1 ? sharing_1_index_ : sharing_2_index_; };
};

template <class T>
void mesh<T>::build_edges(std::vector<edge>& edges, std::size_t n_vertices)
{
    int n_elements = this->length();
    std::cout << "n_vertices= " << n_vertices << std::endl;

    // data structure to store vertices-to-edges data
    // list<linked_list<size_t>> hashv(n_vertices);
    list<linked_list<connectivity_data>> hashv(n_vertices);


    std::vector<std::size_t> adj_vertices(n_vertices, 0);
    std::size_t n_edges = 0;

    // pointers to iterate through linked_list<size_t>
    linked_list<connectivity_data>* p_cell      = nullptr;
    linked_list<connectivity_data>* p_cell_prev = nullptr;

    // pointers to vertices
    vertex* scan_min = nullptr;
    vertex* scan_max = nullptr;

    size_t ismin;
    size_t ismax;

    // looping on elements->next
    for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next())
    {
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
                ismin    = scanner->item().vertex(next)->index();
                ismax    = scanner->item().vertex(kk)->index();
            }

            // std::cout << " ismin= " << ismin << std::endl;
            // std::cout << " ismax= " << ismax << std::endl;

            if (!hashv.item(ismin))
            {
                // std::cout << " allocate new linked_list for ismin= " << ismin << std::endl;
                hashv.item(ismin) = new linked_list<size_t>(ismin);
            }

            // std::cout << "  place pointers at the begining of the linked_list  " << std::endl;
            p_cell_prev = hashv.item(ismin);
            p_cell      = hashv.item(ismin)->p_next();

            while (p_cell && (p_cell->item() <= ismax))
            {
                p_cell_prev = p_cell;
                p_cell      = p_cell_prev->p_next();
            }

            if (p_cell_prev->item() < ismax)
            {
                // std::cout << "  position found " << std::endl;
                p_cell_prev->insert_next_item(ismax);

                ++adj_vertices[ismin];
                ++n_edges;

                edges.push_back(edge(*scan_min, *scan_max));
            }
        }
    }

    std::ofstream os;
    os.open("segment.txt");

    for (auto this_edge : edges)
    {
        os << this_edge[0].location() << std::endl;
        os << this_edge[1].location() << std::endl;
        os << std::endl;
    }
    os.close();

    std::cout << "n_edges= " << n_edges << std::endl << std::endl;

    std::cout << " -- exit build_edges -- " << std::endl;
} //  list of edges

