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

    // easy data-structure to store edges
    std::vector<edge> edges_;

public:
    mesh() {}                         //  default constructor
    // mesh(T& e) : linked_list<T>(e) {} //  constructor

    mesh(std::vector<triangle>&, std::vector<vertex*>& vertices); // new constructor
    
    inline  double mesh_size() const { return mesh_size_; };  //  return min abs(edge)
    const std::vector<vertex*>& vertices() const { return vertices_; };


    int indexing_vertices();
    int indexing_elements();

    int compute_elements_areas();
    int compute_elements_centroids();
    int compute_faces_lengths();
    int compute_faces_outgoing_unit_normals();
    int populate_faces_neighbors_indices();

    void build_edges(std::size_t n_vertices_);

    size_t n_vertices() { return n_vertices_; };
    size_t n_elements() { return n_elements_; };
    
    // template<typename S>
    // friend std::ostream& operator<<(std::ostream& os, const mesh<S>& m);
};

// shortcut for triangulation (2d)
typedef mesh<triangle> triangulation;

template<class T>
mesh<T>::mesh(std::vector<triangle>& triangles, std::vector<vertex*>& vertices)
    : linked_list<T>(triangles(0)), n_elements_(triangles.size()), vertices_(vertices)
{
    for (auto new_triangle : triangles)
    {
        this->append(new_triangle);
        new_triangle.~triangle();
    }

    this->drop_first_item();

    n_vertices_ = vertices.size();

    this->indexing_elements();

    this->compute_elements_areas();
    this->compute_elements_centroids();
    this->compute_faces_lengths();
    this->compute_faces_outgoing_unit_normals();

    this->build_edges(n_vertices_);
    this->populate_faces_neighbors_indices();
}


template <class T>
int mesh<T>::populate_faces_neighbors_indices()
{
    for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next())
    {
        auto current_element_index = scanner->item().index();
        for (auto kk = 0; kk < 3; ++kk)
        {
            // scanner->item().neighbor(kk) = edges.at(scanner->item().face(kk)).neighbor(0) == current_element_index ? edges.at(scanner->item().face(kk)).neighbor(1) : edges.at(scanner->item().face(kk)).neighbor(0);
            scanner->item().set_neighbor_index(kk, edges_.at(scanner->item().face(kk)).neighbor(0) == current_element_index ? edges_.at(scanner->item().face(kk)).neighbor(1) : edges_.at(scanner->item().face(kk)).neighbor(0));
        }

        std::cout << "current element index= " << current_element_index << std::endl;
        std::cout << "indices of this triangle's neighbors through faces 0, 1, 2 (-1 means boundary face): " << scanner->item().neighbor(0) << " " << scanner->item().neighbor(1) << " " << scanner->item().neighbor(2) << " "
                  << std::endl;
    }

    return 1;
}

template <class T>
int mesh<T>::compute_faces_outgoing_unit_normals()
{
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        auto current_element_index = p_scan->item().index();
        p_scan->item().compute_unit_outgoing_normals();

        std::cout << "current element index= " << current_element_index << std::endl;
        std::cout << " normal to face 0 (nx, ny)= " << p_scan->item().normal(0) << std::endl;
        std::cout << " normal to face 1 (nx, ny)= " << p_scan->item().normal(1) << std::endl;
        std::cout << " normal to face 2 (nx, ny)= " << p_scan->item().normal(2) << std::endl;
    }
    return 1;
}

template <class T>
int mesh<T>::compute_elements_areas()
{
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().compute_area();
        std::cout << "element " << p_scan->item().index() << " area= " << p_scan->item().area() << std::endl;
        // p_scan->item().print_vertices_indices();
    }
    return 1;
}

template <class T>
int mesh<T>::compute_elements_centroids()
{
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().compute_centroid();
        std::cout << "element " << p_scan->item().index() << " (xG, yG)= " << p_scan->item().centroid() << std::endl;
        // p_scan->item().print_vertices_indices();
    }
    return 1;
}

template <class T>
int mesh<T>::compute_faces_lengths()
{
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().compute_faces_length();
        std::cout << "element " << p_scan->item().index() << " faces lengths= " << p_scan->item().faces_length(0) << " " << p_scan->item().faces_length(1) << " " << p_scan->item().faces_length(2) << std::endl;
        // p_scan->item().print_vertices_indices();
    }
    return 1;
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

class connectivity_data
{

protected:
    int vertex_index_;
    int edge_index_;

public:
    connectivity_data() : vertex_index_(-1), edge_index_(-1){};
    connectivity_data(int vv, int ed = -1) : vertex_index_(vv), edge_index_(ed){};
    inline int vertex_index() const { return vertex_index_; };
    inline int& vertex_index() { return vertex_index_; };
    inline int edge_index() const { return edge_index_; };
    inline int& edge_index() { return edge_index_; };
    };

template <class T>
void mesh<T>::build_edges(std::size_t n_vertices)
{
    int n_elements = this->length();
    std::cout << "n_vertices= " << n_vertices << std::endl;

    // data structure to store vertices-to-edges data
    list<linked_list<connectivity_data>> hashvd(n_vertices);

    std::vector<std::size_t> adj_vertices(n_vertices, 0);
    std::size_t n_edges = 0;

    // pointers to iterate through linked_list<size_t>
    linked_list<connectivity_data>* pc_cell      = nullptr;
    linked_list<connectivity_data>* pc_cell_prev = nullptr;

    // pointers to vertices
    vertex* scan_min = nullptr;
    vertex* scan_max = nullptr;

    size_t ismin;
    size_t ismax;

    size_t edges_counter{0};
    size_t elements_counter{0};

    // connectivity_data connect;

    // looping on elements->next
    for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next())
    {
        auto current_element_index  = scanner->item().index();
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
                hashvd.item(ismin) = new linked_list<connectivity_data>(ismin);
            }

            // std::cout << "  place pointers at the begining of the linked_list  " << std::endl;
            pc_cell_prev = hashvd.item(ismin);
            pc_cell      = hashvd.item(ismin)->p_next();

            while (pc_cell && (pc_cell->item().vertex_index() <= ismax))
            {
                pc_cell_prev = pc_cell;
                pc_cell      = pc_cell_prev->p_next();
            }

            if (pc_cell_prev->item().vertex_index() < ismax)
            {
                // std::cout << "  position found " << std::endl;
                // new edge is created / edge index is modified
                connectivity_data tmp(ismax, edges_counter);
                pc_cell_prev->insert_next_item(tmp);

                // add this edge as an element's face with local index kk
                scanner->item().face(kk) = edges_counter;

                ++adj_vertices[ismin];
                ++n_edges;
    
                edges_.push_back(edge(*scan_min, *scan_max, edges_counter++, current_element_index, -1, p_current_element));
            }
            else
            {
                // the edge alredy exists and we can identify its global index 
                // std::cout << "this edge is already accounted for" << std::endl;
                // std::cout << "and has index " << pc_cell_prev->item().edge_index() << std::endl;
                
                // thank's to connectivity_data, I can conveniently access the edge_index
                //.  and modify the neighboring index data
                edges_.at(pc_cell_prev->item().edge_index()).neighbor(1) = current_element_index;
                //.  as well as the adress of this neighbor
                edges_.at(pc_cell_prev->item().edge_index()).set_p_neighbor(1, p_current_element);

                // add this edge as an element's face with local index kk
                scanner->item().face(kk) = pc_cell_prev->item().edge_index();
            }
        }
    }

    // check faces as edges ont-to-one relation
    for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next())
    {
        auto current_element_index = scanner->item().index();
        std::cout << "current element index= " << current_element_index << std::endl;

        std::cout << "indices of faces as edges_ indices= " << scanner->item().face(0) << " "
                  << scanner->item().face(1) << " "
                  << scanner->item().face(2) << " " << std::endl;
    }

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

    std::cout << " -- exit build_edges -- " << std::endl<<std::endl;
} //  list of edges
