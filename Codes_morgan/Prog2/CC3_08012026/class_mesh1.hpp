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

public:
    mesh() {}                         //  default constructor
    mesh(T& e) : linked_list<T>(e) {} //  constructor

    int indexing_vertices();

    int indexing_elements();

    void build_edges(std::size_t n_vertices_);
    void print_elements_indices() const;

    size_t n_vertices() { return n_vertices_; };
    size_t n_elements() { return n_elements_; };
    int find_neighbor_index(int index_triangle, vertex& v1, vertex& v2);
    
};

// shortcut for triangulation (2d)
typedef mesh<triangle> triangulation;

// CORRECTION : Ajout du template avant la définition
template<class T>
int mesh<T>::find_neighbor_index(int index_triangle, vertex& v1, vertex& v2)
{
    mesh<T>* target = nullptr;
    int current_idx = 0;
    for(mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        if(current_idx == index_triangle)
        {
            target = p_scan;
            break;
        }
        current_idx++; // CORRECTION : déplacé ici
    }
    
    if(!target) return -1; // CORRECTION : placé après la boucle

    current_idx = 0;
    for(mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        if(current_idx != index_triangle)
        {
            bool has_v1 = false, has_v2 = false; // CORRECTION : has_v1 au lieu de has_v_1
            for (int i = 0; i < 3; i++)
            {
                if(p_scan->item().vertex(i) == &v1) has_v1 = true;
                if(p_scan->item().vertex(i) == &v2) has_v2 = true;
            }
            if (has_v1 && has_v2)
            {
                return current_idx;
            }
           
        }
        current_idx++;
    }
    return -1;

}

// CORRECTION : Ajout du template avant la définition
template<class T>
void mesh<T>::print_elements_indices() const
{
    for (const mesh<T>* p_scan = this; p_scan; p_scan =(const mesh<T>*)p_scan->p_next())
    {
        std::cout << p_scan->item().index() << " "; // CORRECTION : std::cout (pas sdt::cout)
    }
    std::cout << std::endl;
}

template<class T>
int mesh<T>::indexing_elements()
{
    int count = 0;
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().set_index(count++);
    }
    return count; // CORRECTION : ajout du return
}

template <class T>
int mesh<T>::indexing_vertices()
{
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().reset_indices();
        
    }

    int count = 0;
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().indexing_vertices(count);
        
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
        return false; // CORRECTION : false au lieu de -1
    }

    std::string marker;
    std::string line_;

    while (marker.compare("Vertices"))
    {
        mesh_file >> marker;
        if (mesh_file.eof()) return false; // CORRECTION : gestion d'erreur
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
        if (mesh_file.eof()) return false; // CORRECTION : gestion d'erreur
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
        if (mesh_file.eof()) break; // CORRECTION : break au lieu de assert
    }

    mesh_file.close(); // Finished reading from the file stream.
    std::cout << "mesh_reader : mesh file loaded successfully" << std::endl;

    return true; // CORRECTION : true au lieu de 1
}

template <class T>
void mesh<T>::build_edges(std::size_t n_vertices)
{
    int n_elements = this->length();
    
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
                ismin    = scan_min->index();
                ismax    = scan_max->index();
            }

            if (!hashvd.item(ismin))
            {
                hashvd.item(ismin) = new linked_list<std::size_t>(ismin);
            }

            pc_cell_prev = hashvd.item(ismin);
            pc_cell      = hashvd.item(ismin)->p_next();

            while (pc_cell && (pc_cell->item() <= ismax))
            {
                pc_cell_prev = pc_cell;
                pc_cell      = pc_cell_prev->p_next();
            }

            if (pc_cell_prev->item() < ismax)
            {
                pc_cell_prev->insert_next_item(ismax);

                ++adj_vertices[ismin];
                ++n_edges;
                edges_.push_back(edge(*scan_min, *scan_max));
            }
            
        }
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