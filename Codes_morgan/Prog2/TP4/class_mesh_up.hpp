// class_mesh.hpp (updated for Exercice 2,3,4,7)
#pragma once
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "class_element_up.hpp"
#include "class_face_up.hpp"
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

// Ad-hoc structure for connectivity
class connectivity_data
{
protected:
    int vertex_index_;
    int edge_index_;
    int sharing_1_index_;
    int sharing_2_index_;
public:
    connectivity_data()
        : vertex_index_(-1), edge_index_(-1), sharing_1_index_(-1), sharing_2_index_(-1) {}
    connectivity_data(int vv, int ed = -1, int e1 = -1, int e2 = -1)
        : vertex_index_(vv), edge_index_(ed), sharing_1_index_(e1), sharing_2_index_(e2) {}
    inline int vertex_index() const { return vertex_index_; };
    inline void set_vertex_index(int value) { vertex_index_ = value; };
    inline int edge_index() const { return edge_index_; };
    inline void set_edge_index(int value) { edge_index_ = value; };
    inline int sharing_1_index() const { return sharing_1_index_; };
    inline void set_sharing_1_index(int value) { sharing_1_index_ = value; };
    inline int sharing_2_index() const { return sharing_2_index_; };
    inline void set_sharing_2_index(int value) { sharing_2_index_ = value; };
};

template <class T>
class mesh : public linked_list<T>
{
private:
    std::vector<edge> edges_; // Composition
    std::vector<vertex*>& vertices_; // Aggregation (reference)
    std::vector<triangle> triangles_; // Composition (value, copy from input)
    std::size_t n_vertices_;
    std::size_t n_elements_;
    double mesh_size_; // min edge length

public:
    mesh(std::vector<triangle>& triangles, std::vector<vertex*>& vertices);
    mesh() = delete; // No default

    int index_vertices();
    int index_elements();
    void build_edges();
    int compute_elements_areas();
    int compute_elements_centroids();
    int compute_faces_lengths();
    int compute_faces_outgoing_unit_normals();
    int compute_edges_normals_and_lengths();
    int compute_mesh_size();
    int populate_faces_neighbors_indices();

    const std::vector<edge>& edges() const { return edges_; };
    std::vector<edge>& edges() { return edges_; };
    const std::vector<vertex*>& vertices() const { return vertices_; };
    const std::vector<triangle>& triangles() const { return triangles_; };
    size_t n_vertices() const { return n_vertices_; }
    size_t n_elements() const { return n_elements_; }
    double mesh_size() const { return mesh_size_; }
};

// shortcut for triangulation (2d)
typedef mesh<triangle> triangulation;

template <class T>
mesh<T>::mesh(std::vector<triangle>& triangles, std::vector<vertex*>& vertices)
    : linked_list<T>(triangles[0]), vertices_(vertices), n_vertices_(vertices.size()),
      n_elements_(triangles.size()), mesh_size_(0.0)
{
    // Append remaining triangles to the linked list
    for (size_t i = 1; i < triangles.size(); ++i) {
        this->append(triangles[i]);
    }
    // Copy triangles for vector
    triangles_ = triangles; // Note: this copies, if large, consider move or ref
}

template <class T>
int mesh<T>::index_vertices()
{
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().reset_indices();
    }

    int count = 0;
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().indexing(count);
    }
    return count;
} // indexing the nodes in the mesh

template <class T>
int mesh<T>::index_elements() {
    int count = 0;
    for (mesh<T>* p_scan = this; p_scan; p_scan = (mesh<T>*)p_scan->p_next())
    {
        p_scan->item().set_index(count++);
    }
    return count;
}

// populate vertices and triangles from file's data
bool mesh_reader(std::string const& fname,
                 std::vector<vertex*>& vertices,
                 std::vector<triangle>& triangles)
{
    // ... (unchanged from provided)
}

template <class T>
void mesh<T>::build_edges()
{
    edges_.clear();
    int edges_counter = 0;
    list<linked_list<connectivity_data>> hashv(n_vertices_);

    // looping on elements->next
    for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next())
    {
        auto current_element_index = scanner->item().index();

        for (auto kk = 0; kk < 3; kk++)
        {
            size_t next = (kk + 1) % 3;
            vertex* scan_min;
            vertex* scan_max;
            size_t ismin, ismax;

            if (scanner->item().vertex(kk)->index() <= scanner->item().vertex(next)->index())
            {
                scan_min = scanner->item().vertex(kk);
                scan_max = scanner->item().vertex(next);
                ismin = scan_min->index();
                ismax = scan_max->index();
            }
            else
            {
                scan_min = scanner->item().vertex(next);
                scan_max = scanner->item().vertex(kk);
                ismin = scanner->item().vertex(next)->index();
                ismax = scanner->item().vertex(kk)->index();
            }

            if (!hashv.item(ismin))
            {
                hashv.item(ismin) = new linked_list<connectivity_data>(connectivity_data(ismin));
            }

            linked_list<connectivity_data>* p_cell_prev = hashv.item(ismin);
            linked_list<connectivity_data>* p_cell = hashv.item(ismin)->p_next();

            while (p_cell && (p_cell->item().vertex_index() <= ismax))
            {
                p_cell_prev = p_cell;
                p_cell = p_cell_prev->p_next();
            }

            if (p_cell_prev->item().vertex_index() < ismax)
            {
                // position found
                connectivity_data tmp(ismax, edges_counter, current_element_index);
                p_cell_prev->insert_next_item(tmp);

                // add this edge as an element’s face with local index kk
                scanner->item().set_face_index(kk, edges_counter);

                edges_.push_back(
                    edge(*scan_min, *scan_max, edges_counter, current_element_index, -1));
                edges_counter++;
            }
            else
            {
                // edge already exists
                edges_.at(p_cell_prev->item().edge_index()).set_neighbor(1, current_element_index);

                // also, add this edge index as an element’s face with local index kk
                scanner->item().set_face_index(kk, p_cell_prev->item().edge_index());
            }
        }
    }
}

template <class T>
int mesh<T>::compute_elements_areas() {
    for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next()) {
        scanner->item().compute_area();
    }
    return 1;
}

template <class T>
int mesh<T>::compute_elements_centroids() {
    for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next()) {
        scanner->item().compute_centroid();
    }
    return 1;
}

template <class T>
int mesh<T>::compute_faces_lengths() {
    for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next()) {
        scanner->item().compute_faces_length();
    }
    return 1;
}

template <class T>
int mesh<T>::compute_faces_outgoing_unit_normals() {
    for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next()) {
        scanner->item().compute_outgoing_unit_normals();
    }
    return 1;
}

template <class T>
int mesh<T>::compute_edges_normals_and_lengths() {
    for (auto& e : edges_) {
        int t = e.neighbor(0);
        if (t >= 0) {
            for (int kk = 0; kk < 3; ++kk) {
                if (triangles_[t].face(kk) == e.index()) {
                    e.set_normal(triangles_[t].normal(kk));
                    e.set_length(triangles_[t].face_length(kk));
                    break;
                }
            }
        }
    }
    return 1;
}

template <class T>
int mesh<T>::compute_mesh_size() {
    if (edges_.empty()) return 0;
    mesh_size_ = edges_[0].length();
    for (const auto& e : edges_) {
        mesh_size_ = std::min(mesh_size_, e.length());
    }
    return 1;
}

template <class T>
int mesh<T>::populate_faces_neighbors_indices()  
{
    for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next())
    {
        auto current_element_index = scanner->item().index();

        for (auto kk = 0; kk < 3; ++kk)
        {
            int face_idx = scanner->item().face(kk);
            if (face_idx >= 0) {
                scanner->item().set_neighbor_index(kk,
                    edges_[face_idx].neighbor(0) == current_element_index ?
                    edges_[face_idx].neighbor(1) :
                    edges_[face_idx].neighbor(0));
            }
        }
    }
    return 1;
}

template<class T>
int mesh<T>::populate_faces_neighbors_indices()
{
    for (mesh<T>* scanner = this; scanner; scanner = (mesh<T>*)scanner->p_next())
    for (auto kk = 0; kk < 3; ++kk)
    {
        
    }

}