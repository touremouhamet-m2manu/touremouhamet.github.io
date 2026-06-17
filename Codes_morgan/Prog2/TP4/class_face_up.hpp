// class_face.hpp (updated for Exercice 2 and 7)
#pragma once
#include "class_element_up.hpp"
#include "class_node.hpp"
#include "class_point.hpp"
#include <array>
#include <cmath>
#include <iostream>

// underlying euclidian space dimension NDIM
// "face" models the mesh faces (NDIM-1)
// (if NDIM=2, faces are "edges")
// (if NDIM=3, faces are polygons, depending on the elements shapes)

template <typename T, int NVERTICES>
class face
{
protected:
    node<T>* vertices_[NVERTICES];
    int index_;
    std::array<int, NVERTICES> neighbors_; // indices of neighboring triangles (−1 if boundary edge)
    std::array<triangle
    point2d normal_; // arbitrary unit normal (for Exercice 7) 
    double length_; // face length (for Exercice 7)

public:
    face();                    // default constructor
    face(node<T>&, node<T>&);  // constructor for segments
    face(node<T>&, node<T>&, int ind, int tri_1, int tri_2); // new constructor
    face(face<T, NVERTICES> const&); // copy constructor
    const face<T, NVERTICES>& operator=(face<T, NVERTICES> const&);
    ~face();
    node<T>& operator()(size_t i) { return *(this->vertices_[i]); }             // read/write ith vertex
    const node<T>& operator[](size_t i) const { return *(this->vertices_[i]); } // read only ith vertex

    // New accessors
    inline int index() const { return index_; };
    inline void set_index(int i) { index_ = i; };
    inline int neighbor(size_t i) const { return neighbors_[i]; };
    inline void set_neighbor(size_t i, int jj) { neighbors_[i] = jj; };
    inline point2d normal() const { return normal_; };
    inline void set_normal(point2d n) { normal_ = n; };
    inline double length() const { return length_; };
    inline void set_length(double l) { length_ = l; };
};

// face for d=2
typedef face<point2d, 2> edge;

template <typename T, int NVERTICES>
face<T, NVERTICES>::face()
    : index_(-1), normal_(0.0, 0.0), length_(0.0)
{
    for (auto i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i] = nullptr;
        neighbors_[i] = -1;
    }
} // default constructor

template <typename T, int NVERTICES>
face<T, NVERTICES>::face(node<T>& a, node<T>& b)
    : index_(-1), normal_(0.0, 0.0), length_(0.0)
{
    this->vertices_[0] = &a;
    this->vertices_[1] = &b;

    for (auto i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i]->more_sharing_faces();
        neighbors_[i] = -1;
    }
} // constructor

template <typename T, int NVERTICES>
face<T, NVERTICES>::face(node<T>& a, node<T>& b, int ind, int tri_1, int tri_2)
    : index_(ind), normal_(0.0, 0.0), length_(0.0)
{
    this->vertices_[0] = &a;
    this->vertices_[1] = &b;
    neighbors_[0] = tri_1;
    neighbors_[1] = tri_2;

    for (auto i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i]->more_sharing_faces();
    }
} // new constructor

template <typename T, int NVERTICES>
face<T, NVERTICES>::face(face<T, NVERTICES> const& e)
    : index_(e.index_), neighbors_(e.neighbors_), normal_(e.normal_), length_(e.length_)
{
    for (auto i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i] = e.vertices_[i];
        this->vertices_[i]->more_sharing_faces();
    }
} // copy constructor

template <typename T, int NVERTICES>
const face<T, NVERTICES>& face<T, NVERTICES>::operator=(face<T, NVERTICES> const& e)
{
    if (this != &e)
    {
        for (auto i = 0; i < NVERTICES; i++)
        {
            this->vertices_[i] = e.vertices_[i];
            this->vertices_[i]->more_sharing_faces();
        }
        index_ = e.index_;
        neighbors_ = e.neighbors_;
        normal_ = e.normal_;
        length_ = e.length_;
    }
    return *this;
} // assignment operator

template <typename T, int NVERTICES>
face<T, NVERTICES>::~face()
{
    for (auto i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i]->less_sharing_faces();
    }
} // destructor