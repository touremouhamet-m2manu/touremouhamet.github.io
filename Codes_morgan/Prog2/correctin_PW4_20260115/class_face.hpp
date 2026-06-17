#pragma once

#include "class_element.hpp"
#include "class_node.hpp"
#include "class_point.hpp"

#include <iostream>

template <typename T, int NVERTICES>
class face
{
protected:
    node<T>* vertices_[NVERTICES];
    int index_;

    std::array<int, NVERTICES> neighbors_;         // indices des triangles adjacents (-1 si bord)
    std::array<triangle*, NVERTICES> p_neighbors_; // adresses des triangles adjacents (nullptr si bord)
    
public:
    face();                                                                                      // default constructor
    face(node<T>&, node<T>&, int ind = -1, int tri_1 = -1, int tri_2 = -1, triangle* = nullptr); // constructor for segments
    face(face<T, NVERTICES> const&);                                                             // copy constructor
    const face<T, NVERTICES>& operator=(face<T, NVERTICES> const&);
    ~face();
    node<T>& operator()(size_t i) { return *(this->vertices_[i]); }             //  read/write ith vertex
    const node<T>& operator[](size_t i) const { return *(this->vertices_[i]); } //  read only ith vertex

    inline int index() const { return index_; };
    inline int& index() { return index_; };

    inline int neighbor(std::size_t i) const { return this->neighbors_[i]; };
    inline int& neighbor(std::size_t i) { return this->neighbors_[i]; };

    inline triangle* p_neighbor(std::size_t i) const { return this->p_neighbors_[i]; };
    inline void set_p_neighbor(std::size_t i, triangle* pt) { p_neighbors_[i] = pt; };
};

// face for d=2
typedef face<point2d, 2> edge;

template <typename T, int NVERTICES>
face<T, NVERTICES>::face()
{
    for (auto i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i] = nullptr;
    }
} //  default constructor

template <typename T, int NVERTICES>
face<T, NVERTICES>::face(node<T>& a, node<T>& b, int ind, int tri_1, int tri_2, triangle* pt) : index_(ind), neighbors_{tri_1, tri_2}, p_neighbors_{pt, nullptr}
{
    this->vertices_[0] = &a;
    this->vertices_[1] = &b;

    for (auto i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i]->more_sharing_faces();
    }
} //  constructor

template <typename T, int NVERTICES>
face<T, NVERTICES>::face(face<T, NVERTICES> const& e) : index_(e.index_), neighbors_(e.neighbors_), p_neighbors_(e.p_neighbors_)
{
    for (auto i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i] = e.vertices_[i];
        this->vertices_[i]->more_sharing_faces();
        // std::cout << "moreSharingFaces called in copy constructor" << std::endl;
    }
} //  copy constructor

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

        this->index_       = e.index();
        this->neighbors_   = e.neighbors_;
        this->p_neighbors_ = e.p_neighbors_;
    }
    return *this;
} //  assignment operator

template <typename T, int NVERTICES>
face<T, NVERTICES>::~face()
{
    for (auto i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i]->less_sharing_faces();
    }
} //   destructor
