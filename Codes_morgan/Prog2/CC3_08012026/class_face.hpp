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

public:
    face();                          // default constructor
    face(node<T>&, node<T>&);        // constructor for segments
    face(face<T, NVERTICES> const&); // copy constructor
    const face<T, NVERTICES>& operator=(face<T, NVERTICES> const&);
    ~face();
    node<T>& operator()(size_t i) { return *(this->vertices_[i]); }             //  read/write ith vertex
    const node<T>& operator[](size_t i) const { return *(this->vertices_[i]); } //  read only ith vertex

    double x() const {return vertices_[0]->location_[0];};
    double y() const {return vertices_[1]->location_[1];};
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
face<T, NVERTICES>::face(node<T>& a, node<T>& b)
{
    this->vertices_[0] = &a;
    this->vertices_[1] = &b;

    for (auto i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i]->more_sharing_faces();
    }
} //  constructor

template <typename T, int NVERTICES>
face<T, NVERTICES>::face(face<T, NVERTICES> const& e)
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
