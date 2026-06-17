#pragma once

#include "class_node.hpp"
#include "class_point.hpp"

#include <iostream>

template <typename T, int NVERTICES>
class element
{
protected:
    node<T>* vertices_[NVERTICES];
    int index_;
    
public:
    element() : index_(-1) {
        for (int i = 0; i < NVERTICES; i++) {
            vertices_[i] = nullptr;
        }
    }                             //  default constructor
    
    element(node<T>& a, node<T>& b, node<T>& c) : index_(-1) { // for triangles
        this->vertices_[0] = &a;
        this->vertices_[1] = &b;
        this->vertices_[2] = &c;

        for (auto i = 0; i < NVERTICES; i++) {
            this->vertices_[i]->more_sharing_elements();
        }
    }
    
    element(element<T, NVERTICES> const& e) : index_(e.index_) {
        for (int i = 0; i < NVERTICES; i++) {
            this->vertices_[i] = e.vertices_[i];
            this->vertices_[i]->more_sharing_elements();
        }
    }
    
    node<T>& operator()(size_t i) { return *(this->vertices_[i]); }
    const node<T>& operator[](size_t i) const { return *(this->vertices_[i]); }

    node<T>* vertex(std::size_t i) const { return this->vertices_[i]; }
    node<T>*& vertex(std::size_t i) { return this->vertices_[i]; }

    int index() const { return index_; }
    int& index() { return index_; }
    void set_index(int idx) { index_ = idx; }  // <-- AJOUT IMPORTANT

    const element<T, NVERTICES>& operator=(element<T, NVERTICES>& e);
    ~element();

    void reset_indices();
    void indexing_vertices(int& count);

    template <typename S, int MVERTICES>
    friend std::ostream& operator<<(std::ostream& os, element<S, MVERTICES> const& e);
    void print_vertices_indices();

};

typedef element<point2d, 3> triangle;

template <typename T, int NVERTICES>
std::ostream& operator<<(std::ostream& os, element<T, NVERTICES> const& e)
{
    for (auto ii = 0; ii < NVERTICES; ii++)
        os << e[ii];
    return os;
}

template <typename T, int NVERTICES>
void element<T, NVERTICES>::print_vertices_indices()
{
    for (auto ii = 0; ii < NVERTICES; ii++)
        std::cout << "this->vertices_[" << ii << "]->index()= " << this->vertices_[ii]->index() << std::endl;
}

template <typename T, int NVERTICES>
const element<T, NVERTICES>& element<T, NVERTICES>::operator=(element<T, NVERTICES>& e)
{
    if (this != &e)
    {
        for (int i = 0; i < NVERTICES; i++)
            this->vertices_[i]->less_sharing_elements();

        for (int i = 0; i < NVERTICES; i++)
        {
            this->vertices_[i] = e.vertices_[i];
            this->vertices_[i]->more_sharing_elements();
        }
        index_ = e.index_;  // <-- COPIE DE L'INDEX
    }
    return *this;
}

template <typename T, int NVERTICES>
element<T, NVERTICES>::~element()
{
    for (int i = 0; i < NVERTICES; i++)
        this->vertices_[i]->less_sharing_elements();
}

template <typename T, int NVERTICES>
void element<T, NVERTICES>::reset_indices()
{
    for (int i = 0; i < NVERTICES; i++)
        this->vertices_[i]->index() = -1;
}

template <typename T, int NVERTICES>
void element<T, NVERTICES>::indexing_vertices(int& count)
{
    for (int i = 0; i < NVERTICES; i++)
    {
        if (this->vertices_[i]->index() < 0)
        {
            this->vertices_[i]->index() = count++;
        }
    }
}

template <typename T, int NVERTICES>
int operator<(const node<T>& n, const element<T, NVERTICES>& e)
{
    for (int i = 0; i < NVERTICES; i++)
        if (&n == &(e[i]))
            return i + 1;

    return 0;
}