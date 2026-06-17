// class_element.hpp (updated for Exercice 1)
#pragma once

#include "class_node.hpp"
#include "class_point.hpp"
#include <array>
#include <cmath>
#include <iostream>

typedef point<2> point2d; // Assuming from class_point.hpp

// underlying euclidian space dimension NDIM
// "Element" models the mesh elements (NDIM)

template <typename T, int NVERTICES>
class element
{
protected:
    node<T>* vertices_[NVERTICES];
    int index_;

    // geometry (element dependent)
    double area_; // value of the triangle area
    point2d centroid_; // coordinates of the centroid
    std::array<point2d, NVERTICES> normals_; // coordinates of the outgoing unit normals to faces
    std::array<double, NVERTICES> faces_length_; // faces lengths

    // global indices of neighboring triangles (−1 if boundary face)
    std::array<int, NVERTICES> neighbors_;

    // global indices of triangle faces seen as edges (−1 if boundary face)
    std::array<int, NVERTICES> faces_;

public:
    element();                             // default constructor
    element(node<T>&, node<T>&, node<T>&); // for triangles
    element(element<T, NVERTICES> const&);
    // const is needed in copy-cst arg for const-compatibility
    // with the copy-cstr of linked_list object

    node<T>& operator()(size_t i) { return *(this->vertices_[i]); }             // read/write ith vertex
    const node<T>& operator[](size_t i) const { return *(this->vertices_[i]); } // read only ith vertex
    node<T>* vertex(std::size_t i) const { return this->vertices_[i]; };               // get ith vertex address (pointer)
    node<T>*& vertex(std::size_t i) { return this->vertices_[i]; };                        // get ith vertex address (pointer)
    
    const element<T, NVERTICES>& operator=(element<T, NVERTICES>&);
    ~element();
    void reset_indices();      // reset indices to -1
    void indexing(int& count); // indexing the vertices

    // New methods and accessors
    inline int index() const { return index_; };
    inline void set_index(int i) { index_ = i; };

    void compute_area();
    double area() const { return area_; };

    void compute_centroid();
    point2d centroid() const { return centroid_; };

    void compute_faces_length();
    double face_length(size_t i) const { return faces_length_[i]; };

    void compute_outgoing_unit_normals();
    point2d normal(size_t i) const { return normals_[i]; };

    inline int neighbor(std::size_t ii) const { return this->neighbors_[ii]; };
    inline void set_neighbor_index(std::size_t ii, int jj) { this->neighbors_[ii] = jj; };
    inline int face(std::size_t ii) const { return this->faces_[ii]; };
    inline void set_face_index(std::size_t ii, int jj) { this->faces_[ii] = jj; };

    template <typename S, int MVERTICES>
    friend std::ostream& operator<<(std::ostream& os, element<S, MVERTICES> const& e);
    void print_vertices_indices();
};

// element for d=2
typedef element<point2d, 3> triangle;

template <typename T, int NVERTICES>
std::ostream& operator<<(std::ostream& os, element<T, NVERTICES> const& e)
{
    for (auto ii = 0; ii < NVERTICES; ii++)
        os << e[ii];
    return os;
} //

template <typename T, int NVERTICES>
void element<T, NVERTICES>::print_vertices_indices()
{
    for (auto ii = 0; ii < NVERTICES; ii++)
        std::cout << "this->vertices_[i]->index()= " << this->vertices_[ii]->index() << std::endl;
}

template <typename T, int NVERTICES>
element<T, NVERTICES>::element()
    : index_(-1), area_(0.0), centroid_(0.0, 0.0)
{
    for (auto i = 0; i < NVERTICES; i++) {
        vertices_[i] = nullptr;
        neighbors_[i] = -1;
        faces_[i] = -1;
        normals_[i] = point2d(0.0, 0.0);
        faces_length_[i] = 0.0;
    }
}

template <typename T, int NVERTICES>
element<T, NVERTICES>::element(node<T>& a, node<T>& b, node<T>& c)
    : index_(-1), area_(0.0), centroid_(0.0, 0.0)
{
    this->vertices_[0] = &a;
    this->vertices_[1] = &b;
    this->vertices_[2] = &c;

    for (auto i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i]->more_sharing_elements();
        neighbors_[i] = -1;
        faces_[i] = -1;
        normals_[i] = point2d(0.0, 0.0);
        faces_length_[i] = 0.0;
    }
} // constructor

template <typename T, int NVERTICES>
element<T, NVERTICES>::element(element<T, NVERTICES> const& e)
    : index_(e.index_), area_(e.area_), centroid_(e.centroid_), normals_(e.normals_), faces_length_(e.faces_length_),
      neighbors_(e.neighbors_), faces_(e.faces_)
{
    for (int i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i] = e.vertices_[i];
        this->vertices_[i]->more_sharing_elements();
    }
} // copy constructor

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
        index_ = e.index_;
        area_ = e.area_;
        centroid_ = e.centroid_;
        normals_ = e.normals_;
        faces_length_ = e.faces_length_;
        neighbors_ = e.neighbors_;
        faces_ = e.faces_;
    }
    return *this;
} // assignment operator

template <typename T, int NVERTICES>
element<T, NVERTICES>::~element()
{
    for (int i = 0; i < NVERTICES; i++)
        this->vertices_[i]->less_sharing_elements();
} // destructor

template <typename T, int NVERTICES>
void element<T, NVERTICES>::reset_indices()
{
    for (int i = 0; i < NVERTICES; i++)
        this->vertices_[i]->index() = -1;
} // reset indices to -1

template <typename T, int NVERTICES>
void element<T, NVERTICES>::indexing(int& count)
{
    for (int i = 0; i < NVERTICES; i++)
    {
        if (this->vertices_[i]->index() < 0)
        {
            this->vertices_[i]->index() = count++;
        }
    }
} // indexing the vertices

template <typename T, int NVERTICES>
int operator<(const node<T>& n, const element<T, NVERTICES>& e)
{
    for (int i = 0; i < NVERTICES; i++)
        if (&n == &(e[i]))
            return i + 1;

    return 0;
} // check whether a node n is in a finite element e

template <typename T, int NVERTICES>
void element<T, NVERTICES>::compute_area() {
    // Assuming T=point2d, NVERTICES=3
    point2d a = vertices_[0]->location();
    point2d b = vertices_[1]->location();
    point2d c = vertices_[2]->location();
    area_ = 0.5 * std::abs( (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]) );
}

template <typename T, int NVERTICES>
void element<T, NVERTICES>::compute_centroid() {
    point2d sum(0.0, 0.0);
    for (int i = 0; i < NVERTICES; ++i) {
        sum += vertices_[i]->location();
    }
    centroid_ = sum * (1.0 / NVERTICES);
}

template <typename T, int NVERTICES>
void element<T, NVERTICES>::compute_faces_length() {
    for (int i = 0; i < NVERTICES; ++i) {
        int j = (i + 1) % NVERTICES;
        point2d diff = vertices_[j]->location() - vertices_[i]->location();
        faces_length_[i] = std::sqrt(diff[0]*diff[0] + diff[1]*diff[1]);
    }
}

template <typename T, int NVERTICES>
void element<T, NVERTICES>::compute_outgoing_unit_normals() {
    for (int i = 0; i < NVERTICES; ++i) {
        int j = (i + 1) % NVERTICES;
        point2d vi = vertices_[i]->location();
        point2d vj = vertices_[j]->location();
        double dx = vj[0] - vi[0];
        double dy = vj[1] - vi[1];
        double len = faces_length_[i]; // Assume compute_faces_length called first
        if (len > 0) {
            // Outward normal assuming CCW ordering: (dy, -dx) / len
            normals_[i] = point2d(dy / len, -dx / len);
        }
    }
}

// ajout

// template<class T>
// int mesh<T>::compute_faces_outgoing_unit_normals()