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

    // geometry (element dependent)
    double area_; //
    point2d centroid_;
    std::array<point2d, NVERTICES> normals_;
    std::array<double, NVERTICES> faces_length_;

    // element<T, NVERTICES>* neighbors_[NVERTICES];
    std::array<int, NVERTICES> neighbors_;           // indices globaux parmis les {triangles} des adjacents (-1 si bord)
    std::array<int, NVERTICES> faces_;               // indices globaux parmis les {edges} associés aux 3 faces  (-1 si bord)
    std::array<int, NVERTICES> normals_orientation_; // -1/+1 en comparaison avec {unit_normals} associés à {edges}

public:
    element();                             //  default constructor
    element(node<T>&, node<T>&, node<T>&); // for triangles
    element(element<T, NVERTICES> const&);
    // const is needed in copy-cst arg for const-compatibility
    // with the copy-cstr of linked_list object

    node<T>& operator()(size_t i) { return *(this->vertices_[i]); }             //  read/write ith vertex
    const node<T>& operator[](size_t i) const { return *(this->vertices_[i]); } //  read only ith vertex

    node<T>* vertex(std::size_t i) const { return this->vertices_[i]; }; //  get ith vertex address (pointer)
    node<T>*& vertex(std::size_t i) { return this->vertices_[i]; };      //  get ith vertex address (pointer)

    inline int index() const { return index_; };
    inline void set_index(int ii) { index_ = ii; };
    // inline int& index() { return index_; };

    inline int neighbor(std::size_t i) const { return this->neighbors_[i]; }; //
    // inline int& neighbor(std::size_t i) { return this->neighbors_[i]; };      //
    inline void set_neighbor_index(std::size_t ii, int jj) { this->neighbors_[ii] = jj; }; //

    inline int normal_orientation(std::size_t i) const { return this->normals_orientation_[i]; };        //
    inline void set_normal_orientation(std::size_t ii, int jj) { this->normals_orientation_[ii] = jj; }; //

    inline int face(std::size_t i) const { return this->faces_[i]; }; //  get ith face index
    inline int& face(std::size_t i) { return this->faces_[i]; };      //  get ith face index

    inline double faces_length(std::size_t i) const { return this->faces_length_[i]; }; //  get ith face index
    // inline double& faces_length(std::size_t i) { return this->faces_length_[i]; };      //  get ith face index

    inline double area() const { return area_; };
    // inline double& area() { return area_; };

    inline point2d centroid() const { return centroid_; };
    // inline point2d& centroid() { return centroid_; };

    inline point2d normal(size_t ii) const { return normals_[ii]; };

    const element<T, NVERTICES>& operator=(element<T, NVERTICES>&);
    ~element();

    void reset_indices();               //  reset vertices indices to -1
    void indexing_vertices(int& count); //  indexing the vertices

    template <typename S, int MVERTICES>
    friend std::ostream& operator<<(std::ostream& os, element<S, MVERTICES> const& e);
    void print_vertices_indices();

    void compute_area();
    void compute_centroid();
    void compute_faces_length();
    void compute_unit_outgoing_normals();
    // void compute_faces_orientations();
};

// element for d=2
typedef element<point2d, 3> triangle;

template <typename T, int NVERTICES>
void element<T, NVERTICES>::compute_unit_outgoing_normals()
{
    for (auto i = 0; i < 3; ++i)
    {
        // scan vers les sommets de l'arête : (0,1), (1,2), puis (2,0)
        node<T>* pA = vertices_[i];
        node<T>* pB = vertices_[(i + 1) % 3];

        // 1. Vecteur directeur de l'arête
        double dx = pB->x() - pA->x();
        double dy = pB->y() - pA->y();

        // 2. Normale sortante (orthogonale à l'arête)
        // Rotation de 90° vers la droite : (dy, -dx)
        double nx = dy;
        double ny = -dx;

        // 3. Calcul de la longueur (mesure de la face)
        double length    = std::sqrt(nx * nx + ny * ny);
        faces_length_[i] = length;

        // 4. Normalisation unitaire
        if (length > 1e-15)
        {
            double invL = 1.0 / length;
            normals_[i] = point2d(nx * invL, ny * invL);
        }
    }
}

template <typename T, int NVERTICES>
void element<T, NVERTICES>::compute_faces_length()
{
    double dx1       = vertices_[1]->x() - vertices_[0]->x();
    double dy1       = vertices_[1]->y() - vertices_[0]->y();
    double dx2       = vertices_[2]->x() - vertices_[1]->x();
    double dy2       = vertices_[2]->y() - vertices_[1]->y();
    double dx3       = vertices_[0]->x() - vertices_[2]->x();
    double dy3       = vertices_[0]->y() - vertices_[2]->y();

    faces_length_[0] = std::sqrt(dx1 * dx1 + dy1 * dy1);
    faces_length_[1] = std::sqrt(dx2 * dx2 + dy2 * dy2);
    faces_length_[2] = std::sqrt(dx3 * dx3 + dy3 * dy3);
}

template <typename T, int NVERTICES>
void element<T, NVERTICES>::compute_area()
{
    double x1 = vertices_[0]->x();
    double x2 = vertices_[1]->x();
    double x3 = vertices_[2]->x();
    double y1 = vertices_[0]->y();
    double y2 = vertices_[1]->y();
    double y3 = vertices_[2]->y();

    area_     = 0.5 * std::abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
}

template <typename T, int NVERTICES>
void element<T, NVERTICES>::compute_centroid()
{
    double x1 = vertices_[0]->x();
    double x2 = vertices_[1]->x();
    double x3 = vertices_[2]->x();
    double y1 = vertices_[0]->y();
    double y2 = vertices_[1]->y();
    double y3 = vertices_[2]->y();

    double xc = (1. / 3.) * (x1 + x2 + x3);
    double yc = (1. / 3.) * (y1 + y2 + y3);

    centroid_ = point2d(xc, yc);
}

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
element<T, NVERTICES>::element(node<T>& a, node<T>& b, node<T>& c)
{
    this->vertices_[0] = &a;
    this->vertices_[1] = &b;
    this->vertices_[2] = &c;

    for (auto i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i]->more_sharing_elements();
    }
} //  constructor

template <typename T, int NVERTICES>
element<T, NVERTICES>::element()
{
    this->vertices_[0] = nullptr;
    this->vertices_[1] = nullptr;
    this->vertices_[2] = nullptr;
} //  constructor

template <typename T, int NVERTICES>
element<T, NVERTICES>::element(element<T, NVERTICES> const& e)
{
    for (int i = 0; i < NVERTICES; i++)
    {
        this->vertices_[i] = e.vertices_[i];
        this->vertices_[i]->more_sharing_elements();
    }
} //  copy constructor

template <typename T, int NVERTICES>
const element<T, NVERTICES>& element<T, NVERTICES>::operator=(element<T, NVERTICES>& e)
{
    if (this != &e)
    {
        for (int i = 0; i < NVERTICES; i++)
            this->vertices_[i]->less_sharing_elements();
        // delete this->vertices_[i];

        for (int i = 0; i < NVERTICES; i++)
        {
            this->vertices_[i] = e.vertices_[i];
            this->vertices_[i]->more_sharing_elements();
        }
    }
    return *this;
} //  assignment operator

template <typename T, int NVERTICES>
element<T, NVERTICES>::~element()
{
    for (int i = 0; i < NVERTICES; i++)
        this->vertices_[i]->less_sharing_elements();
} //   destructor

template <typename T, int NVERTICES>
void element<T, NVERTICES>::reset_indices()
{
    for (int i = 0; i < NVERTICES; i++)
        this->vertices_[i]->index() = -1;
} //  reset indices to -1

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
} //  indexing the vertices

template <typename T, int NVERTICES>
int operator<(const node<T>& n, const element<T, NVERTICES>& e)
{
    for (int i = 0; i < NVERTICES; i++)
        if (&n == &(e[i]))
            return i + 1;

    return 0;
} //  check whether a node n is in a finite element e
