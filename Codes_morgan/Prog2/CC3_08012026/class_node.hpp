#pragma once
#include "class_point.hpp"
#include <iostream>

template <typename POINT_T>
class node
{
public:
    node(POINT_T const& loc = 0., int ind = -1, size_t sharing_elements = 0, size_t sharing_faces = 0)
    : location_(loc), index_(ind), sharing_elements_(sharing_elements), sharing_faces_(sharing_faces){};

    inline size_t sharing_elements() const { return sharing_elements_; };
    inline size_t& sharing_elements() { return sharing_elements_; };
    inline size_t sharing_faces() const { return sharing_faces_; };
    inline size_t& sharing_faces() { return sharing_faces_; };
    inline int index() const { return index_; };
    inline int& index() { return index_; };
    inline POINT_T location() const { return location_; };
    inline double x() const {return location_.x();}
    inline double y() const {return location_.y();}
    inline double& x() {return location_.x();}
    inline double& y() {return location_.y();}

    inline void more_sharing_elements(){++sharing_elements_;};
    inline void less_sharing_elements(){--sharing_elements_;};
    inline void more_sharing_faces(){++sharing_faces_;};
    inline void less_sharing_faces(){--sharing_faces_;};

    node(node const& n) 
    : location_(n.location_), index_(n.index_), sharing_elements_(n.sharing_elements_), sharing_faces_(n.sharing_faces_){};
    node<POINT_T>& operator=(node const&);

    template <typename POINT_S>
    friend std::ostream& operator<<(std::ostream&, node<POINT_S> const&);

    void print() const {std::cout << "node index= " << index() << std::endl;};

protected:
    POINT_T location_;
    int index_;
    size_t sharing_elements_;
    size_t sharing_faces_;
};

// a shorcut for usual 2d nodes
typedef node<point2d> vertex;

template <typename POINT_S>
std::ostream& operator<<(std::ostream& os, node<POINT_S> const& nn)
{
        os << "(x,y)= ("<< nn.location() << ")" << std::endl;;
        return os;
}


template <typename POINT_T>
node<POINT_T>& node<POINT_T>::operator=(node const& n)
{
    if (this != &n)
    {
        location_         = n.location_;
        index_            = n.index_;
        sharing_elements_ = n.sharing_elements_;
        sharing_faces_    = n.sharing_faces_;
    }
    return *this;
}
