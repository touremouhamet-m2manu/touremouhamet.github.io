#program once
#include"class_point.hpp"

template<typename POINT_T>
class node
{
public:
    node(T const& loc=0., int ind=−1, size_t sharing_elements=0, size_t sharing_faces=0)
    : location_(loc), index_(ind), sharing_elements_(sharing_elements), sharing_faces_(sharing_faces){};
    inline size_t sharing_elements() const {return sharing_elements_;};
    inline size_t& sharing_elements() {return sharing_elements_;};
    inline size_t sharing_faces() const {return sharing_faces_;};
    inline size_t& sharing_faces() {return sharing_faces_;};
    inline size_t index() const {return index_;};
    inline size_t& index() {return index_;};

    inline 

    node(node const& n):location_(n.location_), index_(n.index), sharing_elements_()
    node<POINT_T>& operator=(node const&);

    template<typename POINT_S>
    friend std::ostream& operator<<(std::ostream&)



protected:
    POINT_T location_;
    int index_;
    size_t sharing_elements_;
    size_t sharing_faces_;


};

template<typename POINT_T>
node<POINT_T>& node const&);