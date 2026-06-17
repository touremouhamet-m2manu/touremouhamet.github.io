#pragma once

#include"class_node.hpp"

template<class T, int NVERTICES>
class element
{
    protected:
    node<T>* vertices_[NVERTICES];

    public:
     element(node<T>& v1, node<T>& v2, node<T>& v3);
     element(element&);
     element& operator=(element&);
     ~element()

     const node<T>& operator[](size_t i) const {return *vertices_[i];};
     node<T>& operator()(size_t i){return *vertices_[i];};
     node<T>* vertex(size_t i){return *vertices_[i];};

     voide reset_indices();
     void indexing(int& count);
    //  int operator<(node<T> const& n, element<T,NVERTICES> const& e);

    
};

typedef element<vertex, 3> triangle;

// template<typename T, int NVERTICES>
// void element<T, NVERTICES>::print_vertices_

template<class T, int NVERTICES>
int element<T, NVERTICES>::operator<(node<T> const& n, element<T,NVERTICES> const& e);
{
    // int count{0};
    for(auto ii=0; ii<NVERTICES; ii++)
    {
        if(e.vertex(ii) == &n)
           return ii+1;
    }
    return 0;

}


template<class T, int NVERTICES>
void element<T, NVERTICES>::reset_indices()
{
    for(auto i=0; i<NVERTICES; i++) vertices_[i]->index() = -1;
}

template<class T, int NVERTICES>
void element<T, NVERTICES>::indexing(int& count)
{
for (auto i =0; i<NVERTICES; i++)
{
    i f (vertices[i]−>index()<0)
    {
        vertices[i]−> index() == count++;
    }
}
}

template<class T, int NVERTICES>
element<T, NVERTICES>::element(node<T>& v1, node<T>& v2, node<T>& v3)
{
    this->vertices_[0] = &v1;
    this->vertices_[1] = &v1;
    this->vertices_[2] = &v1;

    for (auto i=0; i<NVERTICES; i++)
    {
        this->vertices_[i]->more_sharing_elements();
    }
}

template<class T, int NVERTICES>
element<T, NVERTICES>::~element()
{

    for (auto i=0; i<NVERTICES; i++)
    {
        this->vertices_[i]->less_sharing_elements();
    }
}

template<class T, int NVERTICES>
element<T, NVERTICES>::element(element<T, NVERTICES>& elem_r)
{
    for (auto i=0; i<NVERTICES; i++)
    {
        this->vertices_[i]=elem_r.vertices_[i];
        this->vertices_[i]->more_sharing_elements();

    }
}


template<class T, int NVERTICES>
element<T, NVERTICES>& element<T, NVERTICES>::operator=(element& el)
{
    if(this != &el)
    {

    for (auto i=0; i<NVERTICES; i++)
    {
        this->vertices_[i]->less_sharing_elements();

    }

    for (auto i=0; i<NVERTICES; i++)
    {
        this->vertices_[i]=elem_r.vertices_[i];
        this->vertices_[i]->more_sharing_elements();

    }
    
    }
    return *this;
}

