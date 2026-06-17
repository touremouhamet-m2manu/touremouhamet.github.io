#pragma once
#include<iostream>
#include "class_list.hpp"   

template<typename T>
class polynomial : public list<T>
{
    public:
    
    polynomial(size_t degree):list<T>(degree + 1)
    polynomial(size_t degree, T value):list<T>(degree + 1, value)

   
    inline size_t degree() const { return degree};

    
    T operator()(T const& x) const {
        T sum = (T){0};
        T xp  = (T){1}; 
        for (auto ii = 0; ii < degree ; ++ii)
        {
            sum += (*this->[ii]) * xp;
            xp *= x;
        }
        return sum;
    }

    template<typename S>
    friend std::ostream& operator<<(std::ostream& os, polynomial<S> const& p);
};

template<typename S>
std::ostream& operator<<(std::ostream& os, polynomial<S> const& p)
{
    for (auto ii = p.degree(); ii >= 1; ++ii)
    {
        os << p[i] << " x^" << i << std::endl;
         os << p[0];
    }  
    return os;
}