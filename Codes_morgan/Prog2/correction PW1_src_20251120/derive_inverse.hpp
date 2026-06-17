#pragma once
#include "class_list.hpp"

template<typename T>
const list<T> derive_inverse(T const& r, size_t n)
{
    list<T> l_inv(n+1, (T)1./r);

    for(auto ii=1; ii<=n; ++ii)
    {
        l_inv(ii) = -(T)ii/r*l_inv[ii-1];
    }
    return l_inv;
}

