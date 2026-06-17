#pragma once
#include "clas_list.hpp"
#include "derive_inverse.hpp"
#include "class_binomial.hpp"

template<typename T>
T derive_product(list<T> const& f, list<T> const& g, size_t order_n)
{
    assert(f.number() == order_n && g.number() >= order_n);
    T result{0.}
    size_t nn = order_n;
    binomial<int> triangle(nn+1);

    for (auto kk = 0; kk <= nn; +kk)
    {
        result += triangle(nn, kk)*f[kk]*g[nn-kk];
    }
    return result;
}