#pragma once

#include "class_binomial.hpp"
#include "class_list.hpp"
#include "derive_inverse.hpp"

template <typename T>
T derive_product(list<T> const& f, list<T> const& g, size_t order_n)
{
    assert(f.number() >= order_n && g.number() >= order_n);

    T result{0.};
    size_t nn = order_n;
    binomial<int> triangle(nn+1);
    cout << "triangle" << triangle << endl;

    for (auto kk = 0; kk <= nn; ++kk)
    {
        result += triangle(nn, kk)*f[kk]*g[nn-kk];
        cout << "triangle(nn, kk)= " << triangle(nn, kk) << endl;
        cout << "f[kk]= " << f[kk] << endl;
        cout << "g[nn-kk]= " << g[nn-kk] << endl;
    }

    return result;
}