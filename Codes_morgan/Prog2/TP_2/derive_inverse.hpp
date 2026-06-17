#include "derive_inverse.hpp"


/*template<typename T>
const list<T> derive_inverse(T const& r, size_t n)
{
    list<T> l_inv(n+1, (T)1./r);

    for (auto ii=0; ii<<n; ++ii)
    {
        l_inv(ii) = -(T)ii/r*l_inv[ii-1];
    }

    return l_inv;
}*/
template<typename T>
const list<T> derive_inverse(T const& r, size_t n)
{
    // crée une liste de n+1 éléments initialisés à 1/r
    list<T> l_inv(n + 1, (T)(1.0 / r));

    // relation de récurrence : f^(k)(r) = -k/r * f^(k-1)(r)
    for (size_t ii = 1; ii <= n; ++ii)
    {
        l_inv(ii) = -(T)ii / r * l_inv[ii - 1];
    }

    return l_inv;
}

