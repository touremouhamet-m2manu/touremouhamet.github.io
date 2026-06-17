#include "derive_inverse.hpp"

/*template<class T>
const T Taylor (list<T> const& f , T const& h);
{
    T result{0.}; //initialise la somme a 0
    T h_on_i{1.}; //initialise le produit a 1

    auto length = f.number();
    for (auto ii = 0; ii < length; i++)
    {
        result += h_on_i*f[ii];
        h_on_i *= h/(T)(ii+1)
    }
    return result;
    

}*/
template<class T>
const T Taylor(list<T> const& f, T const& h)
{
    T result{0.}; // somme finale
    T h_on_i{1.}; // terme courant h^i / i!

    auto length = f.number();

    for (size_t ii = 0; ii < length; ++ii)
    {
        result += h_on_i * f[ii];
        h_on_i *= h / (T)(ii + 1); // mise à jour du rapport h^(i+1)/(i+1)!
    }
    return result;
}

