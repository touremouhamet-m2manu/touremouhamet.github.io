#include "class_list.hpp"

template<class T>
const T Taylor (list<T> const& f , T const& h)
{
    T result{0.}; // init at 0 as populated by sum
    T h_on_i{1.}; // init at 1 as populated by product

    auto length = f.number();

    for(auto ii=0; ii<length; ++ii)
    {
        result += h_on_i*f[ii];
        h_on_i *= h/(T)(ii+1);
    }

    return result;
}