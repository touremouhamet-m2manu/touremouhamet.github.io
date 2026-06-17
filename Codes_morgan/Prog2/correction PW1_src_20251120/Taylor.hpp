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







// // derivé invers et developpement de taylor

// #include"once";
// #include<iostream>



// template<typename T>
// const l i s t <T> derive_inverse (T const& a, int n );
// {
//     list<T> l_inv(n+1, (T)1./r);
//     for (auto ii = 1; ii<n; ++ii)
//     {
//         l_inv(ii)=-(T)ii/r*l_inv[ii-1];
//     }
//     return l_inv;

// }




// template<class T>
// const T Taylor ( l i s t <T> const& f , T const& h )
// {
//     T result{0.};
//     T h_on_i{1.};

//     auto length = f.number();
//     for (auto ii=0; ii<length; ++ii)
//     {
//         result += h_on_i*f[ii];
//         h_on_i *= h/(T)(ii+1);
//     }
//     return result;
// }