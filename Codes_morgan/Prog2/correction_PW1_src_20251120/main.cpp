#include "class_list.hpp"
#include "derive_inverse.hpp"
#include "Taylor.hpp"
#include "class_binomial.hpp"
#include "derive_product.hpp"

#include <iostream>

int main()
{
    // list<double> l1(5, 3.); // cstr with 2 args

    // list<double> l2(l1); // cstr copy

    // // list<double> l3(3); // cstr with 1 arg

    // std::cout << "list l1= " << l1 << std::endl;
    // std::cout << "list l2= " << l2 << std::endl;

    // list<double> l3(2, 3.); // cstr with 2 args

    // std::cout << "list l3 before operator= " << l3 << std::endl;

    // l3 = l1; // operator=

    // std::cout << "list l3 after operator= " << l3 << std::endl;

    //
    double a            = 1.;
    size_t order_approx = 8;
    list<double> linv   = derive_inverse(a, order_approx);

    std::cout << "linv= " << linv << std::endl
              << std::endl;

    double h = 0.1;

    std::cout << "approximated f(x+h)= " << Taylor(linv, h) << std::endl;
    std::cout << "exact f(x+h)= " << 1. / (a + h) << std::endl;

    binomial<int> triangle(5);
    std::cout << "Pascal triangle for N=5" << std::endl;
    std::cout << triangle << endl;
    
    std::cout << "triangle(4,2)= " << triangle(4,2) << endl;

    cout << "(fxf)^(n)(1)= " << derive_product(linv, linv, 2) << endl;

    return 0;
}