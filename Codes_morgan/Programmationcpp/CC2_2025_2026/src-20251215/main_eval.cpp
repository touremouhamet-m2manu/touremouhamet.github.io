#include "class_sparse_polynomial.hpp"
#include "class_monomial.hpp"

#include <iostream>

using namespace std;

int main()
{
    sparse_polynomial rr(monomial<double>(15, 1.));

    cout << "rr(1)= " << rr(1.) << endl;

    rr.append(monomial<double>(2, 5.));
    cout << "2) le degré de rr est maintenant " << rr.degree() << endl;


    cout << "rr(2)= " << rr(2.) << " (32788 is expected) " << endl;
    

    return 0;
}
