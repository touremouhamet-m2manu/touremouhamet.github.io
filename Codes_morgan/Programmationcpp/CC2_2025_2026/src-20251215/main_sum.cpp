#include "class_sparse_polynomial.hpp"
#include "class_monomial.hpp"

#include <iostream>

using namespace std;

int main()
{
    sparse_polynomial rr(monomial<double>(15, 1.));
    
    sparse_polynomial p2(monomial<double>(3, 1.));
    p2.append(monomial<double>(2, 5.));
    cout << "4) p2= " << endl;
    cout << p2 << endl;

    rr += p2;
    cout << "4) rr += p2 " << endl;
    cout << rr << endl;

    sparse_polynomial p4(monomial<double>(4, 9.));
    sparse_polynomial p6(monomial<double>(1, 10.));
    rr = p4 + p6;
    cout << "6) rr = p4 + p6 " << endl;
    cout << rr << endl;

    return 0;
}
