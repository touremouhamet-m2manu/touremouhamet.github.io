#include "class_polynomials.hpp"
#include <iostream>

using namespace std;

int main()
{
    polynomial<double> p(3, 1.0);

    cout << "value of p(2) with Horner algo: " << horner_polynomial(p, 2.) << endl;

    return 0;
}
