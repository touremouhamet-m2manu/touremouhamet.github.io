#include "class_polynomials.hpp"
#include <iostream>

using namespace std;

int main()
{
    polynomial<double> a(2);
    polynomial<double> p(3, 1.0);
    
    cout << "polynomial p: " << endl;
    cout << p << endl;

    cout << "value of p(2): " << p(2.) << endl;
    cout << endl;

    return 0;
}
