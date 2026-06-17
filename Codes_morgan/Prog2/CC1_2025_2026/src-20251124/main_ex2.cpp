#include "class_polynomials.hpp"
#include <iostream>

using namespace std;

int main()
{
    polynomial<double> p1(3, 1.0);
    polynomial<double> p2(4, 2.0);

    polynomial<double> add = p1 + p2;
    cout << "p1+p2= : " << endl;
    cout << add << endl;

    polynomial<double> s1 = 3.0 * p1;
    polynomial<double> s2 = p1 * 10.0;

    cout << "3 * p1= " << endl;
    cout << s1 << endl;

    cout << "p1 * 10= " << endl;
    cout << s2 << endl;

    return 0;
}
