#include "class_sparse_polynomial.hpp"

#include <iostream>

using namespace std;

int main()
{
    sparse_polynomial p0;
    sparse_polynomial rr(monomial<double>(15, 1.));
    
    cout << "1) le degré de rr est " << rr.degree() << endl;

    print(rr);
    
    rr.append(monomial<double>(2, 5.));
    cout << "2) le degré de rr est maintenant " << rr.degree() << endl;
    cout << rr << endl;
    

    monomial<double> mn(16, 100.);
    rr.insert_first_item(mn);

    cout << "3) le degré de rr est devenu " << rr.degree() << endl;
    //cout << rr << endl;
    print(rr);

    return 0;
    
}
