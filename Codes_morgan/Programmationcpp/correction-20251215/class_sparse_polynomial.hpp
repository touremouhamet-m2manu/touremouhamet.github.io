#include <cmath>
#include <iostream>

#include "class_linked_list_modified.hpp"
#include "class_monomial.hpp"

using namespace std;

class sparse_polynomial : public linked_list<monomial<double>>
{
public:
    //  default constructor
    sparse_polynomial() {};

    //  constructor with a T argument
    sparse_polynomial(size_t const& degree, double const& coef, linked_list<monomial<double>>* N = nullptr)
        : linked_list<monomial<double>>(monomial<double>(degree, coef), N) {};

    sparse_polynomial(const monomial<double>& monom, linked_list<monomial<double>>* N = 0)
        : linked_list<monomial<double>>(monom, N) {};

    ~sparse_polynomial() {} //  destructor

    size_t degree() { return this->item().power(); } //  degree of polynomial

    friend ostream& operator<<(ostream& inout, sparse_polynomial polyn);

    friend sparse_polynomial operator+(sparse_polynomial& polyn1, sparse_polynomial& polyn2);

    double operator()(double const& x);
};

double sparse_polynomial::operator()(double const& x)
{
    double val{0.};
    linked_list* scan = this; // polymorphism

    while(scan)
    {
         val += scan->item()(x); 
         scan = scan->p_next();
    };
    return val;
}

sparse_polynomial operator+(sparse_polynomial& polyn1, sparse_polynomial& polyn2)
{
    sparse_polynomial rr(polyn1);
    rr += polyn2;

    return rr;
}

ostream& operator<<(ostream& os, sparse_polynomial polyn)
{
    sparse_polynomial* curr = &polyn;

    os << endl;
    while (curr != nullptr)
    {
        if (curr->item().value() == 1.0)
        {
            os << " x^" << curr->item().power() << " + ";
            curr = (sparse_polynomial*)curr->p_next();
        }
        else
        {
            os << curr->item().value() << " x^" << curr->item().power() << " + ";
            curr = (sparse_polynomial*)curr->p_next();
        }
    }
    cout << '\b' << '\b' << " "; // REMOVE LAST + TO BE BEAUTIFUL
    cout << endl;
    return os;
}
