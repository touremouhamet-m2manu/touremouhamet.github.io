#pragma once

#include<iostream>
#include"class_linked_list.hpp"
#include"class_monomial.hpp"




template<class T>
class sparse_polynomial : public linked_list<monomial<double>>

{

public:

sparse_polynomial() : linked_list<monomial<double>>() {};

sparse_polynomial(monomial<double> const& monom,
                  linked_list<monomial<double>>* pt = nullptr);

~sparse_polynomial(){};


size_t degree() const
{
    return this->item().power();
}


spare_polynomial& operator +=(sparse_polynomial<double>& P)
{
    linked_list<monomial<double>>::operator+=(P)
    return *this;
}

sparse_polynomial operator+(sparse_polynomial A, sparse_polynomial& B)
{
    A +=B;
    return A;
}


double operator() (double x) const
{
    double result = 0.0;
    const linked_list<monomial<double>>* p = this;

    while(p)
    {
        result += p->item().coefficient()
                  *std::pow(x, p->item().pow());
        p = p->p_next();
    }
    return result;
}


  friend sparse_polynomial operator+(sparse_polynomial A,  sparse_polynomial const& B)
    {         
        A += B;
        return A;
    }

 friend std::ostream& operator<<(std::ostream& os, sparse_polynomial const& P)
      {
         const linked_list<monomial<double>>* p = &P;
        while (p)
        {
             os << p->item() << std::endl;
             p = p->p_next();
        }
         return os;
     }

};