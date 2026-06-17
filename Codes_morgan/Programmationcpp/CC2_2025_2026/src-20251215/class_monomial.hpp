#pragma once
#include <iostream>

using namespace std;

// minimal working monomial class storing a_n x^n
template <typename T>
class monomial
{
protected:
    size_t power_;
    T value_;

public:
    monomial():power_(0.), value_(0.) {};
    monomial(size_t nn, T aa) : power_(nn), value_(aa) {};
    size_t power() const { return power_; };
    T value() const { return value_; };
    size_t& power() { return power_; };
    T& value() { return value_; };
    ~monomial() {};

    template <typename S>
    friend ostream& operator<<(ostream& os, const monomial<S>& mn);

    // comparison operators for merging algorithm
    template <typename S>
    friend bool operator==(const monomial<S>& m1, const monomial<S>& m2);

    template <typename S>
    friend bool operator<(const monomial<S>& m1, const monomial<S>& m2);

    const monomial operator+=(monomial& mn)
    {
        //if (*this == mn) // check the degree
            value() += mn.value();
        return *this;
    };

    T operator()(T const& x){return value()*std::pow(x, power());};
};


// output of monomials
template <typename S>
ostream& operator<<(ostream& os, const monomial<S>& mn)
{
    os << mn.value() << " x^" << mn.power() << endl
       << endl;
    return os;
};

template <typename S>
bool operator== (const monomial<S>& m1, const monomial<S>& m2)
{
  return m1.power() == m2.power();
}

template <typename S>
bool operator< (const monomial<S>& m1, const monomial<S>& m2)
{
  return m1.power() > m2.power();
}
