#pragma once

#include <cassert>
#include <iostream>

template <typename T>
class dynamic_vector
{
protected:
    size_t size_;
    T* coord_;

public:
    dynamic_vector(size_t N, T xx = 0.0);
    dynamic_vector(size_t N, T xx[]);
    const T& operator[](size_t ii) const { return this->coord_[ii]; }; // read only
    T& operator()(size_t ii) { return this->coord_[ii]; };             // read-write

    void zero();
    size_t size() const { return size_; }
    ~dynamic_vector() { delete[] coord_; };                  //  destructor
    dynamic_vector(dynamic_vector const& p);                  //  copy constructor
    const dynamic_vector& operator=(dynamic_vector const& p); //  dynamic_vector-to-dynamic_vector assignment
    
    template <typename S>
    friend const dynamic_vector<S> operator*(dynamic_vector<S> const& p, S const& lambda); //  dynamic_vector times T
    template <typename S>
    friend const dynamic_vector<S> operator*(S const& lambda, dynamic_vector<S> const& p); //  T times dynamic_vector
    const dynamic_vector& operator=(T const& x);                                          //  T-to-dynamic_vector assignment
    template <typename S>
    friend const dynamic_vector<S> operator-(dynamic_vector<S> const& p); // unary operator-
    template <typename S>
    friend const dynamic_vector<S> operator+(dynamic_vector<S> const& p); // unary operator+
    const dynamic_vector& operator*=(T const& lambda);                   // multiplying the current dynamic_vector by a scalar
    const dynamic_vector& operator+=(dynamic_vector const& p);            // adding a dynamic_vector to the current dynamic_vector
    const dynamic_vector& operator-=(dynamic_vector const& p);            // subtracting a dynamic_vector to the current dynamic_vector

    template <typename S>
    friend const dynamic_vector<S> operator+(dynamic_vector<S> const& p, dynamic_vector<S> const& q);
    template <typename S>
    friend const dynamic_vector<S> operator-(dynamic_vector<S> const& p, dynamic_vector<S> const& q);
    template <typename S>
    friend std::ostream& operator<<(std::ostream& os, dynamic_vector<S> const& p);
    template <typename S>
    friend S operator*(dynamic_vector<S> const& u, dynamic_vector const& v); //  inner product
};

//========================================
// definitions

using namespace std;

template <class T>
T operator*(const dynamic_vector<T>& u, const dynamic_vector<T>& v)
{
    T sum = 0;
    for (auto i = 0; i < u.size(); i++)
        sum += u[i] * +v[i];

    return sum;
} //  inner product

template <typename T>
dynamic_vector<T>::dynamic_vector(size_t N, T xx[]) : size_(N), coord_(N ? new T[N] : nullptr)
{
    for (auto ii = 0; ii < this->size_; ii++)
    {
        this->coord_[ii] = xx[ii];
    }
};

template <typename T>
dynamic_vector<T>::dynamic_vector(size_t N, T xx) : size_(N), coord_(N ? new T[N] : nullptr)
{
    for (auto ii = 0; ii < this->size_; ii++)
    {
        this->coord_[ii] = xx;
    }
};


template <typename T>
void dynamic_vector<T>::zero()
{
    for (auto ii = 0; ii < this->size_; ii++)
    {
        this->coord_[ii] = 0.;
    }
};

template <typename T>
const dynamic_vector<T> operator*(const dynamic_vector<T>& p, const T& lambda)
{ //  dynamic_vector-to-dynamic_vector assignment
    T product[p.size_];
    for (auto ii = 0; ii < p.size_; ii++)
    {
        product[ii] = lambda * p.coord_[ii];
    }

    return dynamic_vector<T>(p.size_, product);
}

template <typename T>
const dynamic_vector<T> operator*(const T& lambda, const dynamic_vector<T>& p)
{ //  dynamic_vector-to-dynamic_vector assignment
    T product[p.size_];
    for (auto ii = 0; ii < p.size_; ii++)
    {
        product[ii] = lambda * p.coord_[ii];
    }
    return dynamic_vector<T>(p.size_, product);
}

template <typename T>
const dynamic_vector<T>& dynamic_vector<T>::operator*=(const T& lambda)
{ // multiplying the current dynamic_vector by a scalar
    for (auto i = 0; i < this->size_; i++)
    {
        this->coord_[i] *= lambda;
    }
    return *this;
}

template <typename T>
dynamic_vector<T>::dynamic_vector(const dynamic_vector<T>& p) : size_(p.size_), coord_(p.size_ ? new T[p.size_] : 0)
{

    for (auto ii = 0; ii < this->size_; ii++)
    {
        this->coord_[ii] = p.coord_[ii];
    }
}; //  copy constructor

template <typename T>
const dynamic_vector<T>& dynamic_vector<T>::operator=(const dynamic_vector<T>& p)
{
    if (this != &p)
    {
        if (this->size_ > p.size_ || this->size_ < p.size_)
        {
            delete[] this->coord_;
            coord_ = new T[p.size_];
            this->size_ = p.size_;

        }
        for (auto ii = 0; ii < this->size_; ii++)
        {
            this->coord_[ii] = p.coord_[ii];
        }
    }
    return *this;
} //  dynamic_vector-to-dynamic_vector assignment

template <typename T>
const dynamic_vector<T>& dynamic_vector<T>::operator=(const T& x)
{
    for (auto ii = 0; ii < this->size_; ii++)
    {
        this->coord_[ii] = x;
    }
    return *this;
}

template <typename T>
const dynamic_vector<T> operator-(const dynamic_vector<T>& p)
{
    return -1.0 * p;
} // unary -

template <typename T>
const dynamic_vector<T> operator+(const dynamic_vector<T>& p)
{
    return p;
} // unary +
//

template <typename T>
const dynamic_vector<T>& dynamic_vector<T>::operator+=(const dynamic_vector<T>& p)
{
    assert(p.size_ == this->size_);
    for (auto ii = 0; ii < this->size_; ii++)
    {
        this->coord_[ii] += p[ii];
    }
    return *this;
} //  adding a dynamic_vector to the current dynamic_vector

template <typename T>
const dynamic_vector<T>& dynamic_vector<T>::operator-=(const dynamic_vector<T>& p)
{
    assert(p.size_ == this->size_);
    for (auto ii = 0; ii < this->size_; ii++)
    {
        this->coord_[ii] -= p[ii];
    }
    return *this;
} //  subtracting a dynamic_vector to the current dynamic_vector

template <typename T>
const dynamic_vector<T> operator+(const dynamic_vector<T>& p, const dynamic_vector<T>& q)
{
    assert(p.size_ == q.size_);
    dynamic_vector<T> s(p.size_);
    for (auto ii = 0; ii < s.size_; ii++)
    {
        s.coord_[ii] = p[ii] + q[ii];
    }
    return s;
} //  add two dynamic_vectors

template <typename T>
const dynamic_vector<T> operator-(const dynamic_vector<T>& p, const dynamic_vector<T>& q)
{
    assert(p.size_ == q.size_);
    dynamic_vector<T> s(p.size_);
    for (auto ii = 0; ii < s.size_; ii++)
    {
        s.coord_[ii] = p[ii] - q[ii];
    }
    return s;
} //  subtract two dynamic_vectors

template <typename T>
std::ostream& operator<<(std::ostream& os, const dynamic_vector<T>& p)
{
    os << "(";
    for (auto ii = 0; ii < p.size_ - 1; ii++)
    {
        os << p[ii] << ",";
    }
    os << p[p.size_ - 1] << ")";
    os << std::endl << std::endl;

    return os;
}