#pragma once
#include <iostream>

template <class T>
class row_element
{
protected:
    T   value_;
    int column_;

public:
    // (3) constructeur avec valeurs par défaut
    row_element(T const& v = T(0), int c = -1)
        : value_(v), column_(c) {}

    // (4) constructeur de copie
    row_element(row_element const& other)
        : value_(other.value_), column_(other.column_) {}

    // (4) opérateur d’affectation
    const row_element& operator=(row_element const& other)
    {
        if (this != &other)
        {
            value_  = other.value_;
            column_ = other.column_;
        }
        return *this;
    }

    // (2) accesseurs
    T const& value() const   { return value_; }
    int      column() const  { return column_; }

    void set_value(T const& v) { value_ = v; }
    void set_column(int c)     { column_ = c; }

    // (5) opérateurs arithmétiques membres
    const row_element& operator+=(T const& t)
    {
        value_ += t;
        return *this;
    }

    const row_element& operator+=(row_element const& e)
    {
        value_ += e.value_;
        return *this;
    }

    const row_element& operator-=(T const& t)
    {
        value_ -= t;
        return *this;
    }

    const row_element& operator-=(row_element const& e)
    {
        value_ -= e.value_;
        return *this;
    }

    const row_element& operator*=(T const& t)
    {
        value_ *= t;
        return *this;
    }

    const row_element& operator/=(T const& t)
    {
        value_ /= t;
        return *this;
    }
};

// (6) opérateurs de comparaison (sur column_)
template <class T>
bool operator<(row_element<T> const& a, row_element<T> const& b)
{
    return a.column() < b.column();
}

template <class T>
bool operator>(row_element<T> const& a, row_element<T> const& b)
{
    return a.column() > b.column();
}

template <class T>
bool operator==(row_element<T> const& a, row_element<T> const& b)
{
    return a.column() == b.column();
}

// (7) opérateurs binaires avec un scalaire
template <class T>
const row_element<T> operator+(row_element<T> const& e, T const& t)
{
    row_element<T> res(e);
    res += t;
    return res;
}

template <class T>
const row_element<T> operator+(T const& t, row_element<T> const& e)
{
    row_element<T> res(e);
    res += t;
    return res;
}

template <class T>
const row_element<T> operator-(row_element<T> const& e, T const& t)
{
    row_element<T> res(e);
    res -= t;
    return res;
}

template <class T>
const row_element<T> operator-(T const& t, row_element<T> const& e)
{
    row_element<T> res(t - e.value(), e.column());
    return res;
}

template <class T>
const row_element<T> operator*(row_element<T> const& e, T const& t)
{
    row_element<T> res(e);
    res *= t;
    return res;
}

template <class T>
const row_element<T> operator*(T const& t, row_element<T> const& e)
{
    row_element<T> res(e);
    res *= t;
    return res;
}

template <class T>
const row_element<T> operator/(row_element<T> const& e, T const& t)
{
    row_element<T> res(e);
    res /= t;
    return res;
}

// (8) affichage
template <class T>
std::ostream& operator<<(std::ostream& os, row_element<T> const& e)
{
    os << "(" << e.column() << ", " << e.value() << ")";
    return os;
}
