// #ifndef ROW_ELEMENT1_HPP
// #define ROW_ELEMENT1_HPP

// #include<iostream>

// template <class T>
// class row_element {
// protected:
//     T   value_;   // valeur de l’élément
//     int column_;  // indice de colonne

// public:
//     // (3) Constructeur avec valeurs par défaut
//     row_element(T const& v = T(0), int c = -1)
//         : value_(v), column_(c) {}

//     // (4) Copy constructor
//     row_element(row_element<T> const& other)
//         : value_(other.value_), column_(other.column_) {}

//     // (4) operator=
//     row_element<T>& operator=(row_element<T> const& other) {
//         if (this != &other) {
//             value_  = other.value_;
//             column_ = other.column_;
//         }
//         return *this;
//     }

//     // (2) Accesseurs
//     T const& value() const { return value_; }
//     int      column() const { return column_; }

//     void set_value(T const& v) { value_ = v; }
//     void set_column(int c)     { column_ = c; }

//     // (5) opérateurs arithmétiques +=, -=, *=, /=
//     row_element<T>& operator+=(T const& t) {
//         value_ += t;
//         return *this;
//     }

//     row_element<T>& operator+=(row_element<T> const& e) {
//         value_ += e.value_;
//         return *this;
//     }

//     row_element<T>& operator-=(T const& t) {
//         value_ -= t;
//         return *this;
//     }

//     row_element<T>& operator-=(row_element<T> const& e) {
//         value_ -= e.value_;
//         return *this;
//     }

//     row_element<T>& operator*=(T const& t) {
//         value_ *= t;
//         return *this;
//     }

//     row_element<T>& operator/=(T const& t) {
//         value_ /= t;
//         return *this;
//     }
// };

// // (6) opérateurs de comparaison non-membres : sur column_
// template <class T>
// bool operator<(row_element<T> const& a, row_element<T> const& b) {
//     return a.column() < b.column();
// }

// template <class T>
// bool operator>(row_element<T> const& a, row_element<T> const& b) {
//     return a.column() > b.column();
// }

// template <class T>
// bool operator==(row_element<T> const& a, row_element<T> const& b) {
//     return a.column() == b.column();
// }

// // (7) opérateurs binaires avec un scalaire (non-membres)
// template <class T>
// row_element<T> const operator+(row_element<T> const& e, T const& t) {
//     row_element<T> res(e);
//     res += t;
//     return res;
// }

// template <class T>
// row_element<T> const operator+(T const& t, row_element<T> const& e) {
//     row_element<T> res(e);
//     res += t;
//     return res;
// }

// template <class T>
// row_element<T> const operator-(row_element<T> const& e, T const& t) {
//     row_element<T> res(e);
//     res -= t;
//     return res;
// }

// template <class T>
// row_element<T> const operator-(T const& t, row_element<T> const& e) {
//     // t - e.value_
//     row_element<T> res(t - e.value(), e.column());
//     return res;
// }

// template <class T>
// row_element<T> const operator*(row_element<T> const& e, T const& t) {
//     row_element<T> res(e);
//     res *= t;
//     return res;
// }

// template <class T>
// row_element<T> const operator*(T const& t, row_element<T> const& e) {
//     row_element<T> res(e);
//     res *= t;
//     return res;
// }

// template <class T>
// row_element<T> const operator/(row_element<T> const& e, T const& t) {
//     row_element<T> res(e);
//     res /= t;
//     return res;
// }

// // (8) affichage
// template <class T>
// std::ostream& operator<<(std::ostream& os, row_element<T> const& e) {
//     os << "(" << e.column() << ", " << e.value() << ")";
//     return os;
// }

// #endif // ROW_ELEMENT1_HPP

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
    row_element& operator=(row_element const& other)
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
    row_element& operator+=(T const& t)
    {
        value_ += t;
        return *this;
    }

    row_element& operator+=(row_element const& e)
    {
        value_ += e.value_;
        return *this;
    }

    row_element& operator-=(T const& t)
    {
        value_ -= t;
        return *this;
    }

    row_element& operator-=(row_element const& e)
    {
        value_ -= e.value_;
        return *this;
    }

    row_element& operator*=(T const& t)
    {
        value_ *= t;
        return *this;
    }

    row_element& operator/=(T const& t)
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
row_element<T> const operator+(row_element<T> const& e, T const& t)
{
    row_element<T> res(e);
    res += t;
    return res;
}

template <class T>
row_element<T> const operator+(T const& t, row_element<T> const& e)
{
    row_element<T> res(e);
    res += t;
    return res;
}

template <class T>
row_element<T> const operator-(row_element<T> const& e, T const& t)
{
    row_element<T> res(e);
    res -= t;
    return res;
}

template <class T>
row_element<T> const operator-(T const& t, row_element<T> const& e)
{
    row_element<T> res(t - e.value(), e.column());
    return res;
}

template <class T>
row_element<T> const operator*(row_element<T> const& e, T const& t)
{
    row_element<T> res(e);
    res *= t;
    return res;
}

template <class T>
row_element<T> const operator*(T const& t, row_element<T> const& e)
{
    row_element<T> res(e);
    res *= t;
    return res;
}

template <class T>
row_element<T> const operator/(row_element<T> const& e, T const& t)
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
