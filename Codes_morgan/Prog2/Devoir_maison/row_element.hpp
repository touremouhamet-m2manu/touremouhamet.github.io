#pragma once
#include<iostream>

template<class T>
class row_element
{
private:
    T value_;
    int column_;
public:

    inline T value() const;
    inline T& value();
    inline int column() const;
    inline int& column();
    void set_value(T const& val);
    void set_column(int col);

    row_element();
    row_element(T const& val, int col = -1);
    row_element(row_element const& r);
    ~row_element();

    const row_element& operator=(row_element const& r);
    const row_element& operator+=(T const& t);
    const row_element& operator+=(row_element const& e);
    const row_element& operator-=(T const& t);
    const row_element& operator-=(row_element const& e);
    const row_element& operator*=(T const& t);
    const row_element& operator/=(T const& t);
};

template<class T>
inline T row_element<T>::value() const
{
    return value_;
}

template<class T>
inline T& row_element<T>::value()
{
    return value_;
}

template<class T>
inline int row_element<T>::column() const
{
    return column_;
}

template<class T>
inline int& row_element<T>::column()
{
    return column_;
}

template<class T>
void row_element<T>::set_value(T const& val)
{
    value_ = val;
}

template<class T>
void row_element<T>::set_column(int col)
{
    column_ = col;
}

template<class T>
row_element<T>::row_element() : value_(T(0)), column_(-1) {}

template<class T>
row_element<T>::row_element(T const& val, int col) : value_(val), column_(col) {}

template<class T>
row_element<T>::row_element(row_element const& r) : value_(r.value_), column_(r.column_) {}

template<class T>
row_element<T>::~row_element() {}

template<class T>
const row_element<T>& row_element<T>::operator=(row_element const& r)
{
    if (this != &r)
    {
        value_ = r.value_;
        column_ = r.column_;
    }
    return *this;
}

template<class T>
const row_element<T>& row_element<T>::operator+=(T const& t)
{
    value_ += t;
    return *this;
}

template<class T>
const row_element<T>& row_element<T>::operator+=(row_element const& e)
{
    value_ += e.value_;
    return *this;
}

template<class T>
const row_element<T>& row_element<T>::operator-=(T const& t)
{
    value_ -= t;
    return *this;
}

template<class T>
const row_element<T>& row_element<T>::operator-=(row_element const& e)
{
    value_ -= e.value_;
    return *this;
}

template<class T>
const row_element<T>& row_element<T>::operator*=(T const& t)
{
    value_ *= t;
    return *this;
}

template<class T>
const row_element<T>& row_element<T>::operator/=(T const& t)
{
    value_ /= t;
    return *this;
}

template<class T>
bool operator<(row_element<T> const& a, row_element<T> const& b)
{
    return a.column() < b.column();
}

template<class T>
bool operator>(row_element<T> const& a, row_element<T> const& b)
{
    return a.column() > b.column();
}

template<class T>
bool operator==(row_element<T> const& a, row_element<T> const& b)
{
    return a.column() == b.column();
}

template<class T>
const row_element<T> operator+(row_element<T> const& e, T const& t )
{
    row_element<T> res(e);
    res += t;
    return res;
}

template<class T>
const row_element<T> operator+(T const& t , row_element<T> const& e)
{
    row_element<T> res(e);
    res += t;
    return res;
}

template<class T>
const row_element<T> operator-(row_element<T> const& e, T const& t)
{
    row_element<T> res(e);
    res -= t;
    return res;
}

template<class T>
const row_element<T> operator-(T const& t , row_element<T> const& e)
{
    row_element<T> res(t - e.value(), e.column());
    return res;
}

template<class T>
const row_element<T> operator * (row_element<T> const& e, T const& t )
{
    row_element<T> res(e);
    res *= t;
    return res;
}

template<class T>
const row_element<T> operator * (T const& t , row_element<T> const& e)
{
    row_element<T> res(e);
    res *= t;
    return res;
}

template<class T>
const row_element<T> operator/(row_element<T> const& e, T const& t )
{
    row_element<T> res(e);
    res *= t;
    return res;
}

template<class T>
std::ostream& operator<<(std::ostream& os, row_element<T> const& e)
{
    os << "(" << e.column() << ", " << e.value() << ")";
    return os;
}