#pragma once

#include "row_element.hpp"
#include "class_dynamic_vector.hpp"
#include "class_linked_list.hpp"

template<class T>
class row : public linked_list<row_element<T>>
{

public:
    row(T const& val = T(0), int col = -1); 
    row_element<T> const& operator()() const;

    T const& value() const;
    int column() const;

    void insert_next_item(T const& val, int col);
    void insert_first_item(T const& val, int col);
    void append(T const& val, int col);
    
    T const row_sum() const;
    T operator[](int i) const;
    T operator*(dynamic_vector<T> const& v) const;

    const row<T>& operator*=(T const& t);
    const row<T>& operator/=(T const& t);
    

    void renumber_columns(dynamic_vector<int> const& renumber);
    void drop_items(dynamic_vector<int> const mask);
    
};

template<class T>
row<T>::row(T const& val, int col) : linked_list<row_element<T>>() 
{
    if(col != -1)
    {
        row_element<T> e(val, col);
        this->item() = e;
    }
}

template<class T>
row_element<T> const& row<T>::operator()() const
{
    return this->item();
}

template<class T>
T const& row<T>::value() const
{
    return this->item().value();
}

template<class T>
int row<T>::column() const
{
    return this->item().column();
}

template<class T>
void row<T>::insert_next_item(T const& val, int col)
{
    row_element<T> e(val, col);
    linked_list<row_element<T>>::insert_next_item(e);
}

template<class T>
void row<T>::insert_first_item(T const& val, int col)
{
    row_element<T> e(val, col);
    linked_list<row_element<T>>::insert_first_item(e);
}

template<class T>
void row<T>::append(T const& val, int col)
{
    row_element<T> e(val, col);
    linked_list<row_element<T>>::append(e);
}

template<class T>
T const row<T>::row_sum() const
{
    T sum = value();
    if (this->p_next())
    {
        sum += ((row<T>*)this->p_next())->row_sum();
    }
    return sum;
}

template<class T>
T row<T>::operator[](int i) const
{
    int c = column();
    if (c == i) return value();
    if (c > 1) return T(0);
    if (!this->p_next()) return T(0);
    return (*(row<T>*)this->p_next())[i];
}

template<class T>
const row<T>& row<T>::operator*=(T const& t)
{
    this->item().set_value(this->item().value() * t);
    if (this->p_next())
        (*(row<T>*)this->p_next()) *= t;
    return *this;
}

template<class T>
const row<T>& row<T>::operator/=(T const& t)
{
    this->item().set_value(this->item().value() / t);
    if (this->p_next())
        (*(row<T>*)this->p_next()) /= t;
    return *this;
}

template<class T>
T row<T>::operator*(dynamic_vector<T> const& v) const
{
    T res = value() * v[(size_t)column()];
    if (this->p_next())
        res += (*(row<T>*)this->p_next()) * v;
    return res;
}

template<class T>
void row<T>::renumber_columns(dynamic_vector<int> const& renumber)
{
    this->item().set_column(renumber[(size_t)column()]);
    if (this->p_next())
        (*(row<T>*)this->p_next()).renumber_columns(renumber);
    // return *this;
}

template<class T>
void row<T>::drop_items(dynamic_vector<int> const mask)
{
    if (this->p_next())
    {
        row<T>* next_row = (row<T>*)this->p_next();
        if (!mask[(size_t)next_row->column()])
        {
            this->drop_next_item();
            drop_items(mask);
        }
        else
        {
            next_row->drop_items(mask);
        }
    }

    if (this->p_next() && !mask[(size_t)column()])
    {
        this->drop_first_item();
    }
}

template<class T>
const row<T> operator*(row<T> const& r, T const& t)
{
    row<T> res(r);
    res *= t;
    return res;
}

template<class T>
const row<T> operator*(T const& t, row<T> const& r)
{
    row<T> res(r);
    res *= t;
    return res;
}

template<class T>
const row<T> operator/(row<T> const& r, T const& t)
{
    row<T> res(r);
    res /= t;
    return res;
}


