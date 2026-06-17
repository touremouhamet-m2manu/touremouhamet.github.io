#pragma once

#include "element1.hpp"
#include "class_linked_list.hpp"
#include "class_dynamic_vector.hpp"

template <class T>
class row : public linked_list< row_element<T> >
{
    using base = linked_list< row_element<T> >;

public:
    // (1) constructeur
    row(T const& val = T(0), int col = -1)
        : base()
    {
        if (col != -1)
        {
            row_element<T> e(val, col);
            this->item() = e;
        }
    }

    // (2) premier item
    row_element<T> const& operator()() const
    {
        return this->item();
    }

    // (3) valeur du premier élément
    T const& value() const
    {
        return this->item().value();
    }

    // (4) colonne du premier élément
    int column() const
    {
        return this->item().column();
    }

    // (5) versions locales des insertions
    void insert_next_item(T const& val, int col)
    {
        row_element<T> e(val, col);
        base::insert_next_item(e);
    }

    void insert_first_item(T const& val, int col)
    {
        row_element<T> e(val, col);
        base::insert_first_item(e);
    }

    void append(T const& val, int col)
    {
        row_element<T> e(val, col);
        base::append(e);
    }

    // (6) somme de la ligne (récursive)
    T const row_sum() const
    {
        T s = value();
        if (this->p_next())
        {
            s += ((row<T>*)this->p_next())->row_sum();
        }
        return s;
    }

    // (7) operator[] : valeur en colonne i, ou 0 si absente
    T operator[](int i) const
    {
        int c = column();
        if (c == i) return value();
        if (c >  i) return T(0);
        if (!this->p_next()) return T(0);
        return (*(row<T>*)this->p_next())[i];
    }

    // (8) *= et /= par scalaire
    const row<T>& operator*=(T const& t)
    {
        this->item().set_value(this->item().value() * t);
        if (this->p_next())
            (*(row<T>*)this->p_next()) *= t;
        return *this;
    }

    const row<T>& operator/=(T const& t)
    {
        this->item().set_value(this->item().value() / t);
        if (this->p_next())
            (*(row<T>*)this->p_next()) /= t;
        return *this;
    }

    // (10) produit scalaire ligne * dynamic_vector
    T operator*(dynamic_vector<T> const& v) const
    {
        T res = value() * v[(size_t)column()];
        if (this->p_next())
            res += (*(row<T>*)this->p_next()) * v;
        return res;
    }

    // (11) renumérotation des colonnes
    void renumber_columns(dynamic_vector<int> const& renumber)
    {
        this->item().set_column(renumber[(size_t)column()]);
        if (this->p_next())
            (*(row<T>*)this->p_next()).renumber_columns(renumber);
    }

    // (13) drop_items(mask) : 0 => supprimer, 1 => garder
    void drop_items(dynamic_vector<int> const& mask)
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
};

// (12) opérateurs libres avec scalaire
template <class T>
const row<T> operator*(row<T> const& r, T const& t)
{
    row<T> res(r);
    res *= t;
    return res;
}

template <class T>
const row<T> operator*(T const& t, row<T> const& r)
{
    row<T> res(r);
    res *= t;
    return res;
}

template <class T>
const row<T> operator/(row<T> const& r, T const& t)
{
    row<T> res(r);
    res /= t;
    return res;
}
