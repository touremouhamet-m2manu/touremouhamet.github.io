 #ifndef ROW_OBJECT1_HPP
 #define ROW_OBJECT1_HPP

#include "row_element1.hpp"
#include "class_linked_list.hpp"   // à adapter : ta propre liste chaînée
#include "class_dynamic_vector.hpp" // idem, ton vector dynamique

template <class T>
class row : public linked_list< row_element<T> > {
    using base = linked_list< row_element<T> >;

public:
    // (1) constructeur : crée un row avec éventuellement un 1er élément
    row(T const& val = T(0), int col = -1) {
        if (col != -1) {
            row_element<T> e(val, col);
            base::insert_first_item(e); // suppose que ta liste possède ça
        }
    }

    // (2) operator() : renvoie le premier item (const ref)
    row_element<T> const& operator()() const {
        return base::first_item(); // à adapter au nom dans ta liste
    }

    // (3) value() du 1er élément
    T const& value() const {
        return operator()().value();
    }

    // (4) column() du 1er élément
    int column() const {
        return operator()().column();
    }

    // (5) surcharges d’insertion adaptées au sujet
    void insert_next_item(T const& val, int col) {
        row_element<T> e(val, col);
        base::insert_next_item(e);
    }

    void insert_first_item(T const& val, int col) {
        row_element<T> e(val, col);
        base::insert_first_item(e);
    }

    void append(T const& val, int col) {
        row_element<T> e(val, col);
        base::append(e);
    }

    // (6) somme de la ligne par récursion
    T const row_sum() const {
        if (base::empty()) return T(0);
        // valeur du premier élément + somme du reste
        T s = value();
        auto p = base::next(); // pointeur sur le reste de la liste
        if (p) {
            s += ((row<T>*)p)->row_sum();
        }
        return s;
    }

    // (7) operator[] : renvoie valeur en colonne i, ou 0 si absente
    T operator[](int i) const {
        if (base::empty()) return T(0);
        int c = column();
        if (c == i) return value();
        if (c > i)  return T(0);
        auto p = base::next();
        if (!p) return T(0);
        return (*(row<T>*)p)[i];
    }

    // (8) opérateurs *= et /= par scalaire, récursifs
    row<T> const& operator*=(T const& t) {
        if (!base::empty()) {
            base::item_ref().set_value(base::item_ref().value() * t);
            auto p = base::next();
            if (p) (*(row<T>*)p) *= t;
        }
        return *this;
    }

    row<T> const& operator/=(T const& t) {
        if (!base::empty()) {
            base::item_ref().set_value(base::item_ref().value() / t);
            auto p = base::next();
            if (p) (*(row<T>*)p) /= t;
        }
        return *this;
    }

    // (10) produit scalaire avec dynamic_vector
    T operator*(dynamic_vector<T> const& v) const {
        if (base::empty()) return T(0);
        T res = value() * v[column()];
        auto p = base::next();
        if (p) res += (*(row<T>*)p) * v;
        return res;
    }

    // (11) renumérotation des colonnes
    void renumber_columns(dynamic_vector<int> const& renumber) {
        if (base::empty()) return;
        base::item_ref().set_column(renumber[column()]);
        auto p = base::next();
        if (p) (*(row<T>*)p).renumber_columns(renumber);
    }

    // (13) drop_items(mask) – version corrigée de celle du sujet
    void drop_items(dynamic_vector<int> const& mask) {
        if (!base::next()) {
            // fin : vérifier si le premier doit sauter
            if (!mask[column()]) base::drop_first_item();
            return;
        }
        auto* next_row = (row<T>*)base::next();
        if (!mask[next_row->column()]) {
            base::drop_next_item();
            drop_items(mask); // on reste sur le même "courant"
        } else {
            next_row->drop_items(mask);
            if (!mask[column()]) base::drop_first_item();
        }
    }
};

// (12) opérateurs non-membres avec scalaire
template <class T>
row<T> const operator*(row<T> const& r, T const& t) {
    row<T> res(r);
    res *= t;
    return res;
}

template <class T>
row<T> const operator*(T const& t, row<T> const& r) {
    row<T> res(r);
    res *= t;
    return res;
}

template <class T>
row<T> const operator/(row<T> const& r, T const& t) {
    row<T> res(r);
    res /= t;
    return res;
}

 #endif // ROW_OBJECT1_HPP