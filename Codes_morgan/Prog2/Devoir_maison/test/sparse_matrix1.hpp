 #ifndef SPARSE_MATRIX1_H
 #define SPARSE_MATRIX1_H

#include "row_object1.hpp"
#include "class_dynamic_vector.hpp"
#include <iostream>

template <class T>
class sparse_matrix {
    int nrows_;
    int ncols_;
    row<T>** rows_; // tableau de pointeurs vers lignes

public:
    // (1) constructeur par défaut
    sparse_matrix(int Nrows = 0)
        : nrows_(Nrows), ncols_(Nrows) {
        rows_ = new row<T>*[nrows_];
        for (int i = 0; i < nrows_; ++i) rows_[i] = nullptr;
    }

    // (2) constructeur diagonal : M = a * I
    sparse_matrix(int Nrows, T const& a)
        : nrows_(Nrows), ncols_(Nrows) {
        rows_ = new row<T>*[nrows_];
        for (int i = 0; i < nrows_; ++i) {
            rows_[i] = new row<T>(a, i); // diag(i,i) = a
        }
    }

    // destructeur
    ~sparse_matrix() {
        for (int i = 0; i < nrows_; ++i) delete rows_[i];
        delete[] rows_;
    }

    // accès lecture à l’élément (i,j)
    T operator()(int i, int j) const {
        if (i < 0 || i >= nrows_) return T(0);
        if (!rows_[i]) return T(0);
        return (*(rows_[i]))[j]; // utilise row::operator[]
    }

    int row_number() const { return nrows_; }
    int column_number() const { return ncols_; }
    int order() const { return (nrows_ > ncols_) ? nrows_ : ncols_; }

    // accès à la ligne (const / non-const)
    row<T> const& row_ref(int i) const { return *(rows_[i]); }
    row<T>&       row_ref(int i)       { return *(rows_[i]); }

    // (6) opérateurs arithmétiques
    sparse_matrix<T> const& operator+=(sparse_matrix<T> const& M) {
        // on suppose mêmes dimensions
        for (int i = 0; i < nrows_; ++i) {
            if (M.rows_[i]) {
                if (!rows_[i]) rows_[i] = new row<T>();
                // on utilise operator+= des linked_list/row
                row_ref(i) += M.row_ref(i);
            }
        }
        return *this;
    }

    sparse_matrix<T> const& operator-=(sparse_matrix<T> const& M) {
        for (int i = 0; i < nrows_; ++i) {
            if (M.rows_[i]) {
                if (!rows_[i]) rows_[i] = new row<T>();
                row_ref(i) -= M.row_ref(i);
            }
        }
        return *this;
    }

    sparse_matrix<T> const& operator*=(T const& t) {
        for (int i = 0; i < nrows_; ++i) {
            if (rows_[i]) row_ref(i) *= t;
        }
        return *this;
    }

    // (8) produit matrice * vecteur
    template <class S>
    friend dynamic_vector<S> const
    operator*(sparse_matrix<S> const& M, dynamic_vector<S> const& v) {
        dynamic_vector<S> res(M.nrows_);
        for (int i = 0; i < M.nrows_; ++i) {
            if (M.rows_[i]) res[i] = (*(M.rows_[i])) * v;
            else            res[i] = S(0);
        }
        return res;
    }

    // (9) produit matrice * matrice (logique via combinaisons de lignes)
    friend sparse_matrix<T> const
    operator*(sparse_matrix<T> const& A, sparse_matrix<T> const& B) {
        int n = A.row_number();
        int m = B.column_number();
        sparse_matrix<T> C(n);
        C.ncols_ = m;

        // idée simple : pour chaque ligne i de B, combiner les lignes de A
        for (int i = 0; i < n; ++i) {
            if (!B.rows_[i]) continue;
            // BiA = somme_j Bij * Aj
            // à compléter selon comment tu veux combiner row<T>
            // (par exemple via un accumulateur row<T> temp; puis append)
        }
        return C;
    }

    // (10) transpose
    friend sparse_matrix<T> const transpose(sparse_matrix<T> const& M) {
        sparse_matrix<T> Mt(M.ncols_);
        Mt.ncols_ = M.nrows_;
        // à remplir : parcourir toutes les lignes, tous les éléments,
        // et insérer dans la colonne/ligne transposée.
        return Mt;
    }

    // (11) diagonale
    friend sparse_matrix<T> const diagonal(sparse_matrix<T> const& M) {
        sparse_matrix<T> D(M.row_number());
        D.ncols_ = M.column_number();
        for (int i = 0; i < M.row_number(); ++i) {
            T val = M(i, i);
            if (val != T(0)) {
                if (!D.rows_[i]) D.rows_[i] = new row<T>();
                D.row_ref(i).append(val, i);
            }
        }
        return D;
    }

    // (12) print
    template <class S>
    friend void print(sparse_matrix<S> const& M) {
        for (int i = 0; i < M.nrows_; ++i) {
            if (!M.rows_[i]) continue;
            // parcourir la row<T> avec ton itérateur ou récursion
            // et afficher (i, colonne, valeur)
        }
    }
};

// (7) opérateurs libres +, -, *
template <class S>
sparse_matrix<S> const operator+(sparse_matrix<S> const& M1,
                                 sparse_matrix<S> const& M2) {
    sparse_matrix<S> R(M1);
    R += M2;
    return R;
}

template <class S>
sparse_matrix<S> const operator-(sparse_matrix<S> const& M1,
                                 sparse_matrix<S> const& M2) {
    sparse_matrix<S> R(M1);
    R -= M2;
    return R;
}

template <class S>
sparse_matrix<S> const operator*(sparse_matrix<S> const& M, S const& t) {
    sparse_matrix<S> R(M);
    R *= t;
    return R;
}

template <class S>
sparse_matrix<S> const operator*(S const& t, sparse_matrix<S> const& M) {
    sparse_matrix<S> R(M);
    R *= t;
    return R;
}

 #endif // SPARSE_MATRIX1_H
