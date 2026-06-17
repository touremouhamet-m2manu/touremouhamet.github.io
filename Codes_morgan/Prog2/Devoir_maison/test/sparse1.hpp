#pragma once

#include "row1.hpp"
#include "class_dynamic_vector.hpp"
#include <iostream>

template <class T>
class sparse_matrix
{
protected:
    int      nrows_;
    int      ncols_;
    row<T>** rows_;   // tableau de pointeurs sur lignes

public:
    // (1) constructeur par défaut
    explicit sparse_matrix(int Nrows = 0)
        : nrows_(Nrows), ncols_(Nrows)
    {
        rows_ = Nrows ? new row<T>*[Nrows] : nullptr;
        for (int i = 0; i < nrows_; ++i)
            rows_[i] = nullptr;
    }

    // (2) constructeur diagonal (a * I)
    sparse_matrix(int Nrows, T const& a)
        : nrows_(Nrows), ncols_(Nrows)
    {
        rows_ = Nrows ? new row<T>*[Nrows] : nullptr;
        for (int i = 0; i < nrows_; ++i)
            rows_[i] = new row<T>(a, i);
    }

    // constructeur de copie
    sparse_matrix(sparse_matrix const& M)
        : nrows_(M.nrows_), ncols_(M.ncols_)
    {
        rows_ = nrows_ ? new row<T>*[nrows_] : nullptr;
        for (int i = 0; i < nrows_; ++i)
            rows_[i] = M.rows_[i] ? new row<T>(*M.rows_[i]) : nullptr;
    }

    // opérateur d’affectation
    const sparse_matrix& operator=(sparse_matrix const& M)
    {
        if (this != &M)
        {
            for (int i = 0; i < nrows_; ++i)
                delete rows_[i];
            delete[] rows_;

            nrows_ = M.nrows_;
            ncols_ = M.ncols_;
            rows_  = nrows_ ? new row<T>*[nrows_] : nullptr;
            for (int i = 0; i < nrows_; ++i)
                rows_[i] = M.rows_[i] ? new row<T>(*M.rows_[i]) : nullptr;
        }
        return *this;
    }

    ~sparse_matrix()
    {
        for (int i = 0; i < nrows_; ++i)
            delete rows_[i];
        delete[] rows_;
    }

    // (4) accès (i,j)
    T operator()(int i, int j) const
    {
        if (i < 0 || i >= nrows_) return T(0);
        if (!rows_[i])            return T(0);
        return (*(rows_[i]))[j];
    }

    int row_number() const    { return nrows_; }
    int column_number() const { return ncols_; }
    int order() const         { return (nrows_ > ncols_) ? nrows_ : ncols_; }

    row<T>&       row_ref(int i)       { return *rows_[i]; }
    row<T> const& row_ref(int i) const { return *rows_[i]; }

    // (6) opérateurs arithmétiques
    const sparse_matrix& operator+=(sparse_matrix const& M)
    {
        for (int i = 0; i < nrows_; ++i)
        {
            if (M.rows_[i])
            {
                if (!rows_[i]) rows_[i] = new row<T>();
                row_ref(i) += M.row_ref(i); // utilise linked_list::operator+=
            }
        }
        return *this;
    }

    const sparse_matrix& operator-=(sparse_matrix const& M)
    {
        for (int i = 0; i < nrows_; ++i)
        {
            if (M.rows_[i])
            {
                if (!rows_[i]) rows_[i] = new row<T>();
                row<T> tmp = M.row_ref(i);
                tmp *= T(-1);
                row_ref(i) += tmp;
            }
        }
        return *this;
    }

    const sparse_matrix& operator*=(T const& t)
    {
        for (int i = 0; i < nrows_; ++i)
            if (rows_[i]) row_ref(i) *= t;
        return *this;
    }

    // (8) produit matrice * vecteur
    template <class S>
    friend const dynamic_vector<S> operator*(sparse_matrix<S> const& M,
                                             dynamic_vector<S> const& v)
    {
        dynamic_vector<S> res((size_t)M.nrows_, S(0));
        for (int i = 0; i < M.nrows_; ++i)
        {
            if (M.rows_[i])
                res(i) = (*(M.rows_[i])) * v;
        }
        return res;
    }

    // (9) produit matrice * matrice (version simple)
    friend const sparse_matrix<T> operator*(sparse_matrix<T> const& A,
                                            sparse_matrix<T> const& B)
    {
        int n = A.row_number();
        int m = B.column_number();
        sparse_matrix<T> C(n);
        C.ncols_ = m;

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                T sum = T(0);
                int K = A.column_number();
                for (int k = 0; k < K; ++k)
                    sum += A(i, k) * B(k, j);
                if (sum != T(0))
                {
                    if (!C.rows_[i]) C.rows_[i] = new row<T>(sum, j);
                    else            C.row_ref(i).append(sum, j);
                }
            }
        }
        return C;
    }

    // (10) transpose
    friend const sparse_matrix<T> transpose(sparse_matrix<T> const& M)
    {
        sparse_matrix<T> Mt(M.ncols_);
        Mt.ncols_ = M.nrows_;

        for (int i = 0; i < M.nrows_; ++i)
        {
            for (int j = 0; j < M.ncols_; ++j)
            {
                T v = M(i, j);
                if (v != T(0))
                {
                    if (!Mt.rows_[j]) Mt.rows_[j] = new row<T>(v, i);
                    else             Mt.row_ref(j).append(v, i);
                }
            }
        }
        return Mt;
    }

    // (11) diagonale
    friend const sparse_matrix<T> diagonal(sparse_matrix<T> const& M)
    {
        sparse_matrix<T> D(M.row_number());
        D.ncols_ = M.column_number();
        for (int i = 0; i < M.row_number(); ++i)
        {
            T v = M(i, i);
            if (v != T(0))
            {
                if (!D.rows_[i]) D.rows_[i] = new row<T>(v, i);
                else             D.row_ref(i).append(v, i);
            }
        }
        return D;
    }

    // (12) print : non-zéros avec indices
    template <class S>
    friend void print(sparse_matrix<S> const& M)
    {
        for (int i = 0; i < M.nrows_; ++i)
        {
            if (!M.rows_[i]) continue;
            row<S> const* pr = M.rows_[i];
            while (pr)
            {
                std::cout << "i=" << i
                          << " j=" << pr->column()
                          << " val=" << pr->value() << "\n";
                pr = (row<S>*)pr->p_next();
            }
        }
    }
};

// (7) opérateurs libres +, -, * scalaire
template <class S>
const sparse_matrix<S> operator+(sparse_matrix<S> const& M1,
                                 sparse_matrix<S> const& M2)
{
    sparse_matrix<S> R(M1);
    R += M2;
    return R;
}

template <class S>
const sparse_matrix<S> operator-(sparse_matrix<S> const& M1,
                                 sparse_matrix<S> const& M2)
{
    sparse_matrix<S> R(M1);
    R -= M2;
    return R;
}

template <class S>
const sparse_matrix<S> operator*(sparse_matrix<S> const& M, S const& t)
{
    sparse_matrix<S> R(M);
    R *= t;
    return R;
}

template <class S>
const sparse_matrix<S> operator*(S const& t, sparse_matrix<S> const& M)
{
    sparse_matrix<S> R(M);
    R *= t;
    return R;
}
