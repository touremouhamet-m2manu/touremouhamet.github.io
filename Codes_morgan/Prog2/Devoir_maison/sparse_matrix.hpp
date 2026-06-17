#pragma once
#include<iostream>

#include "row_object.hpp"
#include "class_dynamic_vector.hpp"
#include "class_list.hpp"

template<class T>
class sparse_matrix : public list<row<T>>
{
protected:
    int nrows_;
    int ncols_;
    
public:
    sparse_matrix(int Nrows = 0); 
    sparse_matrix(int Nrows, T const& a); 
    sparse_matrix(sparse_matrix const& M); 

    const sparse_matrix& operator=(sparse_matrix const& M);
    ~sparse_matrix();
    T operator()(int i, int j) const;
    
    int row_number() const; 
    int column_number() const; 
    int order () const;
    

    row<T>& row_ref(int i); 
    row<T> const& row_ref(int i) const;
   
    const sparse_matrix& operator+=(sparse_matrix const& M);
    const sparse_matrix& operator-=(sparse_matrix const& M);
    const sparse_matrix& operator*=(T const& t);

    template<class S>
    friend const dynamic_vector<S> operator*(sparse_matrix<S> const& M, dynamic_vector<S> const& v);
    
    template<class U>
    friend const sparse_matrix<U> operator*(sparse_matrix<U> const& A, sparse_matrix<U> const& B);
    
    template<class U>
    friend const sparse_matrix<U> transpose(sparse_matrix<U> const& M);

    template<class U>
    friend const sparse_matrix<U> diagonal(sparse_matrix<U> const& M);

    template<class S>
    friend void print(sparse_matrix<S> const& M);
    

};

template<class T>
sparse_matrix<T>::sparse_matrix(int Nrows) : list<row<T>>((size_t)Nrows), nrows_(Nrows), ncols_(Nrows)
{
    for (int i = 0; i < nrows_; ++i)
        this->item(i) = nullptr;
}

template<class T>
sparse_matrix<T>::sparse_matrix(int Nrows, T const& a) : list<row<T>>((size_t)Nrows), nrows_(Nrows), ncols_(Nrows)
{
    for (int i = 0; i < nrows_; ++i)
        this->item(i) = new row<T>(a, i);
}

template<class T>
sparse_matrix<T>::sparse_matrix(sparse_matrix const& M) : list<row<T>>(M), nrows_(M.nrows_), ncols_(M.ncols_)
{
    for (int i = 0; i < nrows_; ++i)
    {
        if (M.item(i))
            this->item(i) = new row<T>(*M.item(i));
        else
            this->item(i) = nullptr;
    }
}

template<class T>
const sparse_matrix<T>& sparse_matrix<T>::operator=(sparse_matrix<T> const& M)
{
    if (this != &M)
    {
        for (int i = 0; i < nrows_; ++i)
            delete this->item(i);

        list<row<T>>::operator=(M);

        nrows_ = M.nrows_;
        ncols_ = M.ncols_;

        for (int i = 0; i < nrows_; ++i)
        {
            if (M.item(i))
                this->item(i) = new row<T>( *M.item(i) );
            else
                this->item(i) = nullptr;
        }
    }
    return *this;
}

template<class T>
sparse_matrix<T>::~sparse_matrix()
{
    for (int i = 0; i < nrows_; ++i)
        delete this->item(i);
}

template<class T>
T sparse_matrix<T>::operator()(int i, int j) const
{
    if (i < 0 || i >= nrows_) return T(0);
    if (!this->item(i)) return T(0);
    return (*this->item(i))[j];
}

template<class T>
int sparse_matrix<T>::row_number() const
{
    return nrows_;
}


template<class T>
int sparse_matrix<T>::column_number() const
{
    return ncols_;
}

template<class T>
int sparse_matrix<T>::order() const
{
    return (nrows_ > ncols_) ? nrows_ : ncols_;
}

template<class T>
row<T>& sparse_matrix<T>::row_ref(int i)
{
    return *this->item(i);
}

template<class T>
row<T> const& sparse_matrix<T>::row_ref(int i) const
{
    return *this->item(i);
}

template<class T>
const sparse_matrix<T>& sparse_matrix<T>::operator+=(sparse_matrix<T> const& M)
{
    for (int i = 0; i < nrows_; ++i)
    {
        if (M.item(i))
        {
            if (!this->item(i)) 
                this->item(i) = new row<T>();
            row_ref(i) += M.row_ref(i);
        }
    }
    return *this;
}


template<class T>
const sparse_matrix<T>& sparse_matrix<T>::operator-=(sparse_matrix<T> const& M)
{
    for (int i = 0; i < nrows_; ++i)
    {
        if (M.item(i))
        {
            if (!this->item(i)) 
                this->item(i) = new row<T>();
            row<T> tmp = M.row_ref(i);
            tmp *= T(-1);
            row_ref(i) += tmp;
        }
    }
    return *this;
}


template<class T>
const sparse_matrix<T>& sparse_matrix<T>::operator*=(T const& t)
{
    for (int i = 0; i < nrows_; ++i)
        if (this->item(i)) row_ref(i) *= t;
    return *this;
}


    
template <class S>
const dynamic_vector<S> operator*(sparse_matrix<S> const& M, dynamic_vector<S> const& v)
{
    dynamic_vector<S> res((size_t)M.nrows_, S(0));
    for (int i = 0; i < M.nrows_; ++i)
    {
        if (M.item(i))
            res((size_t)i) = (*M.item(i)) * v;
    }
    return res;
}


template<class T>
const sparse_matrix<T> operator*(sparse_matrix<T> const& A, sparse_matrix<T> const& B)
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
            for (int k = 0; k < K; ++k )
                sum += A(i, k) * B(k, j);
            if (sum != T(0))
            {
                if (!C.item(i)) 
                    C.item(i) = new row<T>(sum, j);
                else 
                    C.row_ref(i).append(sum, j);
            }
        }
    }
    return C;
}


template<class T>
const sparse_matrix<T> transpose(sparse_matrix<T> const& M)
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
                if (!Mt.item(j)) 
                    Mt.item(j) = new row<T>(v, i);
                else  
                    Mt.row_ref(j).append(v, i);
            }
        }
    }
    return Mt;
}

template<class T>
const sparse_matrix<T> diagonal(sparse_matrix<T> const& M)
{
    sparse_matrix<T> D(M.row_number());
    D.ncols_ = M.column_number();

    for (int i = 0; i < M.row_number(); ++i)
    {
        T v = M(i, i);
        if (v != T(0))
        {
            if (!D.item(i))
                D.item(i) = new row<T>(v, i);
            else
                D.row_ref(i).append(v, i);
        }
    }
    return D;
}

template<class S>
 void print(sparse_matrix<S> const& M)
{
    for (int i = 0; i < M.nrows_; ++i)
    {
        if (!M.item(i)) continue;
        row<S> const* pr = M.item(i);

        while (pr)
        {
            std::cout <<"i=" << i 
                        <<"j=" << pr->column()
                        <<"val=" << pr->value() <<"\n";
            pr = (row<S>*)pr->p_next();
        }
        
    }
}











template<class S>
const sparse_matrix<S> operator+(sparse_matrix<S> const& M1, sparse_matrix<S> const& M2)
{
    sparse_matrix<S> R(M1);
    R += M2;
    return R;
}

template<class S>
const sparse_matrix<S> operator-(sparse_matrix<S> const& M1, sparse_matrix<S> const& M2)
{
    sparse_matrix<S> R(M1);
    R -= M2;
    return R;
}

template<class S>
const sparse_matrix<S> operator*(sparse_matrix<S> const& M, S const& t)
{
    sparse_matrix<S> R(M);
    R *= t;
    return R;
}

template<class S>
const sparse_matrix<S> operator*(S const& t, sparse_matrix<S> const& M)
{
    sparse_matrix<S> R(M);
    R *= t;
    return R;
}