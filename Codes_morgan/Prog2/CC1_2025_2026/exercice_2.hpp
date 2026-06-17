template<typename S>
const polynomial<S> operator*(S c, polynomial<S> const& p);
{
    polynomial<S> r(p.degree());
    for(auto ii=0;i<p.number();ii++)
    {
        r(i) = c * p[ii];
    }
        
    return r;
}

template<typename S>
const polynomial<S> operator*( polynomial<S> const& p, S c) ;
{
    return c * p;
}


template<typename S>
const polynomial<S> operator+(polynomial<S> const& A, polynomial<S> const& B);
{
    auto m = std::max(A.degree(), B.degree());
    polynomial<S> r(m);

    for(auto ii=0;ii<=m;++ii) 
    {
        S a = (i <= A.degree() ? A[i] : 0);
        S b = (i <= B.degree() ? B[i] : 0);
        R(i) = a + b;
    }
    return r;
}
