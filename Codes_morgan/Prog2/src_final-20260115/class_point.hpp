#pragma once
#include <iostream>

template <int NDIM>
class point
{

private:
    double coord[NDIM];

public:
    point();
    point(double xx[NDIM]);
    point(double xx);
    point(double xx, double yy); // only for point<2>
    double operator[](int ii) const;
    void zero();
    ~point() {};                             //  destructor
    point(const point& p);                   //  copy constructor
    const point& operator=(const point& p);  //  point-to-point assignment
    const point& operator+=(const point& p); // adding a point to the current point
    const point& operator-=(const point& p); // subtracting a point to the current point

    double x() const { return coord[0]; };
    double y() const { return coord[1]; };
    double& x() { return coord[0]; };
    double& y() { return coord[1]; };

    template <int N>
    friend const point<N> operator-(const point<N>& p); // unary operator-

    template <int N>
    friend const point<N> operator+(const point<N>& p); // unary operator+

    template <int N>
    friend const point<N> operator+(const point<N>& p, const point<N>& q);

    template <int N>
    friend const point<N> operator-(const point<N>& p, const point<N>& q);

    template <int N>
    friend std::ostream& operator<<(std::ostream& os, const point<N>& p);

    template <int N>
    friend const point<N> operator*(double lambda, const point<N>& p2); // 


    template <int N>
    friend const double operator*(const point<N>& p1, const point<N>& p2); // (occasional) dot-product
};

    typedef point<2> point2d;

    // DEFINitioNS

    template <int NDIM>
    point<NDIM>::point()
    {
        for (int ii = 0; ii < NDIM; ii++)
            this->coord[ii] = 0.;
    }

    template <int NDIM>
    point<NDIM>::point(double xx[NDIM])
    {
        for (int ii = 0; ii < NDIM; ii++)
            this->coord[ii] = xx[ii];
    }

    template <int NDIM>
    point<NDIM>::point(double xx)
    {
        for (int ii = 0; ii < NDIM; ii++)
            this->coord[ii] = xx;
    }

    // specialization
    template <>
    point<2>::point(double xx, double yy)
    {
        this->coord[0] = xx;
        this->coord[1] = yy;
    }

    template <int NDIM>
    double point<NDIM>::operator[](int ii) const
    {
        return this->coord[ii];
    }

    template <int NDIM>
    void point<NDIM>::zero()
    {
        for (int ii = 0; ii < NDIM; ii++)
            this->coord[ii] = 0.;
    }

    template <int NDIM>
    point<NDIM>::point(const point& p)
    {
        for (int ii = 0; ii < NDIM; ii++)
            this->coord[ii] = p.coord[ii];
    }; //  copy constructor

    template <int NDIM>
    const point<NDIM>& point<NDIM>::operator=(const point& p)
    {
        if (this != &p)
        {
            for (int ii = 0; ii < NDIM; ii++)
                this->coord[ii] = p.coord[ii];
        }
        return *this;
    } //  point-to-point assignment

    template <int NDIM>
    const point<NDIM>& point<NDIM>::operator+=(const point& p)
    {
        for (int ii = 0; ii < NDIM; ii++)
            this->coord[ii] += p[ii];
        return *this;
    } //  adding a point to the current point

    template <int NDIM>
    const point<NDIM>& point<NDIM>::operator-=(const point& p)
    {
        for (int ii = 0; ii < NDIM; ii++)
            this->coord[ii] -= p[ii];
        return *this;
    } //  subtracting a point to the current point

    template <int NDIM>
    const point<NDIM> operator+(const point<NDIM>& p, const point<NDIM>& q)
    {
        point<NDIM> s;
        for (int ii = 0; ii < NDIM; ii++)
            s.coord[ii] = p.coord[ii] + q.coord[ii];
        return s;
    } //  add two points

    template <int NDIM>
    const point<NDIM> operator-(const point<NDIM>& p, const point<NDIM>& q)
    {
        point<NDIM> s;
        for (int ii = 0; ii < NDIM; ii++)
            s.coord[ii] = p.coord[ii] - q.coord[ii];
        return s;
    } //  subtract two points

    template <int NDIM>
    const point<NDIM> operator-(const point<NDIM>& p)
    {
        double minus[NDIM];
        for (int ii = 0; ii < NDIM; ii++)
            minus[ii] = -p.coord[ii];
        return point<NDIM>(minus);
    } // unary -

    template <int NDIM>
    const point<NDIM> operator+(const point<NDIM>& p)
    {
        double plus[NDIM];
        for (int ii = 0; ii < NDIM; ii++)
            plus[ii] = p.coord[ii];
        return point<NDIM>(plus);
    } // unary +

    template <int N>
    const double operator*(const point<N>& p1, const point<N>& p2)
    {
        double res{0.};
        for (auto ii = 0; ii < N; ii++)
            res += p1.coord[ii] * p2.coord[ii];
        return res;
    }

    template <int N>
    const point<N> operator*(double lambda, const point<N>& p2)
    {
        double times[N];
        for (auto ii = 0; ii < N; ii++)
            times[ii] = lambda * p2.coord[ii];

        return point<N>(times);
    }

    template <int NDIM>
    std::ostream& operator<<(std::ostream& os, const point<NDIM>& p)
    {
        for (int ii = 0; ii < NDIM; ii++)
            os << p[ii] << " ";
        return os;
    }
