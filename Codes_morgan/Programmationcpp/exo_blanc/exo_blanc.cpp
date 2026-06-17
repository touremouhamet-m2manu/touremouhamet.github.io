#include <iostream>
#include <cstddef>

//
// =========================
// Classe list<T>
// =========================
//

template <typename T>
class list
{
protected:
    size_t number_;
    T** item_;

public:
    list(size_t number = 0);
    list(size_t number, T t);
    list(list const&);
    ~list();
    list operator=(list const&);

    T operator[](size_t ii) const { return *item_[ii]; }
    T& operator()(size_t ii) { return *item_[ii]; }

    inline size_t number() const { return number_; }
};

// --- Constructeur vide ---
template <typename T>
list<T>::list(size_t number) : number_(number)
{
    item_ = (number ? new T*[number] : nullptr);
    for (size_t i = 0; i < number_; i++)
        item_[i] = nullptr;
}

// --- Constructeur avec initialisation ---
template <typename T>
list<T>::list(size_t number, T t) : number_(number)
{
    item_ = (number ? new T*[number] : nullptr);
    for (size_t i = 0; i < number_; i++)
        item_[i] = new T(t);
}

// --- Constructeur de copie ---
template <typename T>
list<T>::list(list const& l1)
{
    number_ = l1.number_;
    item_   = (number_ ? new T*[number_] : nullptr);

    for (size_t i = 0; i < number_; i++)
        item_[i] = new T(l1[i]);
}

// --- Destructeur ---
template <typename T>
list<T>::~list()
{
    for (size_t i = 0; i < number_; i++)
        delete item_[i];
    delete[] item_;
}

// --- Opérateur d’affectation ---
template <typename T>
list<T> list<T>::operator=(list const& l1)
{
    if (this != &l1)
    {
        for (size_t i = 0; i < number_; i++)
            delete item_[i];
        delete[] item_;

        number_ = l1.number_;
        item_ = new T*[number_];
        for (size_t i = 0; i < number_; i++)
            item_[i] = new T(l1[i]);
    }
    return *this;
}

//
// =========================
// Classe factorial_table<T>
// =========================
// (l'exo blanc du contrôle)
// =========================
//

template <typename T>
class factorial_table : public list<T>
{
public:
    factorial_table(size_t N);
    T operator()(size_t n) const { return (*this)[n]; }
};

// --- Construction des factorielles ---
template <typename T>
factorial_table<T>::factorial_table(size_t N)
    : list<T>(N + 1)
{
    // 0! = 1
    this->item_[0] = new T(1);

    // n! = n × (n-1)!
    for (size_t n = 1; n <= N; n++)
    {
        this->item_[n] = new T(n * (*this->item_[n - 1]));
    }
}

//
// =========================
// MAIN
// =========================
//

int main()
{
    factorial_table<long double> F(10);

    std::cout << "0! = " << F(0) << std::endl;
    std::cout << "1! = " << F(1) << std::endl;
    std::cout << "5! = " << F(5) << std::endl;
    std::cout << "10! = " << F(10) << std::endl;

    return 0;
}
