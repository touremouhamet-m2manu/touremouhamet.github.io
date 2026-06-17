#pragma once
#include <iostream>

template <typename T>
class list
{
protected:
    size_t number_;
    T** item_;

public:
    list(size_t number=0);
    list(size_t number, T t);
    list(list const&);
    ~list();
    list operator=(list const&);
    inline size_t number() const { return number_; };
    inline size_t& number() { return number_; };
    T operator[](size_t ii) const { return *item_[ii]; };
    T& operator()(size_t ii) { return *item_[ii]; };
    T* item(size_t ii) const { return item_[ii]; };
    T*& item(size_t ii) { return item_[ii]; };

    template <typename S>
    friend std::ostream& operator<<(std::ostream&, list<S> const&);
};

//=================================

template <typename S>
std::ostream& operator<<(std::ostream& os, list<S> const& l1)
{
        for (auto i = 0; i < l1.number(); i++)
        {
            os << "item_[" << i << "]= " << l1[i] << std::endl;
        }
    return os;
}

template <typename T>
list<T>::~list()
{
    for (auto ii = 0; ii < number_; ++ii)
    {
            // std::cout << "erase pointer with address = " << item_[ii] << std::endl;
            delete item(ii);
    }
    // std::cout << "erase global pointer with address = " << item_ << std::endl;
    delete[] item_;
}

template <typename T>
list<T>::list(size_t number) : number_(number), item_(number ? new T*[number_] : nullptr)
{
    for (auto ii = 0; ii < number; ++ii)
    {
        item_[ii] = nullptr;
    }
};

template <typename T>
list<T>::list(size_t number, T t) : number_(number), item_(number ? new T*[number_] : nullptr)
{
    for (auto ii = 0; ii < number; ++ii)
    {
        item_[ii] = new T(t);
    }
};

template <typename T>
list<T>::list(list const& l1)
{
    if (l1.number())
    {
        number_ = l1.number();
        item_   = new T*[number_];

        for (auto ii = 0; ii < number_; ++ii)
        {
            item_[ii] = new T(l1[ii]);
        }
    }
    else
    {
        number_ = 0;
        item_   = nullptr;
    }
}

template <typename T>
list<T> list<T>::operator=(list const& l1)
{
    if (this != &l1)
    {
        if (number_ < l1.number() || number_ > l1.number())
        {
            // if (item_ != nullptr)
            // {
                for (auto ii = 0; ii < number_; ++ii)
                {
                    delete item_[ii];
                }
                delete[] item_;
            //}

            item_   = new T*[l1.number()];
            number_ = l1.number();
        }
        for (auto ii = 0; ii < number_; ++ii)
        {
            item_[ii] = new T(l1[ii]);
        }
    }

    return *this;
}