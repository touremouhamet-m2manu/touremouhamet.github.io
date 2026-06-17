#pragma once
#include "class_dynamic_vector.hpp"
#include "class_list.hpp"

template <typename T>
class binomial : public list<dynamic_vector<T>>
{
    public:
    
    binomial(size_t size);
    T operator()(T const& n, T const& k){return this->item_[n]->operator[](k);};
};

template <typename T>
binomial<T>::binomial(size_t size) : list<dynamic_vector<T>>(size)
{
    for (auto ii = 0; ii < size; ++ii)
    {
        this->item_[ii] = new dynamic_vector<T>(ii + 1);
    }
    //(*item_[0])(0) = 1;
    this->item_[0]->operator()(0) = 1;
    this->item_[1]->operator()(0) = 1;
    this->item_[1]->operator()(1) = 1;
    // populate the reccurence
    for (auto ii = 2; ii < size; ii++)
    {
        this->item_[ii]->operator()(0) = 1.;
        this->item_[ii]->operator()(ii) = 1.;

        for (auto j = 1; j <= ii - 1; j++)
        {
            (*this->item_[ii])(j) = this->item_[ii - 1]->operator()(j - 1) + this->item_[ii - 1]->operator()(j);
            //this->operator(ii)(j) = this->item_[ii - 1]->operator()(j - 1) + this->item_[ii - 1]->operator()(j);
        }
    }
}