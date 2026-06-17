#include "class_list.hpp"
#include "class_dynamic_vector.hpp"

template<typename T>
class binomial : public list<dynamic_vecteur<T>>
{
    public :
    binomial(size_t size);
    T operator()(T const& n, T const& k){return item(n)(k);};
};
// On definit ce constructeur
template<typename T>
binomial<T>::binomial(size_t size):list<dynamic_vector<T>>()size
{
    for (auto ii=0; ii<size; ++ii)
    {
        this->item_[ii] = new dynamic_vector<T>(ii+1);
    }
    (*item_[0])(0) = 1;
    //this->item_[0]-> operator()(0) = 1;
    this->item_[0]-> operator()(0) = 1;
    this->item_[1]-> operator()(0) = 1;
    this->item_[1]-> operator()(1) = 1;
    for (auto ii=2; ii < size; ii++)
    {

    }
}