#include "class_list.hpp"

template <typename T>
class factoriel_table : public list<list<T>>
{
    public :
    factoriel_table (size_t N)
    T operator()(T const& n) {return (*this)[n];}
};
template<typename T>
factoriel_table<T>::factoriel_table (size_t N):class_list<T>(N+1)
{
    this->item_[0]=new T(1); 
    for (size_t i = 1; i <= N; i++)
    {
        this->item_[i] = new T(i*(*this->item_[i-1]))
    }
    
}
