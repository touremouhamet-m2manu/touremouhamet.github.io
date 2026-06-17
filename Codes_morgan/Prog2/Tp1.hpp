#include<iostream>

template<typename T>

class list
{
    protected:
        size_t number_;
        T** item_;
    public:
        
       list(size_t number): number_(number), item_(number ? new T*[number]:nullptr);
       list(size_t number, T t);
       inline size_t number() const { return number_; };
       inline size_t& number() const { return number_; };
       T operator [] (size_t ii) { return *item_[ii];};
       T& operator () (size_t ii) { return *item_[ii];};
       T* item() (size_t ii) const {return item_[ii];};
       T*& item() (size_t ii) const {return item_[ii];};


};
//================================================================================
template<typename T>
list<T>::list(size_t number, T t): number_(number), item_(number ? new T*[number]:nullptr)
       
       {
            for (auto ii=0 ; ii<number; ++ii)
            {
                item_[ii]= new T(t);
            }
       };
      
