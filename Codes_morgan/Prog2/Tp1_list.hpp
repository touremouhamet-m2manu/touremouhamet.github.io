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
       ~list();
       
       list(list const&); // constructeur
       list operator=() //operateur d'affectation
       inline size_t number() const { return number_; };
       inline size_t& number() const { return number_; };
       T operator [] (size_t ii) { return *item_[ii];};
       T& operator () (size_t ii) { return *item_[ii];};
       T* item(size_t ii) const {return item_[ii];};
       T*& item(size_t ii) const {return item_[ii];};
       
       template<typename S>
       friend std::ostream& operator<<(std::ostream&, list<S> const&)
};
//================================================================================
template<typename S>
std::ostream& operator<<(std::ostream&, list<S> const& l1)
{
    for (auto i=0; i<l1.number(); i++)
    {
        os << "item_["<<i<<"]"<<l[1]<<std::endl;
    }
    return os;

}


// Destructeur de tout ce qu'on creer en new
template<typename T>
list<T>::~list()
{
    //if (number())
    {

     for (auto ii=0 ; ii<number; ++ii)
     {
       // if (item_[ii])
            if (item_[ii] != nullptr)
                delete item_[ii];
                //item_[ii]= new T(l1[ii]);
     }
     delete[] item_;
    }
}

template<typename T>
list<T>::list(size_t number, T t): number_(number), item_(number ? new T*[number]:nullptr)
       
       {
            for (auto ii=0 ; ii<number; ++ii)
            {
                item_[ii]= new T(t);
            }
       };
//=================================================================================
template<typename T>
list<T>::list(list const& l1)
{
    if (l1.number())
    {
        number_=l1.number();
        item_=new T*[number_];
    
       for (auto ii=0 ; ii<number; ++ii)
        {
          item_[ii]= new T(l1[ii]);
        }
    }
    else
    {
        number_ = 0;
        item_ = nullptr;
    }
}
template<typename T>
list<T> list<T>::operator=(list const& l1)
{
    if( this != &l1)
    {
       if(number_<l1.number() || number_>l1.number())
       {
         if(item_ != nullptr)
         {
            for (auto ii=0 ; ii<number; ++ii)
            {
                delete[] item_;
                //item_[ii]= new T(l1[ii]);
            }
            delete[] item_;
         }
         
         item_ = new T*[number()];
         number_ = l1.number();
       }
       for (auto ii=0 ; ii<number; ++ii)
        {
          item_[ii]= new T(l1[ii]);
        }
    }
    return *this;
}