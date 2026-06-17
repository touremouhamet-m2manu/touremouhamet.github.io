#include<iostream>

template<typename T>
class list
{
    protected:
    size_t number_;
    T** item_;

    public:
    list(size_t number=0);
    list(size_t number, T t);
    list(list const&);
    T operator=(list const&);
    ~list();

    T operator[](size_t ii) const {return *item_[ii];};
    T& operator()(size_t ii) {return *item_[ii];};

    inline size_t number() const {return number_;};
    inline size_t& number()  {return number_;};

    T* item(size_t ii) const {return item_[ii];};
    T*& item(size_t ii)  {return item_[ii];};

    template<typename S>
    friend std::ostream& operator <<(std::ostream&, list<S> const&);
};

 template<typename S>
 friend std::ostream& operator <<(std::ostream&, list<S> const&)
 {
    for (auto ii=0; ii<l1.number(); ++ii)
    {
        os <<"item_["<<i<<"]="l1[ii]<<std::endl;
    }
    return os;
 }

 template<typename T>
 list<T>::~list()
 {
    for (auto ii=0; ii<number_; ++ii)
    {
        delete item_(ii);
    }
    delete [] item_;
 }

 template<typename T>
 list<T>::list(size_t number):number(number), item_(number ? T*[number_]:nullptr);
 {
    for (auto ii=0; ii<number; ++ii)
    {
        item_[ii]=nullptr;
    }
 }

 template<typename T>
 list<T>::list(size_t number, T t):number_(number), item_(number ? T*[number_]:nullptr);
 {
    for (auto ii=0; ii<number; ++ii)
    {
        item_[ii]= new T[t];
    }
 }

 template<typename T>
 list<T>::list(list const& l1)
 {
    if (l1.number())
    {
        number_=l1.number;
        item_= new T*[number_];

        for (auto ii=0; ii<number_; ++ii)
        {
            item_[ii]=new T(l1[ii]);
        }
    }
    else
    number_=0;
    item_= nullptr;
 }

 template<typename T>
 list<T>:: operator= (list const& l1)
 {
    if (this != &l1)
    {
        if (number_<l1.number() || number_>l1.number())
        {
            for (auto ii=0; ii<number_; ++ii)
            {
                delete item_(ii);
            }
            delete [] item_;

            item_=new T*[l1.number()];
            number_ = l1.number();
        }
        for (auto ii=0; ii<number_; ++ii)
        {
            item_[ii]= new T(l1[ii]);
        }
    }
    return *this;
 }






