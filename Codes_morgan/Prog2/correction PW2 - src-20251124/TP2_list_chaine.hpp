#pragma once

template <class T>
class linked_list
{
protected:
    T item_;
    linked_list* p_next_;

public:
    inline T item() const { return item_; };
    inline T& item() { return item_; };
    inline linked_list* p_next() const { return p_next_; };
    inline linked_list*& p_next() { return p_next_; };
    //void print();

    inline T& item() { return item_; };

    linked_list() : p_next_(nullptr);
    linked_list(T const& t, linked_list* pN = nullptr) : item_(t), p_next_(pN) {};
    linked_list(linked_list const& L) : item_(L.item()), p_next_(L.next() ? new linked_list(*L.p_next()) : nullptr) {} // copy
    ~linked_list(){delete p_next_;};
    void append(T const& t ){last ( ).p_next_ = new linked_list ( t ) ;}
    // append some new item
};



template <class T>
linked_list<T>::~linked_list()
{
    delete p_next_;
    p_next_ = nullptr;
}

template <class T>
void print(linked_list<T> const& mylist)
{
    std::cout<<mylist.item() << std::endl;
    if (mylist.next()) print(*mylist.next());
}
template<class T>
linked_list<T>& linked_list<T>::last()
{
    return p_next() ? p_next() -> last() : *this;
}

template<class T>
size_t linked_list<T>::length() const
{
    return p_next() ? p_next() -> length()+1 : 1;
}


