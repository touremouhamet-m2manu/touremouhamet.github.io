#pragma once
#include <iostream>

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

    linked_list() : p_next_(nullptr) {};
    linked_list(T const& t, linked_list* pN = nullptr) : item_(t), p_next_(pN) {};
    linked_list(linked_list const& L) : item_(L.item()), p_next_(L.p_next() ? new linked_list(*L.p_next()) : nullptr) {} // copy
    ~linked_list() { delete p_next_; };
    linked_list& last();
    size_t length() const;
    void append(T const& t) { last().p_next() = new linked_list(t); };
    void insert_next_item(T const& t);
    void insert_first_item(T const& t);
    const linked_list& operator=(linked_list const&);
    void drop_next_item();
    void drop_first_item();
    void truncate_items(double threshold);
    const linked_list& operator+=(linked_list&); // merge two ordered lists
    linked_list& order(size_t);                  // order a list
};

template <class T>
linked_list<T>& linked_list<T>::order(size_t length)
{
    if (length > 1)
    {
        linked_list<T>* p_scan1 = this;
        for (auto count = 0; count < length / 2 - 1; ++count)
        {
            p_scan1 = p_scan1->p_next();
        }
        linked_list<T>* p_scan2 = p_scan1->p_next();
        p_scan1->p_next()       = nullptr; // important to "cut" the second half of the calling linked_list

        this->order(length / 2);
        (*this).operator+=(p_scan2->order(length - length / 2));
    }
    return *this;
}

template <class T>
const linked_list<T>& linked_list<T>::operator+=(linked_list<T>& L)
{
    linked_list<T>* p_scan1 = this;
    linked_list<T>* p_scan2 = &L;

    if (L.item() < item())
    {
        insert_first_item(L.item());
        p_scan2 = p_scan2->p_next();
    }
    while (p_scan1->p_next())
    {
        // hey : overloaded comparison operator==() for item_ may be used
        if (p_scan2 && p_scan1->item() == p_scan2->item())
        {
            // hey : overloaded comparison operator+=() for item_ may be used
            p_scan1->item() += p_scan2->item();
            p_scan2 = p_scan2->p_next();
        }

        while (p_scan2 && (p_scan2->item() < p_scan1->p_next()->item()))
        {
            p_scan1->insert_next_item(p_scan2->item());
            p_scan1 = p_scan1->p_next();
            p_scan2 = p_scan2->p_next();
        }
        p_scan1 = p_scan1->p_next();

        // for (; p_scan2 && (p_scan2->item() < p_scan1->p_next()->item()); p_scan2 = p_scan2->p_next())
        // {
        //     p_scan1->insert_next_item(p_scan2->item());
        //     p_scan1 = p_scan1->p_next();
        // }
    }
    if (p_scan2 && p_scan1->item() == p_scan2->item())
    {
        p_scan1->item() += p_scan2->item();
        p_scan2 = p_scan2->p_next();
    }

    if (p_scan2)
        p_scan1->p_next() = new linked_list(*p_scan2);

    return *this;
}

template <class T>
void linked_list<T>::truncate_items(double threshold)
{
    if (p_next())
    {
        if (abs(p_next()->item().value()) < threshold)
        {
            drop_next_item();
            truncate_items(threshold);
        }
        else
        {
            p_next()->truncate_items(threshold);
        }
    }
    // check first element
    if (p_next() && item().value() < threshold)
    {
        drop_first_item();
    }
}

template <class T>
void linked_list<T>::drop_first_item()
{
    if (p_next())
    {
        this->item() = this->p_next()->item();
        drop_next_item();
    }
    else
    {
        std::cout << "warning : only one element in the list - impossible to drop" << std::endl;
    }
}

template <class T>
void linked_list<T>::drop_next_item()
{
    if (p_next())
    {
        if (p_next()->p_next())
        {
            linked_list<T>* keep = p_next();
            p_next()             = p_next()->p_next();
            keep->item().~T();
        }
        else
        {
            delete p_next();
            p_next() = nullptr;
        }
    }
}

template <class T>
void linked_list<T>::insert_first_item(T const& t)
{
    p_next() = new linked_list<T>(item(), p_next());
    item()   = t;
}

template <class T>
void linked_list<T>::insert_next_item(T const& t)
{
    p_next() = new linked_list<T>(t, p_next());
}

template <class T>
void print(linked_list<T> const& mylist)
{
    std::cout << mylist.item() << std::endl;
    if (mylist.p_next())
        print(*mylist.p_next());
}

template <class T>
linked_list<T>& linked_list<T>::last()
{
    return p_next() ? p_next()->last() : *this;
}

template <class T>
size_t linked_list<T>::length() const
{
    return p_next() ? p_next()->length() + 1 : 1;
}

template <class T>
const linked_list<T>& linked_list<T>::operator=(linked_list<T> const& L)
{
    if (this != &L)
    {
        item() = L.item();

        if (p_next())
        {
            if (L.p_next())
            {
                *p_next_ = *L.p_next(); // recursive call of operator=()
            }
            else
            {
                delete p_next();
                p_next() = nullptr;
            }
        }
        else if (L.p_next())
        {
            p_next_ = new linked_list(*L.p_next());
        }
    }
    return *this;
} //  assignment operator
