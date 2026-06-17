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

    T& operator()(size_t ii) { return *item_[ii]; }
    T operator[](size_t ii) const { return *item_[ii]; }
    inline size_t number() const { return number_; };
};
