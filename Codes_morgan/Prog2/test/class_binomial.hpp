#include "class_dynamic_vector.hpp" // Supposons que cette classe existe

template<typename T>
class binomial : public list<dynamicVector<T>> {
public:
    binomial(int M) : list<dynamicVector<T>>(M + 1) {
        for (int n = 0; n <= M; ++n) {
            dynamicVector<T> line(n + 1);
            line(0) = 1;
            line(n) = 1;

            for (int k = 1; k < n; ++k) {
                line(k) = (*this)(n - 1, k - 1) + (*this)(n - 1, k);
            }

            (*this)(n) = line;
        }
    }

    T operator()(int n, int k) const {
        return (*this)[n][k];
    }
};
