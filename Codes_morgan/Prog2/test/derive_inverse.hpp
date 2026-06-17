template<typename T>
const list<T> derive_inverse(T const& a, int n) {
    list<T> derivatives(n + 1);
    derivatives(0) = 1 / a; // f(a) = 1/a

    for (int k = 1; k <= n; ++k) {
        derivatives(k) = -static_cast<T>(k) / a * derivatives(k - 1);
    }

    return derivatives;
}
