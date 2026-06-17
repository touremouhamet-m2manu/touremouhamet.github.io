template<class T>
T derive_product(const list<T>& f, const list<T>& g, int n) {
    T result = 0;
    for (int i = 0; i <= n; ++i) {
        // Calcul de C(n, i)
        T binomial_coeff = 1;
        for (int j = 1; j <= i; ++j) {
            binomial_coeff *= static_cast<T>(n - i + j) / j;
        }

        result += binomial_coeff * f[i] * g[n - i];
    }
    return result;
}
