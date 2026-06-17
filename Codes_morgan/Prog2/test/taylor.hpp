template<class T>
const T Taylor(list<T> const& f, T const& h) {
    T result = f[0];
    T power_term = h;

    for (int i = 1; i < f.number(); ++i) {
        result += f[i] * power_term;
        power_term *= h / static_cast<T>(i);
    }

    return result;
}
