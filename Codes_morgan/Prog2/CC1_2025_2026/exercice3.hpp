
template<typename S>
const S horner_polynomial(polynomial<S> const& p, S x) 
{
    if (p.number_ == 0) return 0;
    
    S result = p.item_[p.number_ - 1];
    
    for (auto ii = p.number_ - 2; ii >= 0; ii--) {
        result = result * x + p.item_[ii];
    }
    
    return result;
}