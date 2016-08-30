class signature {
private:
	unsigned dim;
	int *s;
public:
	signature(int *sig, unsigned dim){
		s = sig;
		dim = dim;
	}
	signature(vec sig)
	{
		dim = sig.n_rows;
		s = new int[dim];
		for (int i = 0; i < dim; ++i)
		{
			if(sig(i)>0){ s[i]=1 
			} else { s[i]=-1; }
		}
	}
	~signature()
	{
		delete[] s;
	}
		
	bool operator < (const signature& r)
	{
		if(dim > r.dim)
			return true;
		if(dim < r.dim)
			return false;
		for (unsigned i = 0; i < dim; ++i) {
			if (s[i] < r.s[i]) 
				return true;
			if (s[i] > r.s[i]) 
				return false;
		}
		return false;
	}
	bool operator == (const signature& r)
	{
		if(dim == r.dim){
			for (unsigned i = 0; i < dim; ++i) {
				if (s[i] != r.s[i]) 
					return false;
			}
			return true;
		}
		return false;
	}
	bool operator != (const signature& r)
	{
		return !(this == r)
	}
	vec getvec()
	{
		vec v = zeros<vec>(dim);
		for (unsigned i = 0; i < dim; ++i)
		{
			v(i) = s[i];
		}
	}
	int *getarr(){ return s; }
	int getdim(){ return dim; }
};

typedef struct region {
	signature sig;
	vec totvec;
	int numvec;
} region;

class regions {
private:
	std::map <signature, *region> R;
public:
	regions(){}
	void addvector(vec v, nn nurnet){
		signature sig = signature(v);
		if(R[sig] == NULL){
			R[sig] = new region;
			R[sig]->sig = sig;
			R[sig]->totvec = v;
			R[sig]->numvec = 1;
		} 
		if(R[sig] != NULL){
			R[sig]->totvec += v;
			R[sig]->numvec += 1;
		}
	}
	region *getregion(signature sig) 
	{
		if(R[sig] == NULL){
			return NULL;
		} 
		if(R[sig] != NULL){
			return R[sig];
		}
	}
	vec getavgvec(signature sig)
	{
		if(R[sig] == NULL){
			return NULL;
		} 
		if(R[sig] != NULL){
			return (R[sig]->totvec)/(R[sig]->numvec);
		}
	}
	int getpop(signature sig)
	{
		return R[sig]->numvec;
	}
};