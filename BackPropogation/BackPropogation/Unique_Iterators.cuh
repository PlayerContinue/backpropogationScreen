#include <thrust/iterator/iterator_adaptor.h>

namespace Unique_Iterator{
	// derive repeat_iterator from iterator_adaptor
	template<typename Iterator>
	class repeat_iterator
		: public thrust::iterator_adaptor <
		repeat_iterator<Iterator>, // the first template parameter is the name of the iterator we're creating
		Iterator                   // the second template parameter is the name of the iterator we're adapting
		// we can use the default for the additional template parameters
		>
	{
	public:
		// shorthand for the name of the iterator_adaptor we're deriving from
		typedef thrust::iterator_adaptor <
			repeat_iterator<Iterator>,
			Iterator
		> super_t;
		__host__ __device__
			repeat_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
		// befriend thrust::iterator_core_access to allow it access to the private interface below
		friend class thrust::iterator_core_access;
	private:
		// repeat each element of the adapted range n times
		unsigned int n;
		// used to keep track of where we began
		const Iterator begin;
		// it is private because only thrust::iterator_core_access needs access to it
		__host__ __device__
			typename super_t::reference dereference() const
		{
			return *(begin + (this->base() - begin) / n);
		}
	};



	//Special Iterator for skipping a set number of places on each iteration
	template<typename Iterator>
	class skip_iterator : public thrust::iterator_adaptor < skip_iterator<Iterator>, Iterator >
	{
	public:
		// shorthand for the name of the iterator_adaptor we're deriving from
		typedef thrust::iterator_adaptor <
			skip_iterator<Iterator>,
			Iterator
		> super_t;
		__host__ __device__
			skip_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
		// befriend thrust::iterator_core_access to allow it access to the private interface below
		friend class thrust::iterator_core_access;
	private:
		// skip n elements
		unsigned int n;
		// used to keep track of where we began
		const Iterator begin;
		// it is private because only thrust::iterator_core_access needs access to it
		__host__ __device__
			typename super_t::reference dereference() const
		{
			
			return *(begin + ((this->base() - begin)*n));
		}
	};

	template<typename Iterator>
	Unique_Iterator::skip_iterator<Iterator> make_skip_iterator(Iterator it, int n){
		return Unique_Iterator::skip_iterator<Iterator>(it, n);
	};

	//Special Iterator for skipping a set number of places on each iteration and then counting and skipping again
	/*template<typename Iterator>
	class skip_and_count_iterator : public thrust::iterator_adaptor < skip_iterator<Iterator>, Iterator >
	{
	public:
		// shorthand for the name of the iterator_adaptor we're deriving from
		typedef thrust::iterator_adaptor <
			skip_iterator<Iterator>,
			Iterator
		> super_t;
		__host__ __device__
			skip_and_count_iterator(const Iterator &x, int n, int count) : super_t(x), begin(x), n(n), count(count) {}
		// befriend thrust::iterator_core_access to allow it access to the private interface below
		friend class thrust::iterator_core_access;
	private:
		// skip n elements
		unsigned int n;

		//count the number before next skip
		unsigned int count;

		// used to keep track of where we began
		const Iterator begin;
		// it is private because only thrust::iterator_core_access needs access to it
		__host__ __device__
			typename super_t::reference dereference() const
		{
			return *(begin + ((this->base() - begin)*n));
		}
	};

	template<typename Iterator>
	Unique_Iterator::skip_and_count_iterator skip_and_count_iterator(Iterator it, int n, int count){
		return Unique_Iterator::skip_and_count_iterator<Iterator>(it, n,count);
	}*/
}