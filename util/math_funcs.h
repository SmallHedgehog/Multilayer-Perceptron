#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>

#ifdef _WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif // _WIN32


namespace mlp {

	inline void softmaxd(const std::vector<double>& value, std::vector<double>& res_) {
		double valueCount = 0;
		std::vector<double>::const_iterator iter = value.cbegin();
		for (; iter != value.end(); ++iter) {
			double item = exp((*iter));
			res_.push_back(item);
			valueCount += item;
		}
		std::vector<double>::iterator it = res_.begin();
		for (; it != res_.end(); ++it) {
			(*it) = (*it) / valueCount;
		}
	}

	inline void softmaxf(const std::vector<float>& value, std::vector<double>& res_) {
		double valueCount = 0;
		std::vector<float>::const_iterator iter = value.cbegin();
		for (; iter != value.end(); ++iter) {
			double item = exp((*iter));
			res_.push_back(item);
			valueCount += item;
		}
		std::vector<double>::iterator it = res_.begin();
		for (; it != res_.end(); ++it) {
			(*it) = (*it) / valueCount;
		}
	}

	template<typename TYPE>
	inline double sigmiod(TYPE x) {
		return 1 / (1 + exp<TYPE>(-x));
	}

	template<typename TYPE>
	inline double tanh(TYPE x) {
		return (exp<TYPE>(x) - exp<TYPE>(-x)) /
			(exp<TYPE>(x) + exp<TYPE>(-x));
	}

	struct generate_random
	{
		inline double operator () (double factor = 0.1){
			return factor * (double(rand()) / RAND_MAX);
		}
	};

}