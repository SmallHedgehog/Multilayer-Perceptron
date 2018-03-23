#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <functional>

#include "util/math_funcs.h"


namespace mlp
{
	// Weights init type
	typedef enum ParaInitType{
		CONSTANT,
		XARIVER
	}PIT;

	class Neuron
	{
	public:
		Neuron() : input_nums(0), bias(0), pit(PIT::CONSTANT)
		{}
		Neuron(size_t in_nums, PIT _pit, double constant=0) {
			assert(in_nums > 0);
			input_nums = in_nums;
			bias = 0.0;
			weights.clear();
			if (_pit == PIT::CONSTANT) {
				pit = PIT::CONSTANT;
				weights.resize(input_nums, constant);
			}
			else if (_pit == PIT::XARIVER) {
				pit = PIT::XARIVER;
				weights.resize(input_nums);
				std::generate_n(weights.begin(), input_nums, generate_random());
			}
		}
		~Neuron() {
			input_nums = 0;
			bias = 0.0;
			weights.clear();
			pit = PIT::CONSTANT;
		}

		inline void set_input_nums(size_t in_nums) {
			input_nums = in_nums;
		}

		inline size_t get_input_nums() const {
			return input_nums;
		}

		inline void set_bias(double _bias) {
			bias = _bias;
		}

		inline double get_bias() const {
			return bias;
		}

		inline size_t get_weights_size() const {
			return weights.size();
		}

		inline const std::vector<double>& get_weights() const {
			return weights;
		}

		inline void update_weights(size_t idx, double _diff, double lr) {
			assert((idx >= 0 && idx < weights.size()));
			weights[idx] += lr * _diff;
		}

		inline double multiadd(std::vector<double>& x_vec) {
			assert(x_vec.size() == weights.size());
			double ma = 0.0;
			for (size_t idx = 0; idx != weights.size(); ++idx) {
				ma += weights[idx] * x_vec[idx];
			}
			return (ma + bias);
		}

		inline double activation_func(std::vector<double>& x_vec,
			std::function<double(double)> ac) {
			return ac(multiadd(x_vec));
		}

	private:
		// Input neurons number
		size_t input_nums;
		// The bias
		double bias;
		std::vector<double> weights;
		// Weights init type
		ParaInitType pit;
	};

}