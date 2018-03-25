#pragma once

#include <vector>

#include "../neuron.h"


namespace mlp
{
	typedef struct DataParameter {
		size_t input_neuron_nums;
		char* type;
		char* file_path;
	}DP;

	typedef struct InnerParameter {
		// size_t inner_layer_nums;
		size_t inner_idx;
		// std::vector<size_t> inner_neuron_nums;
		// std::vector<char*> weights_init_type;
		size_t inner_neuron_nums;
		char* weights_init_type;
		char* activate_type;
	}IP;

	typedef struct LossParameter {
		size_t output_neuron_nums;
		size_t loss_type;
	}LP;

	// The network layer type
	typedef enum LayerType {
		DATA,
		INNER,
		LOSS,
		NONE
	}LT;

	typedef union Parameters {
		DataParameter data_parameters;
		InnerParameter inner_parameters;
		LossParameter loss_parameters;
	}UnionParameters;

	typedef struct LayerParameters {
		LayerType _lt;
		UnionParameters _up;
	}LayerParameters;

	class Layer {
	public:
		explicit Layer(const LayerParameters& para) {
			init_layer(para);
		}

		virtual ~Layer() {
			_neuron_nums = 0;
			_lt = LT::NONE;
			neurons.clear();
		}

		LT get_layer_type() const {
			return _lt;
		}

		void set_layer_type(LT lt) {
			_lt = lt;
		}

		size_t get_neuron_nums() const {
			return _neuron_nums;
		}

		void set_neuron_nums(size_t neuron_nums) {
			assert(neuron_nums > 0);
			_neuron_nums = neuron_nums;
			neurons.clear();
			neurons.resize(_neuron_nums);
		}

		const std::vector<mlp::Neuron>& get_neurons() const {
			return neurons;
		}

		virtual bool get_backward_down() = 0;
		virtual void SetUp() = 0;
		virtual void Forward() = 0;
		virtual void Backward() = 0;

	private:
		void init_layer(const LayerParameters& para) {
			switch (para._lt) {
			case LT::DATA:
				_neuron_nums = para._up.data_parameters.input_neuron_nums;
				_lt = LT::DATA;
				break;
			case LT::INNER:
				_neuron_nums = para._up.inner_parameters.inner_neuron_nums;
				_lt = LT::INNER;
				break;
			case LT::LOSS:
				_neuron_nums = para._up.loss_parameters.output_neuron_nums;
				_lt = LT::LOSS;
				break;
			}
			neurons.clear();
			neurons.resize(_neuron_nums);
			_lp = para;
		}

	protected:
		size_t _neuron_nums;
		LT _lt;
		std::vector<mlp::Neuron*> neurons;

		LayerParameters _lp;
	};
}
