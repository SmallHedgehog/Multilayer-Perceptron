# Multilayer-Perceptron
Multilayer perception implementation by C++

## How to define network structure for mlp

Using Json file to define mlp network structure

```Python
{
	"Name": "multilayer perception",
	"Data": {
		"output_num": 10,
		"type": "text",
		"file_path": "/bigData2/ycf/net.json"
	},
	"Inner": {
		"hidden_num": 5,
		"neuron_num": [
			20,
			25,
			30,
			25,
			20
		],
		"init_type": [
			"constant",
			"xavier",
			"constant",
			"xavier",
			"constant",
		],
		"type": "sigmoid"
	},
	"Loss": {
		"output_num": 10,
		"type": "softmax"
	}
}
```
### Data
using Json object to stand for input layer
* output_num: this item means having ten neurons input.
* type:
* file_path:
### Inner
inner product layer
* hidden_num:
* neuron_num:
* init_type:
* type:
### Loss
loss output
* output_num:
* type:
