# Multilayer-Perceptron
Multilayer perception implementation by C++

## How to define network structure for this mlp implemention

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
		"type": "softmax",
		"weights_init_type": "xavier"
	}
}
```
### Data
#### using Json object to stand for input layer
* output_num: The number of input layer
* type: The form of the data, text or image
* file_path: The absolute data path
### Inner
#### fully connected layer
* hidden_num: The number of the fully connected layer
* neuron_num: Each of hidden layers' number
* init_type: The initialization method of each hidden layers' weights
* type: Activatation function type
### Loss
#### loss output
* output_num: The number of output layer
* type: The loss function type
* weights_init_type: The initialization method of loss layer' weights

## How to difine optimization method for this mlp implemention

Using Json file to difine optimization method

```Python
{
	"net_path": "net.mlp"
	"solver": "SGD"
	"max_iter": 1000
}
```

* net_path: 
* solver: The optimization method
* max_iter:
