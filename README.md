# Multilayer-Perceptron
Multilayer perception implementation by C++

# How to define network structure for mlp

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
		"type": "sigmoid"
	},
	"Loss": {
		"output_num": 10,
		"type": "softmax"
	}
}
```

* Data: using Json object to stand for input layer, the 'output_num' item means having ten neurons input.
* Inner: innerproduct layer
* Loss: loss output
