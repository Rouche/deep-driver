
## Notebook

Run `C:>jupyter-notebook`

## Functional API

- Multi-input models
- Multi-output models- Models with shared layers
- Models with non-sequential data flows

The Functional API is more functional

- Built around models that can be called (like functions)

- Train just like Sequential models

## Model Subclassing

- tf.keras.Model
- Can be trained
- Can encapsulate multiple layers
- Can be subclassed

### Model subclassing:
Subclass tf.keras.Model and only define your own forward pass imperatively.

Usefull for eager execution

## Custom Layers

Containes a call method wich defines the transformation applied to input to obtain the output.

Also contains a set of weights.

## Workarounds

- Error with GraphViz : "dot" with args ['-Tps', 'C:\Users\user\AppData\Local\Temp\tmprst0j6pn'] returned code: 1
	
	- Open CMD with administrator privileges (simply right click on the CMD and click "run as administrator")
    - Insert the command dot -c, this will configure the GraphViz plugins