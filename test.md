why the TensorFlow Federated tutirials use:

```python
    def batch_flatten(element) :
        return collections.OrderedDict(x=tf.reshape(element['pixels'], [-1, 784]),
          y=tf.reshape(element['label'], [-1, 1]))
```

instead of

```python
    def batch_flatten(element) :
        return {"x": tf.reshape(element['pixels'], [-1]), "y": element['label']}
```

when preprocessing data?